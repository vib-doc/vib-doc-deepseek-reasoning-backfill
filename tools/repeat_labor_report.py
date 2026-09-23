#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure "repeated labour across segments" — the side effect to watch.

Why this exists
---------------
Narrowing the reasoning backfill (`patches/patch_backfill.py`) means the model can no
longer re-read its own **non-tool** reasoning from earlier segments. The plausible
cost is *repeated labour*: re-reading a file it already read, re-running a command it
already ran, re-deriving something it already concluded.

That cost is measurable, which is the only reason it is acceptable to take the risk.

Method
------
A **segment** is everything after the last *final reply* (an assistant message with
visible text and no tool call). For every tool call we build a key:

  * read-like tools  -> ("read", path)
  * run-like tools   -> ("run",  normalized command text)
  * anything else    -> (tool name, sorted arguments)

A key seen in **more than one segment** is a cross-segment repeat. A key seen twice
inside one segment is normal re-checking and is reported separately.

Noise control
-------------
Polling commands (`ls`, `ps`, `git status`, health checks, `tail`, …) are *supposed*
to repeat. They are counted separately and excluded from the headline number —
otherwise the metric is dominated by them and stops meaning anything.

Read it as a trend, not a verdict
---------------------------------
Reading the same file twice is not automatically waste: the file may have changed, or
the second read may be a deliberate re-check. This counts **traces**; a human decides
what they mean. What matters is the direction after deploying the change.

Usage
-----
    python3 repeat_labor_report.py
    python3 repeat_labor_report.py --detail
    python3 repeat_labor_report.py --since 2026-09-23T12:40
    python3 repeat_labor_report.py --session 1a2b3c4d    # one session id substring
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SESSIONS = Path(os.environ.get("DSH_SESSIONS_DIR", Path.home() / ".dsh" / "sessions"))
ARCHIVE_NAME = "session.v3.jsonl.zstd"
TZ = timezone(timedelta(hours=8))

READ_TOOLS = {"read", "read_file", "view", "cat", "Read"}
RUN_TOOLS = {"bash", "shell", "exec", "run", "terminal"}

# Commands that are *meant* to be repeated (status polling, listing, health checks).
TRIVIAL = re.compile(
    r"^(ls|pwd|ps|top|df|du|date|whoami|id|env|which|git status|git log|git diff|"
    r"tail|head|wc|sleep|echo|curl -s? ?-?m? ?[0-9]* ?-?o? ?/dev/null)\b"
)


def _read(f: Path):
    raw = subprocess.run(["zstd", "-dc", str(f)], capture_output=True).stdout.decode("utf-8", "replace")
    for ln in raw.splitlines():
        if '"tool/call"' not in ln and '"assistant/message"' not in ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("type") in ("tool/call", "assistant/message"):
            yield e


def _is_final_reply(e) -> bool:
    parts = ((e.get("data") or {}).get("message") or {}).get("content") or []
    has_text = any(p.get("type") == "text" and str(p.get("text") or "").strip() for p in parts)
    has_call = any(p.get("type") == "tool-call" for p in parts)
    return has_text and not has_call


def _key(name: str, args):
    try:
        a = json.loads(args) if isinstance(args, str) else (args or {})
    except Exception:
        a = {"_raw": str(args)}
    if name in READ_TOOLS:
        p = a.get("path") or a.get("file_path") or a.get("file") or a.get("_raw")
        return ("read", str(p))
    if name in RUN_TOOLS:
        c = re.sub(r"\s+", " ", str(a.get("command") or a.get("_raw") or "")).strip()
        return ("run", c[:400])
    return (name, json.dumps(a, sort_keys=True, ensure_ascii=False)[:400])


def scan(since: datetime | None = None):
    calls = []
    if not SESSIONS.exists():
        return calls
    for slug in sorted(os.listdir(SESSIONS)):
        sd = SESSIONS / slug
        if not sd.is_dir():
            continue
        for d in sorted(os.listdir(sd)):
            f = sd / d / ARCHIVE_NAME
            if not f.exists():
                continue
            seg = 0
            for e in _read(f):
                if e.get("type") == "assistant/message":
                    if _is_final_reply(e):
                        seg += 1
                    continue
                da = e.get("data") or {}
                ts = (e.get("time") or 0) / 1000
                when = datetime.fromtimestamp(ts, TZ) if ts else None
                if since and when and when < since:
                    continue
                name = da.get("name") or "?"
                calls.append((when, f"{d}#{seg}", d, name, _key(name, da.get("arguments"))))
    return calls


def is_trivial(key) -> bool:
    return key[0] == "run" and bool(TRIVIAL.match(str(key[1]).strip()))


def report(calls, detail: bool = False) -> None:
    by_key = defaultdict(list)
    for when, seg, sess, name, key in calls:
        by_key[key].append((when, seg, sess, name))

    cross, same = [], []
    for key, hits in by_key.items():
        if len(hits) < 2:
            continue
        (cross if len({h[1] for h in hits}) > 1 else same).append((key, hits))

    meaningful = [(k, h) for k, h in cross if not is_trivial(k)]
    trivial = [(k, h) for k, h in cross if is_trivial(k)]

    print("=== repeated labour across segments ===")
    print(f"  tool calls scanned: {len(calls):,}")
    print(f"  repeat groups (>=2 calls with the same key): {len(cross) + len(same)}")
    print(f"    · across segments : {len(cross)}")
    print(f"    · within a segment: {len(same)}   (normal re-checking, not counted)")
    print(f"    · polling commands excluded: {len(trivial)}")
    print(f"    => **headline: {len(meaningful)} cross-segment repeat groups**")

    kinds = defaultdict(int)
    for key, _ in meaningful:
        kinds[key[0]] += 1
    if kinds:
        print(f"  by type: {dict(kinds)}")

    per_day = defaultdict(lambda: [0, 0])
    for key, hits in meaningful:
        days = sorted({h[0].strftime("%m-%d") for h in hits if h[0]})
        for d in days:
            per_day[d][1] += 1
        for d in days[:-1]:
            per_day[d][0] += 1
    print("\n  per day (groups spanning days / groups seen that day):")
    for d in sorted(per_day)[-10:]:
        a, b = per_day[d]
        print(f"    {d}  spanning {a:>4} / seen {b:>4}")

    if detail:
        print("\n=== cross-segment repeats ===")
        for key, hits in sorted(meaningful, key=lambda kv: -len(kv[1]))[:40]:
            print(f"\n  [{hits[0][3]}] {str(key[1])[:100]}")
            for when, seg, sess, name in hits[:8]:
                t = when.strftime("%m-%d %H:%M") if when else "?"
                print(f"      {t}  seg {seg.split('#')[-1]:<3} {sess[:16]}")
            if len(hits) > 8:
                print(f"      ... {len(hits)-8} more")


def main() -> int:
    ap = argparse.ArgumentParser(description="repeated labour across segments")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--since", default="", help="local time, e.g. 2026-09-23T12:40")
    ap.add_argument("--session", default="", help="only sessions whose id contains this")
    ap.add_argument("--sessions", default="", help="archive root (default ~/.dsh/sessions)")
    args = ap.parse_args()

    global SESSIONS
    if args.sessions:
        SESSIONS = Path(args.sessions)

    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since).replace(tzinfo=TZ)
        except Exception:
            print(f"cannot parse --since {args.since} (want 2026-09-23T12:40)")
            return 2
        print(f"(only after {since.strftime('%m-%d %H:%M')})")

    calls = scan(since)
    if args.session:
        calls = [c for c in calls if args.session in c[2]]
        print(f"(only sessions containing {args.session})")
    report(calls, detail=args.detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
