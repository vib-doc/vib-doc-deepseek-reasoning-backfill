#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repeated labour across segments - the side effect to watch.

Why this exists
---------------
Narrowing the reasoning backfill (`patches/patch_backfill.py`) means the model can no
longer re-read its own **non-tool** reasoning from earlier segments. The plausible cost
is *repeated labour*: re-reading a file it already read, re-running a command it already
ran, re-deriving something it already concluded. That cost is measurable, which is the
only reason it is acceptable to take the risk.

Note the precise lifetime of reasoning under the patch: it is kept for the tool steps of
the turn that produced it, and stripped from every earlier turn. **Reasoning lives one
turn.** So the thing to watch is what the model has to redo *after* a turn ends.

Method
------
A **segment** is everything after the last *final reply* (an assistant message with
visible text and no tool call). For every tool call we build a key:

  * read-like tools  -> ("read", path)
  * run-like tools   -> ("run",  normalized command text)
  * anything else    -> (tool name, sorted arguments)

A key seen in **more than one segment, within the same session** is a cross-segment
repeat. Two grouping choices matter:

  * **per session, not global** - two windows reading the same file is two agents doing
    their own work, not one agent repeating itself (global grouping inflated the count
    ~6x). Both numbers are printed so older observations stay comparable;
  * **polling excluded** - `ls`, `ps`, `git status`, health checks are *supposed* to
    repeat. This filter matched nothing until 2026-09-23 (a literal `\b` instead of a
    word boundary, a `^` anchor, and a curl branch requiring `/dev/null`).

Read it as a trend, not a verdict
---------------------------------
Reading the same file twice is not automatically waste: the file may have changed, or the
second read may be a deliberate re-check. This counts **traces**; a human decides what
they mean. What matters is the direction after deploying the change.

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

SESSIONS = Path.home() / ".dsh" / "sessions"
TZ = timezone(timedelta(hours=8))

# 哪些工具算"读/跑"（重复它们才可能是浪费）
READ_TOOLS = {"read", "read_file", "view", "cat", "Read"}
RUN_TOOLS = {"bash", "shell", "exec", "run", "terminal"}


def _load_events(path: Path):
    """一条会话档案 → [事件]。只留判定要用的类型。"""
    raw = subprocess.run(["zstd", "-dc", str(path)], capture_output=True).stdout.decode("utf-8", "replace")
    out = []
    for ln in raw.splitlines():
        if '"tool/call"' not in ln and '"assistant/message"' not in ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        t = e.get("type")
        if t not in ("tool/call", "assistant/message"):
            continue
        out.append(e)
    return out


def _is_final_reply(e) -> bool:
    parts = ((e.get("data") or {}).get("message") or {}).get("content") or []
    has_text = any(p.get("type") == "text" and str(p.get("text") or "").strip() for p in parts)
    has_call = any(p.get("type") == "tool-call" for p in parts)
    return has_text and not has_call


def _key(name: str, args: str):
    """把一次调用压成一个可比较的键。"""
    try:
        a = json.loads(args) if isinstance(args, str) else (args or {})
    except Exception:
        a = {"_raw": str(args)}
    if name in READ_TOOLS:
        p = a.get("path") or a.get("file_path") or a.get("file") or a.get("_raw")
        return ("read", str(p))
    if name in RUN_TOOLS:
        c = str(a.get("command") or a.get("_raw") or "")
        c = re.sub(r"\s+", " ", c).strip()
        return ("run", c[:400])
    return (name, json.dumps(a, sort_keys=True, ensure_ascii=False)[:400])


def scan(since: datetime | None = None):
    calls = []          # (time, seg_id, session, name, key)
    for slug in sorted(os.listdir(SESSIONS)) if SESSIONS.exists() else []:
        sd = SESSIONS / slug
        if not sd.is_dir():
            continue
        for d in sorted(os.listdir(sd)):
            f = sd / d / "session.v3.jsonl.zstd"
            if not f.exists():
                continue
            seg = 0
            for e in _load_events(f):
                if e.get("type") == "assistant/message":
                    if _is_final_reply(e):
                        seg += 1
                    continue
                da = e.get("data") or {}
                ts = (e.get("time") or 0) / 1000
                when = datetime.fromtimestamp(ts, TZ) if ts else None
                if since and when and when < since:
                    continue
                calls.append((when, f"{d}#{seg}", d, da.get("name") or "?", _key(da.get("name") or "?", da.get("arguments"))))
    return calls


# Commands that are *meant* to repeat: state polling, listing, health checks.
#
# Fixed 2026-09-23. The first version of this filter never matched anything:
#   (1) the pattern ended in a literal `\b` instead of a word boundary, so even
#       `ls -la` failed to match - 0 of 1051 repeated run commands were excluded;
#   (2) it was anchored at `^`, missing the common `cd x && ls -la` form;
#   (3) its curl branch required `/dev/null` in a fixed position, missing health checks.
# The headline consequence: "polling excluded: 0" was printed as if it were a finding
# when it actually meant the exclusion step never ran.
#
# New rule, auditable: split the command on `&&` / `||` / `;` / `|` and require **every**
# segment to be state-reading. One segment doing real work (python / npm / git commit)
# makes the whole command non-trivial: `cd /x && ls -la` is polling, `python x.py | tail -3` is not.
_POLL_SEG = re.compile(
    r"^\s*(cd|ls|pwd|ps|top|df|du|date|whoami|id|env|which|tail|head|wc|sleep|echo|true|"
    r"stat|lsof|pgrep|nc\s+-z|git\s+(status|log|diff|rev-parse|branch)|launchctl\s+list)\b"
)
# Health/status endpoints: curl against the local host or a local service, any flags.
_HEALTH_CURL = re.compile(
    r"^\s*curl\b[^\n]*?(/health|/api/status|/status\b|127\.0\.0\.1|localhost|:8000|:3001)"
)
_SEG_SPLIT = re.compile(r"&&|\|\||;|\|")


def is_trivial(key) -> bool:
    if key[0] != "run":
        return False
    segs = [s for s in _SEG_SPLIT.split(str(key[1])) if s.strip()]
    if not segs:
        return False
    return all(_POLL_SEG.match(s) or _HEALTH_CURL.match(s) for s in segs)


def report(calls, detail: bool = False):
    by_key = defaultdict(list)
    for when, seg, sess, name, key in calls:
        by_key[key].append((when, seg, sess, name))

    # Group per session. The first version grouped keys globally, so two windows each
    # reading the same file once counted as one 'repeat' - that is two agents doing
    # their own work, not one agent repeating itself. It inflated the baseline ~6x
    # (1524 vs 243). Both numbers are printed so older observations stay comparable.
    per_window, global_cross = [], []
    gmap = defaultdict(list)
    same = []
    for key, hits in by_key.items():
        if len(hits) >= 2 and len({h[1] for h in hits}) == 1:
            same.append((key, hits))          # all in one segment = re-checking, not a repeat
        by_sess = defaultdict(list)
        for h in hits:
            by_sess[h[2]].append(h)
        for sess, hs in by_sess.items():
            if len(hs) >= 2 and len({h[1] for h in hs}) > 1:
                per_window.append(((sess, key), hs))
        if len(hits) >= 2 and len({h[1] for h in hits}) > 1:
            global_cross.append((key, hits))

    print("=== repeated labour across segments ===")
    print("  unit: same (tool + arguments) recurring in different segments")
    print(f"  tool calls scanned: {len(calls):,}")
    print(f"    - within a segment          : {len(same)}   (re-checking, not counted)")
    print(f"    - **cross-segment, same window: {len(per_window)}**   <- headline")
    print(f"    - cross-segment, global     : {len(global_cross)}   (legacy grouping, kept for comparison)")

    # 把"本来就该反复跑"的命令单独算：轮询类的重复不是重复劳动
    meaningful = [(k, h) for k, h in per_window if not is_trivial(k[1])]
    trivial = [(k, h) for k, h in per_window if is_trivial(k[1])]
    print(f"    - after removing polling    : **{len(meaningful)}**")
    print(f"    - polling commands excluded : {len(trivial)}")
    if trivial:
        print("      (this filter matched nothing before 2026-09-23)")

    kinds = defaultdict(int)
    for (sess, key), _ in meaningful:
        kinds[key[0]] += 1
    if kinds:
        print(f"  by type: {dict(kinds)}")

    # Per day: the backfill patch landed 2026-09-23 12:40 local; only later days compare.
    per_day = defaultdict(lambda: [0, 0])
    for (sess, key), hits in meaningful:
        days = sorted({h[0].strftime("%m-%d") for h in hits if h[0]})
        for d in days:
            per_day[d][1] += 1
        if len(days) > 1:
            for d in days[:-1]:
                per_day[d][0] += 1
    print("\n  per day (groups spanning days / groups seen that day):")
    for d in sorted(per_day)[-10:]:
        a, b = per_day[d]
        print(f"    {d}  spanning {a:>4} / seen {b:>4}")
    if detail:
        print("\n=== cross-segment repeats (same window, polling excluded) ===")
        for (sess, key), hits in sorted(meaningful, key=lambda kv: -len(kv[1]))[:40]:
            print(f"\n  [{hits[0][3]}] {sess[:16]}  {str(key[1])[:100]}")
            for when, seg, _sess, name in hits[:8]:
                t = when.strftime("%m-%d %H:%M") if when else "?"
                print(f"      {t}  seg {seg.split(chr(35))[-1]:<3}")
            if len(hits) > 8:
                print(f"      …还有 {len(hits)-8} 次")
    return len(cross), len(same)


def main() -> int:
    ap = argparse.ArgumentParser(description="跨轮重复劳动（形状 B）")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--since", default="", help="只看这个本地时刻之后，如 2026-09-23T12:40")
    ap.add_argument("--session", default="", help="只看会话 id 含这个字串的（如 REDACTED-SESSION）")
    args = ap.parse_args()
    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since).replace(tzinfo=TZ)
        except Exception:
            print(f"看不懂 --since {args.since}（要 2026-09-23T12:40 这种）")
            return 2
        print(f"（只看 {since.strftime('%m-%d %H:%M')} 之后）")
    calls = scan(since)
    if args.session:
        calls = [c for c in calls if args.session in c[2]]
        print(f"（只看会话含 {args.session} 的）")
    report(calls, detail=args.detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
