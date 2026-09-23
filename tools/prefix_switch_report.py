#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count prefix switches: requests that got *smaller* than the one before.

Why this is a separate measurement
----------------------------------
A KV cache is matched by prefix. If the payload of request N+1 is a strict extension of
request N, everything cached still applies. If the payload instead gets *shorter* (or its
earlier messages change), the cached prefix is invalidated and the difference is
recomputed.

So "how often does the request get smaller right after growing" is a direct, cheap proxy
for "how often does something invalidate the prefix" — no knowledge of cache internals
required. This is the number quoted for the three backfill designs: the mid-turn flip
(v1) and the per-segment switch (v2) both change the payload's shape, while the
per-message rule (v3) never changes a past message, so the prefix only ever grows.

Criterion
---------
For consecutive requests in one session, ordered by time:

    delta = totalTokens(N+1) - totalTokens(N)
    a "shrinking step" is  delta <= -MIN_DROP   (default 10,000 tokens)

⚠️ **Compaction must be excluded.** A context compaction legitimately cuts the payload
from hundreds of thousands of tokens to a few thousand. Those steps are not prefix
switches caused by the client; they are the client discarding history on purpose. Any
step within `--compact-window` seconds of a `compaction/*` event is skipped and counted
separately.

Usage
-----
    python3 prefix_switch_report.py
    python3 prefix_switch_report.py --session 1a2b3c4d
    python3 prefix_switch_report.py --since 2026-09-23T12:41
    python3 prefix_switch_report.py --min-drop 10000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SESSIONS = Path(os.environ.get("DSH_SESSIONS_DIR", Path.home() / ".dsh" / "sessions"))
ARCHIVE_NAME = "session.v3.jsonl.zstd"
TZ = timezone(timedelta(hours=8))


def _read(f: Path):
    raw = subprocess.run(["zstd", "-dc", str(f)], capture_output=True).stdout.decode("utf-8", "replace")
    for ln in raw.splitlines():
        if '"assistant/message"' not in ln and "compaction" not in ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        yield e


def scan():
    """-> (steps, compactions). A step is one request's token total."""
    steps, comps = [], []
    if not SESSIONS.exists():
        return steps, comps
    for slug in sorted(os.listdir(SESSIONS)):
        sd = SESSIONS / slug
        if not sd.is_dir():
            continue
        for d in sorted(os.listdir(sd)):
            f = sd / d / ARCHIVE_NAME
            if not f.exists():
                continue
            for e in _read(f):
                t = e.get("type") or ""
                ts = (e.get("time") or 0) / 1000
                when = datetime.fromtimestamp(ts, TZ) if ts else None
                if t.startswith("compaction/"):
                    comps.append((d, when, t))
                    continue
                if t != "assistant/message":
                    continue
                u = (e.get("data") or {}).get("usage") or {}
                tot = u.get("totalTokens")
                if not tot:
                    continue
                steps.append({"session": d, "at": when, "total": tot,
                              "turn": (e.get("data") or {}).get("turn"),
                              "step": (e.get("data") or {}).get("step")})
    for s in steps:
        s["at"] = s["at"] or datetime.min.replace(tzinfo=TZ)
    steps.sort(key=lambda s: (s["session"], s["at"]))
    return steps, comps


def report(steps, comps, min_drop: int, compact_window: float, detail: bool):
    by_comp = defaultdict(list)
    for sess, when, _t in comps:
        if when:
            by_comp[sess].append(when)

    def near_compaction(sess, when) -> bool:
        for c in by_comp.get(sess, []):
            if abs((when - c).total_seconds()) <= compact_window:
                return True
        return False

    prev = {}
    shrinks, skipped = [], 0
    total_steps = 0
    for s in steps:
        total_steps += 1
        p = prev.get(s["session"])
        prev[s["session"]] = s
        if not p:
            continue
        delta = s["total"] - p["total"]
        if delta > -min_drop:
            continue
        if near_compaction(s["session"], s["at"]):
            skipped += 1
            continue
        shrinks.append({"session": s["session"], "at": s["at"], "before": p["total"],
                        "after": s["total"], "delta": delta, "turn": s["turn"]})

    print("=== prefix switches (request got smaller) ===")
    print(f"  criterion : totalTokens(N+1) - totalTokens(N) <= -{min_drop:,}")
    print(f"  excluded  : steps within {compact_window:.0f}s of a compaction event "
          f"({skipped} skipped - those are the client discarding history, not a switch)")
    print(f"  requests scanned: {total_steps:,}")
    print(f"  shrinking steps : {len(shrinks)}")
    if shrinks:
        print()
        for r in shrinks:
            t = r["at"].strftime("%m-%d %H:%M:%S")
            print(f"    {t}  turn {str(r['turn']):<5} {r['before']:>9,} -> {r['after']:>9,}"
                  f"  ({r['delta']:+,})  {r['session'][:24]}")
    return shrinks


def main() -> int:
    ap = argparse.ArgumentParser(description="prefix switches: requests that got smaller")
    ap.add_argument("--min-drop", type=int, default=10000,
                    help="a shrinking step must lose at least this many tokens (default 10000)")
    ap.add_argument("--compact-window", type=float, default=1800.0,
                    help="ignore steps within N seconds of a compaction (default 1800)")
    ap.add_argument("--session", default="", help="only sessions whose id contains this")
    ap.add_argument("--since", default="", help="only steps after this local time")
    ap.add_argument("--sessions", default="", help="archive root (default ~/.dsh/sessions)")
    args = ap.parse_args()

    global SESSIONS
    if args.sessions:
        SESSIONS = Path(args.sessions)

    steps, comps = scan()
    if args.session:
        steps = [s for s in steps if args.session in s["session"]]
        comps = [c for c in comps if args.session in c[0]]
    if args.since:
        try:
            cut = datetime.fromisoformat(args.since).replace(tzinfo=TZ)
        except Exception:
            print(f"cannot parse --since {args.since} (want 2026-09-23T12:41)")
            return 2
        steps = [s for s in steps if s["at"] >= cut]
        print(f"(only steps after {cut.strftime('%m-%d %H:%M')})")
    if args.session:
        print(f"(only sessions containing {args.session})")

    report(steps, comps, args.min_drop, args.compact_window, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
