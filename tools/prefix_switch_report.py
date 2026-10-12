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

⚠️ **Compaction must be excluded, and the exclusion is positional, not a time window.**
A context compaction legitimately cuts the payload from hundreds of thousands of tokens to
a few thousand (one recorded case: 792k -> 16k). That shrink appears in the **first
request after the compaction**, however long the user was idle in between — so that is
exactly the step skipped, and no window constant is involved.

Both window choices are wrong in one direction. Too wide (30 min) swallows real flips that
happen after a compaction; too narrow (2 min) misses the legitimate shrink when the user
walks away for ten minutes after one. Measured on the deployment: all 22 exclusions fell
within 87s of a compaction **and all 22 were the first request after it**, so the exact
test reproduces every case a window got right.

⚠️ **The newest line is provisional.** Archives are append-only and a turn in progress
keeps appending steps, so the most recent boundary's figure can still change while the
turn is being written. Re-run after the turn ends before quoting it.

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

# Archive-selection rule lives in tools/dsh_archive.py (2026-09-30):
# never hardcode an archive generation — see that file's docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh_archive import latest_for, iter_archives  # noqa: E402

SESSIONS = Path(os.environ.get("DSH_SESSIONS_DIR", Path.home() / ".dsh" / "sessions"))
TZ = timezone(timedelta(hours=8))


def _read(f: Path):
    raw = subprocess.run(["zstd", "-dc", str(f)], capture_output=True).stdout.decode("utf-8", "replace")
    for ln in raw.split("\n"):  # 用 \n 切：splitlines() 会在 U+2028 上误切
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
            # newest generation (v4 > v3 > unversioned), never a hardcoded one
            f = latest_for(sd / d)
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


def report(steps, comps, min_drop: int, detail: bool):
    by_comp = defaultdict(list)
    for sess, when, _t in comps:
        if when:
            by_comp[sess].append(when)

    # Which steps exist per session, for the "is this the first request after a
    # compaction?" test below.
    times_by_sess = defaultdict(list)
    for s_ in steps:
        times_by_sess[s_["session"]].append(s_["at"])

    def is_first_request_after_compaction(sess, when) -> bool:
        """Exact criterion; the causal position, with no arbitrary time window.

        A compaction lowers the payload once, and that shows up in the **first request
        after it** however long the user was idle in between. So instead of "within N
        seconds of a compaction" we ask: is there any request between the latest
        compaction and this one? If not, this is that first request.

        ⚠️ Why this replaced a window (a review finding, 2026-09-23): the window was
        1800s, which would also swallow *real* flips happening up to half an hour after a
        compaction. Narrowing it to 120s (also suggested) fixes that direction but opens
        the opposite one: after a compaction, a user who walks away for ten minutes
        produces that same legitimate shrink later than 120s, and it would be counted as
        a prefix switch. Measured on the deployment: all 22 exclusions are within 87s of
        a compaction **and all 22 are the first request after it** - so the exact test
        reproduces every case the window got right, without the window.
        """
        latest = None
        for c in by_comp.get(sess, []):
            if c <= when and (latest is None or c > latest):
                latest = c
        if latest is None:
            return False
        for t in times_by_sess.get(sess, []):
            if latest < t < when:
                return False
        return True

    prev = {}
    shrinks, skipped = [], 0
    last_compaction_skip = []
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
        if is_first_request_after_compaction(s["session"], s["at"]):
            skipped += 1
            last_compaction_skip.append(s["at"])
            continue
        shrinks.append({"session": s["session"], "at": s["at"], "before": p["total"],
                        "after": s["total"], "delta": delta, "turn": s["turn"]})

    print("=== prefix switches (request got smaller) ===")
    print(f"  criterion : totalTokens(N+1) - totalTokens(N) <= -{min_drop:,}")
    print(f"  excluded  : the first request after a compaction event "
          f"({skipped} skipped - that is the client discarding history, not a switch)")
    print(f"  requests scanned: {total_steps:,}")
    print(f"  shrinking steps : {len(shrinks)}")
    if shrinks:
        print()
        # ⚠️ The newest boundary is provisional: archives are append-only and a turn in
        # progress keeps appending steps (and its current step's usage can still be
        # rewritten). Measured 2026-09-23: turn 704 read -13,438 once the turn settled,
        # where a mid-write read of the same boundary gave a smaller figure. Re-run after
        # the turn ends before quoting the last line.
        print("    (the last line is provisional if a turn is still running - re-run later)")
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
    # (no --compact-window: the exclusion is exact, see is_first_request_after_compaction)
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

    report(steps, comps, args.min_drop, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
