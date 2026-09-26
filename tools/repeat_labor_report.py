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

Two numbers come out of that, and only one of them is usable as a trend:

  * **groups** - keys occurring in more than one segment. A group is a property of the
    whole history, so the total keeps growing as the archive grows. Useful as a stock
    description, useless as a trend.
  * **returning calls per 1000 calls** - a call whose key already appeared in an earlier
    segment *of the same session*. Denominator = calls that day, so it does not drift
    upward with archive size. **Use this one to judge the side effect.**

Two grouping choices matter, and both were wrong in the first draft of this tool:

  * **per session, not global** - two windows reading the same file is two agents doing
    their own work, not one agent repeating itself (global grouping inflated the count
    ~6x, 1524 vs 244);
  * **polling excluded** - `ls`, `ps`, `git status`, health checks are *supposed* to
    repeat. That filter matched nothing until 2026-09-23 (a literal `\b` instead of a
    word boundary, a `^` anchor, and a curl branch requiring `/dev/null`).

A third trap, worth recording because it produced a confidently wrong reading: an earlier
version printed a per-day table where a group's "spans days" credit went to every day
except its last. **The last day of any group could therefore never receive credit**, so
new repeats were booked against old rows - a table that says "it is getting worse" no
matter what happens. It has been replaced by the per-day rate above.

Recommended usage
-----------------
Fix a baseline, then re-run over the window since the last check and compare **rates**:

    baseline (one window, before the change): 50 cross-segment groups
    every few days:  --since <last check time>   -> look at groups/returning per 1000 calls
    do NOT compare group totals across runs

Read it as a trend, not a verdict: reading the same file twice is not automatically waste
(the file may have changed, or the second read may be a deliberate re-check). This counts
**traces**; a human decides what they mean.

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

# Which tools count as "read / run" (repeating those may be waste)
READ_TOOLS = {"read", "read_file", "view", "cat", "Read"}
RUN_TOOLS = {"bash", "shell", "exec", "run", "terminal"}


def _load_events(path: Path):
    """One session archive -> [events], keeping only the types we judge on."""
    raw = subprocess.run(["zstd", "-dc", str(path)], capture_output=True).stdout.decode("utf-8", "replace")
    out = []
    for ln in raw.split("\n"):  # 用 \n 切：splitlines() 会在 U+2028 上误切
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
    """Compress one tool call into a comparable key."""
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
    calls = []          # (time, seg_id, session, name, key, returning)
    for slug in sorted(os.listdir(SESSIONS)) if SESSIONS.exists() else []:
        sd = SESSIONS / slug
        if not sd.is_dir():
            continue
        for d in sorted(os.listdir(sd)):
            f = sd / d / "session.v3.jsonl.zstd"
            if not f.exists():
                continue
            seg = 0
            # `returning` = this call redoes something an EARLIER SEGMENT of the same
            # session already did. Computed per call on purpose: a per-group table
            # credits "spans days" to every day except the last, so the last day of any
            # group can never receive credit and new repeats get booked against old rows
            # (a table that reads "getting worse" whatever happens). Group totals are
            # also a property of the whole history and grow with the archive; a per-call
            # incidence has a stable denominator and can be read per day.
            closed: set = set()          # keys used by earlier segments
            seg_keys: set = set()        # keys used by the current segment
            for e in _load_events(f):
                if e.get("type") == "assistant/message":
                    if _is_final_reply(e):
                        closed |= seg_keys      # seal this segment into history
                        seg_keys = set()
                        seg += 1
                    continue
                da = e.get("data") or {}
                ts = (e.get("time") or 0) / 1000
                when = datetime.fromtimestamp(ts, TZ) if ts else None
                if since and when and when < since:
                    continue
                name = da.get("name") or "?"
                key = _key(name, da.get("arguments"))
                returning = key in closed
                seg_keys.add(key)
                calls.append((when, f"{d}#{seg}", d, name, key, returning))
    return calls


# Commands that are *meant* to repeat: state polling, listing, health checks.
#
# Fixed 2026-09-23. The first version of this filter never matched anything:
#   (1) the pattern ended in a literal backslash-b instead of a word boundary, so even
#       `ls -la` failed to match - 0 of 1051 repeated run commands were excluded;
#   (2) it was anchored at `^`, missing the common `cd x && ls -la` form;
#   (3) its curl branch required `/dev/null` in a fixed position, missing health checks.
# The consequence: "polling excluded: 0" was printed as if it were a finding, when it
# actually meant the exclusion step never ran.
#
# New rule, auditable: split the command on `&&` / `||` / `;` / `|` and require every
# segment to be state-reading. One segment doing real work makes the whole command
# non-trivial: `cd /x && ls -la` is polling, `python x.py | tail -3` is not.
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
    for when, seg, sess, name, key, returning in calls:
        by_key[key].append((when, seg, sess, name))

    # Group per session, not globally: two windows each reading the same file is two
    # agents doing their own work, not one agent repeating itself. Global grouping
    # inflated the count ~6x (1524 vs 244). Both are printed for comparability.
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

    n_calls = len(calls)
    n_ret = sum(1 for c in calls if c[5])
    rate = 1000.0 * n_ret / max(n_calls, 1)
    print("=== repeated labour across segments ===")
    print(f"  unit: same (tool + arguments) recurring in a different segment")
    print(f"  tool calls scanned: {n_calls:,}")
    print(f"    - within a segment            : {len(same)}   (re-checking, not counted)")
    print(f"    - **cross-segment, same window  : {len(per_window)}**   <- stock, not a trend")
    print(f"    - cross-segment, global       : {len(global_cross)}   (legacy grouping, kept for comparison)")
    print(f"  **returning calls per 1000 (use THIS as the trend): {rate:.1f}**"
          f"({n_ret}/{n_calls})")

    # Separate commands that are supposed to repeat; polling is not repeated labour.
    meaningful = [(k, h) for k, h in per_window if not is_trivial(k[1])]
    trivial = [(k, h) for k, h in per_window if is_trivial(k[1])]
    print(f"    - after removing polling      : **{len(meaningful)}**")
    print(f"    - polling commands excluded   : {len(trivial)}")
    if trivial:
        print(f"      (this filter matched nothing before 2026-09-23)")

    kinds = defaultdict(int)
    for (sess, key), _ in meaningful:
        kinds[key[0]] += 1
    if kinds:
        print(f"  by type: {dict(kinds)}")

    # Reworked 2026-09-23: the previous per-day table could not judge the side effect.
    # It credited "spans days" to every day except the group's last, so new repeats were
    # booked against old rows (one real group spans 09-11..09-23 and was credited to
    # 09-11) - a table that always reads "getting worse". Now: per-day incidence.
    per_day = defaultdict(lambda: [0, 0])       # day -> [calls, returning calls]
    for when, seg, sess, name, key, returning in calls:
        if not when:
            continue
        d = when.strftime("%m-%d")
        per_day[d][0] += 1
        if returning:
            per_day[d][1] += 1
    print("\n  per day (returning / calls = per 1000):")
    for d in sorted(per_day)[-12:]:
        tot, ret = per_day[d]
        r = 1000.0 * ret / max(tot, 1)
        bar = "█" * min(40, int(r / 5))
        print(f"    {d}  {ret:>4}/{tot:<5} = {r:>5.1f}  {bar}")
    print("    (the patch landed 2026-09-23 12:40; compare windows after it against windows before it)")
    if detail:
        print("\n=== cross-segment repeats (same window, polling excluded) ===")
        for (sess, key), hits in sorted(meaningful, key=lambda kv: -len(kv[1]))[:40]:
            print(f"\n  [{hits[0][3]}] {sess[:16]}  {str(key[1])[:100]}")
            for when, seg, _sess, name in hits[:8]:
                t = when.strftime("%m-%d %H:%M") if when else "?"
                print(f"      {t}  seg {seg.split(chr(35))[-1]:<3}")
            if len(hits) > 8:
                print(f"      ... {len(hits)-8} more")
    return len(meaningful), len(same)


def main() -> int:
    ap = argparse.ArgumentParser(description="repeated labour across segments")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--since", default="", help="only after this local time, e.g. 2026-09-23T12:40")
    ap.add_argument("--session", default="", help="only sessions whose id contains this")
    args = ap.parse_args()
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
