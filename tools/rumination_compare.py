#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the reasoning chains of swallowed vs. successful responses.

This produces the strongest result in the accompanying note, and the one most worth
verifying independently: **at the level of the visible chain, a swallowed attempt and a
retry of the same step that succeeded are indistinguishable.**

Two comparisons, and the difference between them is the whole point
-------------------------------------------------------------------
1. **All data** — every swallowed chain vs. every successful chain. Differences here
   describe *which kinds of conversations* swallow more often (they are longer,
   more self-referential, more repetitive). They do NOT predict any individual attempt.
2. **Matched within a step** — for a step that was swallowed and then succeeded on
   retry, compare the failed chain to the successful one. Same input, same context,
   different sample. Anything that still separates them would be a warning sign; in the
   measured deployment, **nothing did** (5-gram repetition: 20.7% vs 20.8% over 13 pairs).

So the honest reading is: topic sets the probability, sampling decides the instance.

Metrics
-------
* **character 5-gram repetition rate** - the same quantity the R1 Thoughtology paper plots
  as "rumination" (their Figure 3.3b, 0-35% across tasks), so the numbers are comparable
  to published ones rather than invented here;
* **median chain length** - to test the intuitive explanation "it thought too much to
  finish". It is false: swallowed chains are not longer, and are often shorter.

Unit, stated precisely
----------------------
This tool counts **every retried attempt** that left a raw stream - which includes
retries for transport and server errors, not only the `EMPTY_RESPONSE` case. That is
deliberate (a failed attempt is a failed attempt for the chain comparison), but it means
the totals here are **larger** than the `EMPTY_RESPONSE` counts in
`silence_shape_report.py --empty-only`. In the measured deployment: 29 of the 31 retried
attempts in the long session were `EMPTY_RESPONSE`. Quote whichever fits the claim, and
say which one it is.

Limitations worth stating
-------------------------
* a matched pair only exists for steps that were retried **and then** produced a reply;
  steps that exhausted the retry budget have no successful partner, and are excluded from
  comparison 2 (they still count in comparison 1);
* sample sizes are small (tens of chains, tens of pairs). Treat the direction as the
  finding, not the third digit;
* the metric is a proxy for "re-examining what was already examined" - it will also rise
  on any repetitive style, so read it comparatively, never absolutely.

Usage
-----
    python3 rumination_compare.py
    python3 rumination_compare.py --session 1a2b3c4d
    python3 rumination_compare.py --pairs-only
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
from statistics import median

SESSIONS = Path(os.environ.get("DSH_SESSIONS_DIR", Path.home() / ".dsh" / "sessions"))
ARCHIVE_NAME = "session.v3.jsonl.zstd"
TZ = timezone(timedelta(hours=8))


def _read(f: Path):
    raw = subprocess.run(["zstd", "-dc", str(f)], capture_output=True).stdout.decode("utf-8", "replace")
    for ln in raw.splitlines():
        if '"assistant/attempt"' not in ln and '"assistant/message"' not in ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("type") in ("assistant/attempt", "assistant/message"):
            yield e


def _stream_reasoning(stream) -> str:
    out = ""
    for e in stream or []:
        if e.get("type") == "reasoning-chunks":
            t = e.get("texts") or []
            out += "".join(t) if isinstance(t, list) else str(t)
    return out


def repetition_rate(text: str, n: int = 5) -> float:
    """Percentage of character n-grams that are duplicates (the paper's rumination metric)."""
    t = re.sub(r"\s+", "", text or "")
    if len(t) < n:
        return 0.0
    grams = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 100.0 * (1 - len(set(grams)) / len(grams))


def collect():
    """-> (swallowed, ok). Each entry: (session, turn, step, chain_text).

    ⚠️ **Deduplicated by content.** One deployment had several session branches that
    shared a stretch of history (a reconnect or fork leaves the same turns in more than
    one archive). Without this, one real failure is counted once per branch and every
    rate is inflated. The key is the chain text itself, so legitimately repeated content
    (e.g. an identical trivial reply) is also collapsed — that is the conservative
    direction for a rate, and the raw count is printed alongside.
    """
    swallowed, ok = [], []
    seen = set()

    def _push(bucket, sess, turn, step, txt):
        key = (bucket is ok, hash(txt))
        if key in seen:
            return False
        seen.add(key)
        bucket.append((sess, turn, step, txt))
        return True

    raw_swallowed = raw_ok = 0
    if not SESSIONS.exists():
        return swallowed, ok
    for slug in sorted(os.listdir(SESSIONS)):
        sd = SESSIONS / slug
        if not sd.is_dir():
            continue
        for d in sorted(os.listdir(sd)):
            f = sd / d / ARCHIVE_NAME
            if not f.exists():
                continue
            for e in _read(f):
                da = e.get("data") or {}
                turn, step = da.get("turn"), da.get("step")
                if turn is None:
                    continue
                if e.get("type") == "assistant/attempt":
                    txt = _stream_reasoning(da.get("stream"))
                    if txt:
                        raw_swallowed += 1
                        _push(swallowed, d, turn, step, txt)
                else:
                    parts = (da.get("message") or {}).get("content") or []
                    r = "".join(p.get("text", "") for p in parts if p.get("type") == "reasoning")
                    x = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                    if r and x.strip():
                        raw_ok += 1
                        _push(ok, d, turn, step, r)
    collect.raw_counts = (raw_swallowed, raw_ok)
    return swallowed, ok


def main() -> int:
    ap = argparse.ArgumentParser(description="swallowed vs. successful reasoning chains")
    ap.add_argument("--session", default="", help="only sessions whose id contains this")
    ap.add_argument("--pairs-only", action="store_true", help="skip the all-data comparison")
    ap.add_argument("--sessions", default="", help="archive root (default ~/.dsh/sessions)")
    args = ap.parse_args()

    global SESSIONS
    if args.sessions:
        SESSIONS = Path(args.sessions)

    swallowed, ok = collect()
    if args.session:
        swallowed = [s for s in swallowed if args.session in s[0]]
        ok = [o for o in ok if args.session in o[0]]
        print(f"(only sessions containing {args.session})")

    if not swallowed:
        print("no swallowed responses found in these archives")
        return 0

    raw_sw, raw_ok = getattr(collect, "raw_counts", (len(swallowed), len(ok)))
    if raw_sw != len(swallowed) or raw_ok != len(ok):
        print(f"deduplicated: {raw_sw} swallowed -> {len(swallowed)}, "
              f"{raw_ok} successful -> {len(ok)}  (session branches sharing history)")

    if not args.pairs_only:
        rs = [repetition_rate(s[3]) for s in swallowed]
        ro = [repetition_rate(o[3]) for o in ok]
        print("=== all data (describes which conversations swallow, not which attempt) ===")
        print(f"  swallowed chains : {len(rs):>4}   median length {median([len(s[3]) for s in swallowed]):>7,.0f} chars"
              f"   median 5-gram repetition {median(rs):>5.1f}%")
        print(f"  successful chains: {len(ro):>4}   median length {median([len(o[3]) for o in ok]):>7,.0f} chars"
              f"   median 5-gram repetition {median(ro):>5.1f}%")

    # matched pairs: same session+turn+step, one swallowed attempt and one success
    pairs = defaultdict(lambda: {"bad": [], "good": []})
    for s in swallowed:
        pairs[(s[0], s[1], s[2])]["bad"].append(s[3])
    for o in ok:
        pairs[(o[0], o[1], o[2])]["good"].append(o[3])
    matched = [(k, v) for k, v in pairs.items() if v["bad"] and v["good"]]

    print(f"\n=== matched within a step (same input, different sample) ===")
    print(f"  usable pairs: {len(matched)}")
    if not matched:
        return 0
    print(f"\n  {'session':<14}{'turn/step':<12}{'swallowed':>10}{'success':>9}{'rep% bad':>10}{'rep% good':>11}")
    rb, rg = [], []
    for (sess, turn, step), v in sorted(matched, key=lambda kv: kv[0][1]):
        b, g = v["bad"][0], v["good"][0]
        x, y = repetition_rate(b), repetition_rate(g)
        rb.append(x); rg.append(y)
        print(f"  {sess[:12]:<14}t{turn}/s{step:<8}{len(b):>10,}{len(g):>9,}{x:>10.1f}{y:>11.1f}")
    print(f"\n  median 5-gram repetition: swallowed {median(rb):.1f}%  vs  successful {median(rg):.1f}%")
    print("  median chain length     : swallowed "
          f"{median([len(v['bad'][0]) for _, v in matched]):,.0f}  vs  successful "
          f"{median([len(v['good'][0]) for _, v in matched]):,.0f}")
    print("\n  Reading: if these two rows look alike, the chain carries no warning sign -")
    print("  the failure is a sampling event at the reasoning/content boundary, not a")
    print("  property of what the model was thinking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
