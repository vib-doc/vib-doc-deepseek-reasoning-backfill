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
* **one pair per (session, turn, step), using the FIRST swallowed attempt.** A single step
  can be swallowed several times before it succeeds - measured here: 6 attempts for one
  step, 4 for three others, 2 for two more (35 attempts across 19 distinct steps). Because
  the paired table has one row per step, a reader who takes the *last* attempt instead gets
  the same row count but different values, and may reasonably suspect the table is wrong.
  The choice is therefore stated here, and printed by the tool, rather than left implicit;
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
    for ln in raw.split("\n"):  # 用 \n 切：splitlines() 会在 U+2028 上误切
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
    """Percentage of character n-grams that are duplicates (the paper's rumination metric).

    ⚠️ **Whitespace is stripped before counting** (including newlines). This choice moves
    the number by ~8 points on this data - 20.7 / 20.8 stripped vs 28.3 / 28.8 unstripped
    for the matched pairs - so it has to be stated, not assumed. Both versions support the
    same conclusion (the two groups do not separate), but a reader comparing against their
    own implementation needs to know which convention produced these figures.
    """
    t = re.sub(r"\s+", "", text or "")
    if len(t) < n:
        return 0.0
    grams = [t[i:i + n] for i in range(len(t) - n + 1)]
    return 100.0 * (1 - len(set(grams)) / len(grams))


# Self-referential wording. ⚠️ LANGUAGE-SPECIFIC: this list is the Chinese one used in the
# measured deployment. Measuring English traffic with it would report ~0 and look like a
# clean negative result. Pass your own list with --self-ref-words FILE (one per line).
DEFAULT_SELF_REF = [
    "我自己", "我的状态", "我在想", "我在判断", "我是不是", "我有没有",
    "我意识到", "我决定", "我的判断", "我在做", "我这个人",
]


def metrics(text: str, self_ref=None) -> dict:
    """Every metric that was computed when making the claim, in one place.

    The point of having them all here: the claim in the note is that *none* of them
    separates a swallowed attempt from a retry that succeeded, on the same input. That
    claim is only as good as the metrics behind it, so each one is reproducible.
    """
    text = text or ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    run = best = 1 if lines else 0
    for i in range(1, len(lines)):
        run = run + 1 if lines[i] == lines[i - 1] else 1
        best = max(best, run)
    tail = lines[int(len(lines) * 0.8):] or [""]
    from collections import Counter
    tail_share = 100.0 * max(Counter(tail).values()) / len(tail)
    flat = re.sub(r"\s+", "", text)
    shingles = [flat[i:i + 20] for i in range(0, len(flat) - 19, 5)] if len(flat) >= 20 else []
    shingle20 = 100.0 * (1 - len(set(shingles)) / len(shingles)) if shingles else 0.0
    words = self_ref if self_ref is not None else DEFAULT_SELF_REF
    selfref = 1000.0 * sum(text.count(w) for w in words) / max(len(text), 1)
    return {
        "chars": len(text),
        "lines": len(lines),
        "max_line_run": best,
        "tail_repeat_%": tail_share,
        "shingle20_%": shingle20,
        "ngram5_%": repetition_rate(text, 5),
        "selfref_per_1000": selfref,
    }


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
    ap.add_argument("--self-ref-words", default="",
                    help="file of self-referential markers, one per line (default: built-in Chinese list)")
    args = ap.parse_args()

    global SESSIONS
    if args.sessions:
        SESSIONS = Path(args.sessions)

    words = None
    if args.self_ref_words:
        with open(args.self_ref_words, encoding="utf-8") as fh:
            words = [ln.strip() for ln in fh if ln.strip()]
        print(f"(self-referential markers: {len(words)} from {args.self_ref_words})")

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
    # Every metric, both groups, over the SAME pairs - this is the table the note's
    # strongest claim rests on, so it is printed in full rather than summarised.
    bad_m, good_m = [], []
    for (sess, turn, step), v in sorted(matched, key=lambda kv: kv[0][1]):
        bad_m.append(metrics(v["bad"][0], words))
        good_m.append(metrics(v["good"][0], words))

    keys = list(bad_m[0].keys())
    print(f"\n  one pair per (session, turn, step) using the FIRST attempt; 5-gram and shingle")
    print(  "  rates are computed with whitespace stripped")
    print(f"\n  {'metric':<20}{'swallowed':>12}{'successful':>12}{'separates?':>12}")
    separated = []
    for k in keys:
        mb = median(m[k] for m in bad_m)
        mg = median(m[k] for m in good_m)
        # A metric "separates" if the group medians differ by more than 20% of the larger
        # one. Deliberately crude: with tens of samples, a finer test would overclaim.
        hi = max(abs(mb), abs(mg))
        sep = hi > 0 and abs(mb - mg) / hi > 0.20
        if sep:
            separated.append(k)
        print(f"  {k:<20}{mb:>12.2f}{mg:>12.2f}{('YES' if sep else 'no'):>12}")
    print(f"\n  metrics compared: {len(keys)}   separating (>20% apart): {len(separated)}"
          + (f"  -> {separated}" if separated else ""))
    print("  two of them are degenerate on this data and are reported for completeness:")
    print("    tail_repeat_%   ~100 in BOTH groups - the tail of every chain ends in the")
    print("                    same short self-prompt loop, so it measures nothing here;")
    print("    max_line_run    ~32 vs ~35 - that long run IS the loop, and it is present in")
    print("                    successful chains too, which is why it cannot be a detector.")
    print()
    if separated:
        print("  Reading: at least one metric does separate the two groups on matched input -")
        print("  that is a lead worth chasing, not a finished result. Check it on more pairs")
        print("  before believing it.")
    else:
        print("  Reading: none of the metrics separates a swallowed attempt from a retry of")
        print("  the same step that succeeded. The chain carries no warning sign - the failure")
        print("  is a sampling event at the reasoning/content boundary, not a property of what")
        print("  the model was thinking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
