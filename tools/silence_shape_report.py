#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report "swallowed replies" from DeepSeek Harness session archives.

The symptom
-----------
Occasionally a turn produces **no visible reply at all**. The stream looks like this:

    block-start(reasoning) → reasoning-chunks → block-end → usage → finish(error)

Reasoning was emitted normally — often complete, with the answer already drafted
inside it — but the **content block never opened**, so the reply is empty. The
provider reports:

    failure.code    = EMPTY_RESPONSE
    failure.message = "model returned a completed response with no content"
    usage.outputTokens == usage.reasoningTokens   (all of it went to reasoning)

This is *not* a truncation (that would be `finish.reason = length`) and *not* a
content failure: retrying the same step usually succeeds.

Why a tool is needed
--------------------
`assistant/attempt` events (the raw stream of a failed response) exist **only for
responses that were retried**. So the shape is invisible unless you go looking.
This script reads those archives and prints:

  * per day: how many swallows, in how many turns, and the longest consecutive run
    within one turn — "sticky" (≥2 in a row) vs "isolated" (a single one);
  * `--shape`: for each affected step, the reasoning length, whether a content block
    ever opened, and where the output tokens went;
  * `--watch`: append newly-seen swallows to a file, idempotently.

Reading the shapes (measured)
-----------------------------
Of 15 swallowed attempts in one long session, **every one** had a complete reasoning
chain and **no content block**. Chain length was unrelated to failure: the shortest
was 56 output tokens, the longest 6,657. Retried steps that succeeded were often
*shorter* than the attempts that failed — so "it thought too much to finish" is not
the mechanism.

The tail of the reasoning chain often degenerates into a self-prompt loop
(`(output.) (output.) (output.) ...`). That loop is **not** a predictor: successful
chains show the same tail. It is a symptom, not a signal.

Usage
-----
    python3 silence_shape_report.py                       # per-day summary
    python3 silence_shape_report.py --shape               # add per-attempt shapes
    python3 silence_shape_report.py --detail              # list every event
    python3 silence_shape_report.py --days 3              # last N days
    python3 silence_shape_report.py --watch --alarm out.md  # append new ones, idempotent
    python3 silence_shape_report.py --sessions DIR        # archive root
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
from typing import Any, Dict, List, Optional, Tuple

# Archive-selection rule lives in tools/dsh_archive.py (2026-09-30):
# never hardcode an archive generation — see that file's docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsh_archive import latest_for, iter_archives  # noqa: E402

SESSIONS = Path(os.environ.get("DSH_SESSIONS_DIR", Path.home() / ".dsh" / "sessions"))
TZ = timezone(timedelta(hours=8))


def _iter_archives():
    if not SESSIONS.exists():
        return
    for slug in sorted(os.listdir(SESSIONS)):
        sd = SESSIONS / slug
        if not sd.is_dir():
            continue
        for d in sorted(os.listdir(sd)):
            # newest generation (v4 > v3 > unversioned), never a hardcoded one
            f = latest_for(sd / d)
            if f.exists():
                yield f


def _read(f: Path):
    """Decompress one archive and yield parsed events (skipping undecodable lines)."""
    raw = subprocess.run(["zstd", "-dc", str(f)], capture_output=True).stdout.decode("utf-8", "replace")
    for ln in raw.split("\n"):  # 用 \n 切：splitlines() 会在 U+2028 上误切
        if '"type"' not in ln:
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def collect() -> List[Dict[str, Any]]:
    """Every retry event (one per failed attempt) across all archives."""
    rows: List[Dict[str, Any]] = []
    for f in _iter_archives():
        for e in _read(f):
            if e.get("type") != "llm/retry":
                continue
            da = e.get("data") or {}
            fail = da.get("failure") or {}
            ts = (e.get("time") or 0) / 1000
            rows.append({
                "at": datetime.fromtimestamp(ts, TZ) if ts else None,
                "session": f.parent.name,
                "turn": da.get("turn"),
                "step": da.get("step"),
                "retry": da.get("retry"),
                "code": fail.get("code") or "",
                "msg": str(fail.get("message") or "")[:100],
            })
    rows.sort(key=lambda r: (r["at"] or datetime.min.replace(tzinfo=TZ)))
    return rows


def _shape_of(stream) -> Dict[str, Any]:
    """Compress one raw attempt stream into the few numbers that matter."""
    reasoning = ""
    text_len = 0
    has_text_block = False
    out_tok = reason_tok = None
    for e in stream or []:
        et = e.get("type")
        if et == "reasoning-chunks":
            t = e.get("texts") or []
            reasoning += "".join(t) if isinstance(t, list) else str(t)
        elif et == "text-chunks":
            has_text_block = True
            t = e.get("texts") or []
            text_len += len("".join(t) if isinstance(t, list) else str(t))
        elif et == "chunk":
            c = e.get("chunk") or {}
            if c.get("type") == "text":
                has_text_block = True
                text_len += len(str(c.get("text") or ""))
            elif c.get("type") == "usage":
                u = c.get("usage") or {}
                out_tok = u.get("outputTokens")
                reason_tok = u.get("reasoningTokens")
    tail = reasoning[-400:]
    loop = sum(tail.count(x) for x in ("（输出。）", "（做。）", "（写。）", "(output.)", "(write.)"))
    return {"chain": len(reasoning), "has_text": has_text_block, "text_len": text_len,
            "out_tok": out_tok, "reason_tok": reason_tok, "loop": loop}


def collect_attempts() -> List[Dict[str, Any]]:
    """Shape of every attempt that has a raw stream (= the retried ones)."""
    rows: List[Dict[str, Any]] = []
    for f in _iter_archives():
        for e in _read(f):
            if e.get("type") != "assistant/attempt":
                continue
            da = e.get("data") or {}
            ts = (e.get("time") or 0) / 1000
            rows.append({
                "at": datetime.fromtimestamp(ts, TZ) if ts else None,
                "session": f.parent.name, "turn": da.get("turn"), "step": da.get("step"),
                **_shape_of(da.get("stream")),
            })
    rows.sort(key=lambda r: (r["at"] or datetime.min.replace(tzinfo=TZ)))
    return rows


def print_shapes(rows: List[Dict[str, Any]]) -> None:
    attempts = collect_attempts()
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for a in attempts:
        groups[(a["session"], a["turn"], a["step"])].append(a)
    keys = {(r["session"], r["turn"], r["step"]) for r in rows}

    print(f"\n{'time':<17}{'turn/step':<12}{'chain':>8}{'content':>10}{'out(reason)':>16}{'tail loop':>10}")
    n = 0
    for k in sorted(keys, key=lambda k: min((g["at"] for g in groups.get(k, []) if g["at"]),
                                            default=datetime.min.replace(tzinfo=TZ))):
        seq = sorted(groups.get(k, []), key=lambda x: x["at"] or datetime.min.replace(tzinfo=TZ))
        for i, a in enumerate(seq, 1):
            n += 1
            tok = f"{a['out_tok']}({a['reason_tok']})" if a["out_tok"] is not None else "?"
            print(f"{a['at'].strftime('%m-%d %H:%M:%S'):<17}t{a['turn']}/s{a['step']:<8}"
                  f"{a['chain']:>8}{'NONE':>10}{tok:>16}{a['loop']:>10}   swallowed #{i}")
    print(f"\n  {n} swallowed response(s). Every one: reasoning present, content block never opened.")
    print("  `assistant/attempt` only exists for retried responses, so this list IS the swallow list.")


# ── --watch: append newly-seen swallows, idempotently ─────────────────────────
ALARM_STATE = "silence_alert_state.json"


def watch(alarm_path: str) -> int:
    """A turn that exhausted its retries: `assistant/attempt` immediately followed by
    `step/end` (no `llm/retry` in between) => no content was produced and no retry
    remains. That is the case a user actually experiences as 'it didn't answer'."""
    seen: set = set()
    files: Dict[str, Any] = {}
    if os.path.exists(ALARM_STATE):
        try:
            st = json.load(open(ALARM_STATE, encoding="utf-8"))
            seen = set(st.get("seen") or [])
            files = st.get("files") or {}
        except Exception:
            pass

    new = []
    for f in _iter_archives():
        key = str(f)
        try:
            sig = [f.stat().st_size, int(f.stat().st_mtime)]
        except OSError:
            continue
        if files.get(key) == sig:      # unchanged archive: skip the decompression
            continue
        files[key] = sig
        prev = None
        for e in _read(f):
            t = e.get("type")
            da = e.get("data") or {}
            if prev and prev[0] == "assistant/attempt" and t == "step/end":
                rid = f"t{prev[2]}/s{prev[3]}/{int((prev[1] or 0) / 1000)}"
                if rid not in seen:
                    seen.add(rid)
                    new.append({"id": rid, "session": f.parent.name, "turn": prev[2],
                                "at": datetime.fromtimestamp((prev[1] or 0) / 1000, TZ)})
            # A retry or a produced message breaks adjacency; so does anything else.
            prev = (t, e.get("time") or 0, da.get("turn"), da.get("step")) if t in (
                "assistant/attempt", "step/end", "llm/retry", "assistant/message") else None

    if new and alarm_path:
        first = not os.path.exists(alarm_path)
        os.makedirs(os.path.dirname(os.path.abspath(alarm_path)) or ".", exist_ok=True)
        with open(alarm_path, "a", encoding="utf-8") as fh:
            if first:
                fh.write("# Silence alarm\n\n"
                         "Criterion: `assistant/attempt` immediately followed by `step/end`\n"
                         "(no `llm/retry` between) = the last attempt was swallowed AND no retry\n"
                         "remains => **that turn produced no reply at all**.\n\n"
                         "Retried-and-recovered swallows are NOT listed here.\n\n---\n")
            for r in new:
                fh.write(f"\n## {r['at'].strftime('%Y-%m-%d %H:%M:%S')} - turn produced no reply\n\n"
                         f"- session: `{r['session']}`\n- turn **{r['turn']}**\n")
    try:
        json.dump({"seen": sorted(seen), "files": files}, open(ALARM_STATE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    except Exception:
        pass
    print(f"new alarms: {len(new)}; total seen: {len(seen)}")
    for r in new:
        print(f"  · {r['at'].strftime('%m-%d %H:%M:%S')}  turn {r['turn']}  {r['session'][:24]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Swallowed-reply report (reasoning with no content)")
    ap.add_argument("--days", type=int, default=0, help="only the last N days (0 = all)")
    ap.add_argument("--detail", action="store_true", help="list every event")
    ap.add_argument("--shape", action="store_true", help="per-attempt shape table")
    ap.add_argument("--watch", action="store_true", help="append new 'no reply at all' turns, idempotent")
    ap.add_argument("--alarm", default="silence_alarm.md", help="where --watch appends")
    ap.add_argument("--sessions", default="", help="archive root (default ~/.dsh/sessions)")
    ap.add_argument("--empty-only", action="store_true", help="only EMPTY_RESPONSE (default in practice)")
    args = ap.parse_args()

    global SESSIONS
    if args.sessions:
        SESSIONS = Path(args.sessions)

    if args.watch:
        return watch(args.alarm)

    rows = collect()
    if args.empty_only:
        rows = [r for r in rows if r["code"] == "EMPTY_RESPONSE"]
    if args.days and rows:
        cut = datetime.now(TZ) - timedelta(days=args.days)
        rows = [r for r in rows if r["at"] and r["at"] >= cut]

    print("=== swallowed replies (reasoning emitted, no content) ===")
    print(f"  events: {len(rows)}" + ("  [EMPTY_RESPONSE only]" if args.empty_only else ""))
    print("  note: TRANSPORT/SERVER retries are a different failure; use --empty-only to exclude them")
    print()

    per_day: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"events": 0, "codes": defaultdict(int), "turns": set()})
    per_turn: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r["at"]:
            continue
        d = r["at"].strftime("%m-%d")
        per_day[d]["events"] += 1
        per_day[d]["codes"][r["code"]] += 1
        per_day[d]["turns"].add((r["session"], r["turn"]))
        per_turn[(r["session"], r["turn"])].append(r)

    print(f"{'date':<8}{'events':>7}{'turns':>7}{'EMPTY':>7}{'other':>7}   shape")
    for d in sorted(per_day):
        v = per_day[d]
        empty = v["codes"].get("EMPTY_RESPONSE", 0)
        worst = 0
        for (w, t), evs in per_turn.items():
            if not evs or evs[0]["at"].strftime("%m-%d") != d:
                continue
            worst = max(worst, sum(1 for e in evs if e["code"] == "EMPTY_RESPONSE"))
        shape = f"sticky (>=2 in a row)" if worst >= 2 else ("isolated" if empty else "-")
        print(f"{d:<8}{v['events']:>7}{len(v['turns']):>7}{empty:>7}{v['events']-empty:>7}   {shape} (longest run {worst})")

    if args.shape:
        print_shapes(rows)
    if args.detail:
        print()
        for r in rows:
            t = r["at"].strftime("%m-%d %H:%M:%S") if r["at"] else "?"
            print(f"  {t}  turn {r['turn']} step {r['step']} retry {r['retry']}  {r['code']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
