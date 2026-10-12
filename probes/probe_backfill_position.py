#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: is the `reasoning_content` backfill requirement scoped to the *active* segment?

The question
------------
The official docs (Thinking Mode → Tool Calls) state the requirement without a scope:

    "for requests carrying the `tools` parameter, the `reasoning_content` must be
     fully passed back to the API in all subsequent requests — even for turns where
     the model did not perform a tool call. If your code does not correctly pass back
     `reasoning_content`, the API will return a 400 error."

Read literally, that forbids a client from ever folding old reasoning out — which
would make `patches/patch_backfill.py` (§3 of the README) impossible. So the scope
of the requirement decides whether the patch is legal at all.

Why this probe exists alongside `probe_backfill_enforcement.py`
---------------------------------------------------------------
That probe reported "the server did not enforce it", from a request whose tool-calling
message sat **behind a later user turn**. If the requirement is scoped to the active
segment, then that request exercised the *permitted* case — and the probe's conclusion
("not enforced") does not follow from its own evidence. This probe isolates the
variable that the earlier one held fixed: **the position of the stripped message.**

Design
------
Four requests. The two that matter (`B` and `C`) are **byte-identical except for one
trailing user message**, so the only thing that differs is whether the tool-calling
assistant message is the most recent assistant message or an older one.

    A  plain turn,                     reasoning stripped   (baseline)
    B  tool-turn in the ACTIVE segment, reasoning stripped   <- docs say 400
    C  tool-turn in a PAST segment,     reasoning stripped   <- same message, later user turn
    D  tool-turn in the ACTIVE segment, reasoning backfilled (control for B)

Expected if the requirement is scoped to the active segment:  A 200 · B 400 · C 200 · D 200.
Expected if it is enforced globally, as written:              A 200 · B 400 · C 400 · D 200.

`B` vs `C` is the whole experiment; `A` and `D` are there so a lone 200 cannot be
read as "nothing is enforced" and a lone 400 cannot be read as "everything is".

Result (2026-10-12, `deepseek-flash` @ api.deepseek.com, reasoning_effort=low)
------------------------------------------------------------------------------
    A plain turn, stripped                           HTTP 200
    B tool-turn, ACTIVE segment, stripped            HTTP 400   <- docs are right
    C tool-turn, PAST segment,   stripped            HTTP 200   <- ...but only here
    D tool-turn, ACTIVE segment, backfilled          HTTP 200   (control for B)
    E 2-step segment, EARLIER tool step stripped     HTTP 400   <- the rule does not loosen
    F 2-step segment, both backfilled                HTTP 200   (control for E)

The rule this pins down:

> **A tool-calling assistant message must carry `reasoning_content` unless a later
> `user` message closes its segment. Inside the open segment, *every* tool-call
> message must carry it — not just the latest one (E).**

`B` vs `C` is byte-identical except for one trailing user message, and it flips
400 -> 200, so the scope really is the segment and not the request. `E` says the
narrowing stops there: the README's §3 rule ("tool-call messages within the current
segment keep their reasoning") is the loosest rule the server accepts, which is a
useful thing to know before trying to tighten it.

This supersedes the reading in `probe_backfill_enforcement.py`'s header, whose case C
sat in a *past* segment and therefore never tested the case the docs describe. Both
probes are kept: the older one is the record of how the mistake was made.

Caveat, unchanged: one model, one endpoint, one day, n=1 per cell. Re-run before
relying on it — that is what the script is for.

Usage
-----
    export DEEPSEEK_API_KEY=sk-...
    python3 probe_backfill_position.py
    python3 probe_backfill_position.py --model deepseek-reasoner
    python3 probe_backfill_position.py --env /path/to/.env      # read key/base from a file
    python3 probe_backfill_position.py --dry-run                # print the payloads, send nothing
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TOOL = [{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Return the current time.",
        "parameters": {"type": "object", "properties": {}},
    },
}]

# One tool-calling assistant message, reused by both cases. It is the same object in
# B and C on purpose: if the two diverge, the "position" variable is no longer isolated.
TOOL_TURN_MSG = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{
        "id": "call_probe_pos_1",
        "type": "function",
        "function": {"name": "get_time", "arguments": "{}"},
    }],
}

# A stand-in for a real reasoning_content. The server does not check that it is
# *correct*, only that the field is present, so a fixed string is enough for the control.
FAKE_REASONING = "(reasoning text omitted by this probe)"


def case_b_active_segment():
    """The tool-calling message IS the most recent assistant message."""
    return [
        {"role": "user", "content": "What time is it? Use the tool."},
        dict(TOOL_TURN_MSG),
        {"role": "tool", "tool_call_id": "call_probe_pos_1", "content": "12:00"},
    ]


def case_c_past_segment():
    """Same three messages, plus one more user turn — pushing the tool turn behind it."""
    return case_b_active_segment() + [
        {"role": "user", "content": "Thanks, reply with one word."},
    ]


def case_d_active_backfilled():
    """Case B with the reasoning put back: the control that separates
    'the request was rejected' from 'this arbitrary string was rejected'."""
    msgs = case_b_active_segment()
    msgs[1] = dict(TOOL_TURN_MSG, reasoning_content=FAKE_REASONING)
    return msgs


def case_a_plain_stripped():
    return [
        {"role": "user", "content": "1+1=? Reply with the number only."},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "Add one more. Number only."},
    ]


def _two_step_segment(first_stripped):
    """One segment with two tool steps — the shape `patch_backfill.py`'s rule has to
    decide about. `first_stripped` controls whether the EARLIER tool step keeps its
    reasoning; the latest one always does, because B/C/D say it must.

    The rule in §3 of the README keeps reasoning for *every* tool-call message in the
    segment. If E is accepted, it only needs to keep the latest one — which is a
    strictly smaller payload, and a different patch."""
    first = dict(TOOL_TURN_MSG, tool_calls=[{
        "id": "call_probe_pos_e1", "type": "function",
        "function": {"name": "get_time", "arguments": "{}"},
    }])
    if not first_stripped:
        first = dict(first, reasoning_content=FAKE_REASONING)
    return [
        {"role": "user", "content": "What time is it? Use the tool."},
        first,
        {"role": "tool", "tool_call_id": "call_probe_pos_e1", "content": "12:00"},
        {"role": "assistant", "content": "", "reasoning_content": FAKE_REASONING,
         "tool_calls": [{
             "id": "call_probe_pos_e2", "type": "function",
             "function": {"name": "get_time", "arguments": "{}"},
         }]},
        {"role": "tool", "tool_call_id": "call_probe_pos_e2", "content": "12:00"},
    ]


def case_e_earlier_tool_step_stripped():
    return _two_step_segment(first_stripped=True)


def case_f_earlier_tool_step_backfilled():
    return _two_step_segment(first_stripped=False)


CASES = [
    ("A plain turn, stripped", case_a_plain_stripped),
    ("B tool-turn, ACTIVE segment, stripped", case_b_active_segment),
    ("C tool-turn, PAST segment, stripped", case_c_past_segment),
    ("D tool-turn, ACTIVE segment, backfilled", case_d_active_backfilled),
    ("E 2-step segment, EARLIER tool step stripped", case_e_earlier_tool_step_stripped),
    ("F 2-step segment, both backfilled (control for E)", case_f_earlier_tool_step_backfilled),
]


def describe(messages):
    """Say what was actually SENT, not what came back.

    (The earlier probe printed `tool_calls=<bool>` from the *response*, i.e. whether the
    model chose to call a tool this turn. That reads like a statement about the request
    and is easy to mistake for one. This line is about the request.)
    """
    n_tool_calls = sum(1 for m in messages if m.get("tool_calls"))
    n_reasoning = sum(1 for m in messages if m.get("reasoning_content"))
    return f"sent: {len(messages)} msgs, tool_calls in {n_tool_calls}, reasoning in {n_reasoning}"


def call(tag, messages, *, key, base, model, max_tokens=16, dry_run=False):
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOL,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "thinking": {"type": "enabled"},
    }
    if dry_run:
        print(f"[{tag}] DRY RUN — {describe(messages)}")
        print("    " + json.dumps(body, ensure_ascii=False)[:400])
        return
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        print(
            f"[{tag}] HTTP {resp.status}   {describe(messages)}\n"
            f"    responded: reasoning={bool(msg.get('reasoning_content'))} "
            f"content={bool(msg.get('content'))} "
            f"tool_calls={bool(msg.get('tool_calls'))}"
        )
    except urllib.error.HTTPError as exc:
        print(f"[{tag}] HTTP {exc.code}   {describe(messages)}")
        print("    " + exc.read().decode("utf-8", "replace")[:400].replace("\n", " "))
    except Exception as exc:  # network etc.
        print(f"[{tag}] failed: {type(exc).__name__}: {exc}")


def load_env_file(path):
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description="Probe the scope of the reasoning_content backfill requirement")
    ap.add_argument("--env", default="", help="optional .env file to read the key from")
    ap.add_argument("--base", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    ap.add_argument("--dry-run", action="store_true", help="print the payloads without sending them")
    args = ap.parse_args()

    if args.env:
        load_env_file(args.env)
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base = os.environ.get("OPENAI_BASE_URL") or args.base
    if not key and not args.dry_run:
        print("No API key. Set DEEPSEEK_API_KEY (or pass --env FILE with OPENAI_API_KEY).")
        return 2

    print(f"endpoint: {base}\nmodel:    {args.model}\n")
    for tag, build in CASES:
        call(tag, build(), key=key, base=base, model=args.model, dry_run=args.dry_run)
    print(
        "\nRead B against C — they are the same request plus one trailing user message.\n"
        "  B 400 / C 200  -> the requirement is scoped to the active segment (old reasoning is strippable)\n"
        "  B 400 / C 400  -> enforced globally, as the docs are written\n"
        "  B 200          -> not enforced at all on this model/endpoint today"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
