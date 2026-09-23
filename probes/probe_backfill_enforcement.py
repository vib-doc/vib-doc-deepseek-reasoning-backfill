#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: does the DeepSeek API actually enforce full `reasoning_content` backfill?

The question
------------
The official docs (Thinking Mode → Tool Calls) say:

    "for requests carrying the `tools` parameter, the `reasoning_content` must be
     fully passed back to the API in all subsequent requests — even for turns where
     the model did not perform a tool call. If your code does not correctly pass back
     `reasoning_content`, the API will return a 400 error."

That statement is a hard constraint *if enforced*. This probe tests whether the
server actually enforces it, because it changes what a client is allowed to do:

  * if enforced  → folding old reasoning out of the request is impossible;
  * if not       → a client may choose what to send back (which controls both cost
                   and how much of its own past reasoning the model can re-read).

Result (2026-09-23, `deepseek-flash` @ api.deepseek.com)
--------------------------------------------------------
Both probes returned **HTTP 200**. No 400.

  [C] tool-calling turn, assistant message WITHOUT reasoning_content  → 200
  [A] ordinary turn,     assistant message WITHOUT reasoning_content  → 200

Caveat, stated plainly: that is **one model, one endpoint, sampled once**. It does
not prove the field is optional forever, or on other models. Treat it as "the server
did not enforce this on <date>", and keep a way to revert.

Usage
-----
    export DEEPSEEK_API_KEY=sk-...
    python3 probe_backfill_enforcement.py
    python3 probe_backfill_enforcement.py --model deepseek-reasoner --base https://api.deepseek.com/v1

    # or point it at a .env file
    python3 probe_backfill_enforcement.py --env /path/to/.env
"""
from __future__ import annotations

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
        "description": "Get the current time",
        "parameters": {"type": "object", "properties": {}},
    },
}]


def load_env_file(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


def call(tag: str, messages, *, key: str, base: str, model: str, max_tokens: int = 16) -> None:
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOL,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "thinking": {"type": "enabled"},
    }
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
            f"[{tag}] HTTP {resp.status}  "
            f"reasoning={bool(msg.get('reasoning_content'))}  "
            f"content={bool(msg.get('content'))}  "
            f"tool_calls={bool(msg.get('tool_calls'))}"
        )
    except urllib.error.HTTPError as exc:
        print(f"[{tag}] HTTP {exc.code}  <-- the server rejected it")
        print("      " + exc.read().decode("utf-8", "replace")[:400])
    except Exception as exc:  # network etc.
        print(f"[{tag}] failed: {type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe reasoning_content backfill enforcement")
    ap.add_argument("--env", default="", help="optional .env file to read the key from")
    ap.add_argument("--base", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    args = ap.parse_args()

    if args.env:
        load_env_file(args.env)
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base = os.environ.get("OPENAI_BASE_URL") or args.base
    if not key:
        print("No API key. Set DEEPSEEK_API_KEY (or pass --env FILE with OPENAI_API_KEY).")
        return 2

    print(f"endpoint: {base}\nmodel:    {args.model}\n")

    # [C] A tool-calling assistant message with the reasoning stripped out.
    #     This is the exact shape the docs say must cause a 400.
    call("C tool-turn, reasoning stripped", [
        {"role": "user", "content": "What time is it? Use the tool."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_probe_1", "type": "function",
            "function": {"name": "get_time", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_probe_1", "content": "12:00"},
        {"role": "user", "content": "Thanks, reply with one word."},
    ], key=key, base=base, model=args.model)

    # [A] An ordinary assistant turn with the reasoning stripped out.
    call("A plain turn, reasoning stripped", [
        {"role": "user", "content": "1+1=? Reply with the number only."},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "Add one more. Number only."},
    ], key=key, base=base, model=args.model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
