#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Narrow the `reasoning_content` backfill in the DeepSeek adapter of DeepSeek Harness.

Background
----------
When a request carries `tools`, the DeepSeek docs require every previous turn's
`reasoning_content` to be passed back, and the server concatenates it into the
context. For a long-running assistant session that means **every request re-sends
the model's own past reasoning** — a self-conditioning channel, and a large part of
the bill (measured here: ~280k tokens/request, most of it cache reads).

Measured motivation (one long session, 2026-09-23)
--------------------------------------------------
* swallowed replies ("reasoning emitted, no content block ever opened"):
  31 occurrences in ~3 days, clustered in long reflective conversations;
* contexts of swallowed steps carried ~22% more of the model's own reasoning;
* narrowing the backfill took a request from ~280k tokens to ~143k, with no
  observed loss of task continuity.

What this patch does
--------------------
**Only assistant messages that carry a tool call, within the current segment, keep
their reasoning.** Everything older is stripped.

  * "segment" = everything after the last *final reply* (an assistant message with
    visible text and no tool call);
  * past segments: reasoning stripped (their conclusions are already in the text
    those turns produced);
  * tool-calling messages in the current segment: reasoning kept — that is the
    "why am I calling this tool" working state.

Overrides, highest priority first: `force` file (`on`/`off`/`auto`) → a marker at
the start of the user's message (`【工作】` / `【聊天】`, stripped before sending) →
the rule above.

Why per-message, and not a single on/off switch
-----------------------------------------------
Three designs were tried on live traffic the same day; the first two failed for
structural reasons worth recording:

1. **"backfill if this turn already called a tool"** — flipped *within* one turn:
   step 1 = 142,563 tokens, step 2 = 311,237 (+169k). A mid-turn change splits the
   KV-cache prefix and changes what the model can see of itself halfway through an
   answer.
2. **"decide once per segment, from the previous segment"** — no more flipping, but
   the predicate contradicted what the segment actually did in 27.2% of segments
   (measured over 678 segments: previous-used/this-didn't = 92, previous-didn't/
   this-did = 92). The "decide at segment start" constraint makes this unavoidable:
   at that moment the current segment hasn't happened yet.
3. **(this patch)** — drop the global switch. Decide *per message*. No prediction
   needed, and **prefix switches drop to zero** (design 2 had 187, the "always start
   false" variant would have had 818).

Scope — deliberately narrow
---------------------------
This touches **only** `reasoning_content` backfill. It does not touch tool
availability, thinking mode, prompts, memory retrieval, or any write path. A
swallowed reply is unaffected either way: a failed attempt only leaves an
`assistant/attempt` event and never enters the conversation history.

Known side effect to watch
--------------------------
The model can no longer re-read its own non-tool reasoning from previous segments.
That may show up as **repeated labour across segments** (re-reading a file it
already read, re-running a command). `tools/repeat_labor_report.py` measures it;
keep an eye on that number after deploying.

Usage
-----
    python3 patch_backfill.py --check     # read-only
    python3 patch_backfill.py             # apply (backs up, then `node --check`)
    python3 patch_backfill.py --revert    # reverse the edits (does not need the backup)

Requires a restart of the harness: the module is loaded at startup.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

GLOB = str(Path.home() / ".npm/_npx/*/node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js")
MARK = "function shouldBackfill(message, index, segStart, mode)"

HELPERS = r"""// ── Reasoning backfill rule (per-message; see patch_backfill.py) ─────────────
// Keep reasoning only on tool-calling assistant messages inside the current
// segment. Past segments are stripped. Overrides: force file > user marker > rule.
const MODE_MARK_RE = /^\s*【(工作|聊天)】\s*/;
function takeModeMarker(text) {
	const s = String(text ?? "");
	const m = MODE_MARK_RE.exec(s);
	if (!m) return {
		mode: void 0,
		text: s
	};
	return {
		mode: m[1] === "工作" ? "work" : "chat",
		text: s.slice(m[0].length)
	};
}
function stripModeMarker(text) {
	return takeModeMarker(text).text;
}
function stripMarkerFromContent(content) {
	if (typeof content === "string") return stripModeMarker(content);
	if (Array.isArray(content)) return content.map((p) => p && p.type === "text" ? {
		...p,
		text: stripModeMarker(p.text)
	} : p);
	return content;
}
function isFinalReply(message) {
	if (!message || message.role !== "assistant") return false;
	const blocks = message.content || [];
	const hasText = blocks.some((b) => b.type === "text" && String(b.text || "").trim().length > 0);
	return hasText && !blocks.some((b) => b.type === "tool-call");
}
function hasToolCall(message) {
	const blocks = (message || {}).content || [];
	return blocks.some((b) => b.type === "tool-call");
}
function lastFinalReplyIndex(messages) {
	for (let i = messages.length - 1; i >= 0; i--) if (isFinalReply(messages[i])) return i;
	return -1;
}
function readBackfillMode() {
	try {
		const raw = readFileSync(`${process.env.HOME}/.dsh/backfill_mode`, "utf8").trim();
		if (raw === "on" || raw === "off" || raw === "auto") return raw;
	} catch {}
	return "auto";
}
function resolveMode(messages, forced) {
	if (forced === "on" || forced === "off") return forced;
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].role !== "user") continue;
		const t = flattenText(messages[i].content);
		if (!t.trim()) continue;
		const { mode } = takeModeMarker(t);
		if (mode === "work" || mode === "chat") return mode;
		break;
	}
	return "auto";
}
function shouldBackfill(message, index, segStart, mode) {
	if (mode === "on" || mode === "work") return true;
	if (mode === "off" || mode === "chat") return false;
	if (index < segStart) return false;
	return hasToolCall(message);
}
"""

# (old, new, expected occurrences of `old`)
OPS = [
    ('import { mkdir, readFile } from "node:fs/promises";',
     'import { mkdir, readFile } from "node:fs/promises";\nimport { readFileSync } from "node:fs";', 1),
    ("function serializeAssistant(message) {",
     HELPERS + "function serializeAssistant(message, backfill = true) {", 1),
    ("...reasoning.length > 0 ? { reasoning_content: reasoning } : {},",
     "...reasoning.length > 0 && backfill ? { reasoning_content: reasoning } : {},", 1),
    ("function serializeMessages(messages) {\n\tconst wire = [];\n\tfor (const message of messages) {",
     "function serializeMessages(messages, forced) {\n\tconst wire = [];\n\tconst mode = resolveMode(messages, forced);\n"
     "\tconst segStart = lastFinalReplyIndex(messages) + 1;\n\tfor (const [index, message] of messages.entries()) {", 1),
    ("\t\t\twire.push(serializeAssistant(message));\n\t\t\tcontinue;\n\t\t}\n\t\tconst toolResults = message.content.filter((block) => block.type === \"tool-result\");",
     "\t\t\twire.push(serializeAssistant(message, shouldBackfill(message, index, segStart, mode)));\n"
     "\t\t\tcontinue;\n\t\t}\n\t\tconst toolResults = message.content.filter((block) => block.type === \"tool-result\");", 1),
    ("\t\t\trole: \"user\",\n\t\t\tcontent: text\n\t\t});",
     "\t\t\trole: \"user\",\n\t\t\tcontent: stripModeMarker(text)\n\t\t});", 1),
    ("async function serializeMessagesWithImages(messages, images) {",
     "async function serializeMessagesWithImages(messages, images, forced) {", 1),
    ("\tassertSupportedImageRoles(messages);\n\tconst wire = [];\n\tlet pendingToolImages = [];",
     "\tassertSupportedImageRoles(messages);\n\tconst wire = [];\n\tconst mode = resolveMode(messages, forced);\n"
     "\tconst segStart = lastFinalReplyIndex(messages) + 1;\n\tlet pendingToolImages = [];", 1),
    ("\t\t\tflushToolImages();\n\t\t\twire.push(serializeAssistant(message));",
     "\t\t\tflushToolImages();\n\t\t\twire.push(serializeAssistant(message, shouldBackfill(message, messageIndex, segStart, mode)));", 1),
    ("\t\tconst content = userContent(await contentParts(regular, images, messageIndex + 1, nextImage));",
     "\t\tconst content = stripMarkerFromContent(userContent(await contentParts(regular, images, messageIndex + 1, nextImage)));", 1),
    ("messages.push(...serializeMessages(options.messages));",
     "messages.push(...serializeMessages(options.messages, readBackfillMode()));", 1),
    ("messages.push(...await serializeMessagesWithImages(requestMessages, images));",
     "messages.push(...await serializeMessagesWithImages(requestMessages, images, readBackfillMode()));", 1),
]


def targets():
    return sorted(glob.glob(GLOB))


def version_of(path: Path) -> str:
    try:
        pkg = path.parent.parent / "package.json"
        return str(json.loads(pkg.read_text(encoding="utf-8")).get("version", "unknown"))
    except Exception:
        return "unknown"


def node_check(path: Path) -> bool:
    try:
        subprocess.run(["node", "--check", str(path)], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def handle(path: Path, mode: str) -> str:
    src = path.read_text(encoding="utf-8")
    patched = MARK in src
    ver = version_of(path)

    if mode == "check":
        lines = [f"version {ver}"]
        if patched:
            lines.append("status: patched")
        else:
            miss = [old for old, _, _ in OPS if old not in src]
            lines.append("status: ready to patch" if not miss else "status: ANCHORS MISSING - do not touch")
            for m in miss:
                lines.append(f"  missing: {m.splitlines()[0][:70]}")
        return "\n    ".join(lines)

    if mode == "revert":
        if not patched:
            return "no patch present, nothing to revert"
        out = src
        for old, new, _ in OPS:
            if new in out:
                out = out.replace(new, old, 1)
            else:
                return f"REFUSED: cannot find the fragment to reverse: {new.splitlines()[0][:60]}"
        path.write_text(out, encoding="utf-8")
        return "reverted (reverse substitution)" + (", syntax OK" if node_check(path) else ", WARNING: syntax check failed")

    if patched:
        return f"already patched (version {ver}), skipping"
    for old, _, cnt in OPS:
        n = src.count(old)
        if n != cnt:
            return f"REFUSED: anchor occurs {n} time(s), expected {cnt}:\n      {old.splitlines()[0][:80]}"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = Path(str(path) + f".bak-backfill-{ver}-{stamp}")
    shutil.copy2(path, backup)
    out = src
    for old, new, _ in OPS:
        out = out.replace(old, new, 1)
    path.write_text(out, encoding="utf-8")
    if not node_check(path):
        shutil.copy2(backup, path)
        return "FAILED syntax check, rolled back"
    return f"patched (version {ver}), syntax OK\n      backup: {backup.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Narrow DeepSeek reasoning_content backfill")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    mode = "revert" if args.revert else ("check" if args.check else "apply")
    ts = targets()
    if not ts:
        print("dsh-llm-deepseek index.js not found")
        return 2
    for t in ts:
        print(f"{t}\n    {handle(Path(t), mode)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
