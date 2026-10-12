#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick the right DSH session archive file — one rule, used everywhere.

Why this exists (2026-09-30)
----------------------------
DSH migrated its session format from v3 to v4. The migration **does not delete
or rewrite the v3 file** — it publishes a new `session.v4.jsonl.zstd` next to it.
So from that moment on, any `glob("session.v3.jsonl.zstd")` **silently returns
stale data**: the v3 file is frozen at the moment of migration and every new
event goes to the v4 file. It does not raise. It does not warn. It just quietly
stops showing you anything recent.

What to do instead: **take the highest generation**, by version *number*.

`session.jsonl.zstd` (no version) counts as generation 0.

⚠️ Do **not** use `sorted(glob("session*.jsonl.zstd"))[-1]`. That compares the
filenames lexicographically, so it happens to work for v3 vs v4 but breaks the
day a v10 appears (`"…v10…" < "…v9…"`) — and it breaks silently, which is the
whole problem this module is about.

⚠️ **`seq` is not an identity across generations.** The v3→v4 migration inserts
a synthetic `turn/end` for each interrupted turn, which shifts every later
`seq` by +1 (measured on real archives; one archive shifted by +2). Anything
that keys on `seq` — a dedup ledger, a "have I seen this already" set — has to
be re-keyed when the generation changes. Prefer keying on content.

Usage
-----
    from dsh_archive import latest_for, iter_archives, read_events

    p = latest_for(Path.home() / ".dsh" / "sessions" / slug / session_id)
    for slug, session_id, path in iter_archives():
        for event in read_events(path):
            ...
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# `DSH_SESSIONS_DIR` lets every tool in this repo point at a copied/archived
# session tree instead of the live one — the same convention the other tools use.
SESSIONS = Path(os.environ.get("DSH_SESSIONS_DIR", Path.home() / ".dsh" / "sessions"))

_VERSIONED = re.compile(r"^session\.v(\d+)\.jsonl\.zstd$")
_UNVERSIONED = re.compile(r"^session\.jsonl\.zstd$")


def generation(path: Path) -> int:
    """Format generation of one archive file; -1 if the name is unrecognized."""
    m = _VERSIONED.match(path.name)
    if m:
        return int(m.group(1))
    if _UNVERSIONED.match(path.name):
        return 0
    return -1


def candidates(session_dir: Path) -> List[Path]:
    """Every generation present in one session directory, oldest first."""
    if not session_dir.is_dir():
        return []
    out = [p for p in session_dir.glob("session*.jsonl.zstd") if p.is_file()]
    return sorted(out, key=lambda p: (generation(p), p.name))


def latest_for(session_dir: Path) -> Optional[Path]:
    """The newest generation in that directory, or None."""
    c = candidates(session_dir)
    return c[-1] if c else None


def iter_archives(root: Optional[Path] = None) -> Iterator[Tuple[str, str, Path]]:
    """Yield `(workspace_slug, session_id, newest_archive_path)` for every session.

    Only the newest generation is yielded. Use `candidates()` if you need all.
    """
    base = Path(root) if root else SESSIONS
    if not base.is_dir():
        return
    for slug_dir in sorted(base.iterdir()):
        if not slug_dir.is_dir():
            continue
        for session_dir in sorted(slug_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            p = latest_for(session_dir)
            if p is not None:
                yield slug_dir.name, session_dir.name, p


def read_events(path: Path) -> List[Dict[str, Any]]:
    """Decompress and parse one archive into a list of events.

    ⚠️ Split on `"\\n"`, **never** the `splitlines` method. the `splitlines` method also breaks
    on U+2028 / U+2029 / \\x85, all of which are legal inside JSON strings
    (JSON only requires escaping code points below 0x20), so it will split
    records in half without complaining. Real archives contain such records.
    """
    raw = subprocess.run(["zstd", "-dc", str(path)], capture_output=True)
    text = raw.stdout.decode("utf-8", "replace")
    out: List[Dict[str, Any]] = []
    for ln in text.split("\n"):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def read_lines(path: Path) -> str:
    """Decompressed text (use with `in` for cheap substring filtering)."""
    raw = subprocess.run(["zstd", "-dc", str(path)], capture_output=True)
    return raw.stdout.decode("utf-8", "replace")


if __name__ == "__main__":
    rows = list(iter_archives())
    print(f"session root: {SESSIONS}")
    print(f"{len(rows)} archives (newest generation only)\n")
    vers: Dict[int, int] = {}
    for _slug, _sid, path in rows:
        g = generation(path)
        vers[g] = vers.get(g, 0) + 1
    for g in sorted(vers):
        print(f"  generation {g}: {vers[g]}")
    multi = [(sid, candidates(p.parent)) for _s, sid, p in rows if len(candidates(p.parent)) > 1]
    print(f"\n{len(multi)} session(s) have more than one generation on disk")
    for sid, c in multi:
        names = " + ".join(f"{p.name}({p.stat().st_size:,}B)" for p in c)
        print(f"  {sid[:34]:36s} {names}")
