"""Tamper-evident local audit trail.

Every analysis appends one JSON line recording who ran what, on which input
(by SHA-256, never the images themselves), with which detector version, and a
summary of the result. Each entry includes the hash of the previous entry, so
editing or deleting history breaks the chain - `verify()` detects it. This
supports clinical audit, research reproducibility and the traceability expected
for a regulated medical device.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def _hash(entry: dict) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def _last_hash(path: Path) -> str:
    if not path.exists():
        return GENESIS
    last = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last)["entry_hash"] if last else GENESIS


def append(path: Path | str, event: str, **fields) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "user": fields.pop("user", None) or getpass.getuser(),
        "event": event,
        **fields,
        "prev_hash": _last_hash(path),
    }
    entry["entry_hash"] = _hash(entry)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def verify(path: Path | str) -> tuple[bool, str]:
    prev = GENESIS
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            stored = entry.pop("entry_hash")
            if entry["prev_hash"] != prev:
                return False, f"line {n}: chain broken (previous entry missing or altered)"
            if _hash(entry) != stored:
                return False, f"line {n}: entry contents altered"
            prev = stored
    return True, "audit log intact"
