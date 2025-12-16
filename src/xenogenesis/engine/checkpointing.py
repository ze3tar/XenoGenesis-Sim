"""Checkpoint utilities using SQLite + compressed blob."""
from __future__ import annotations
import sqlite3
import zlib
import json
from pathlib import Path
from typing import Any, Dict


def save_checkpoint(path: Path, state: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS checkpoints(id INTEGER PRIMARY KEY, payload BLOB)")
    payload = zlib.compress(json.dumps(state).encode("utf-8"))
    conn.execute("INSERT INTO checkpoints(payload) VALUES (?)", (payload,))
    conn.commit()
    conn.close()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(path)
    cur = conn.execute("SELECT payload FROM checkpoints ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if not row:
        raise FileNotFoundError("No checkpoint entries")
    return json.loads(zlib.decompress(row[0]).decode("utf-8"))
