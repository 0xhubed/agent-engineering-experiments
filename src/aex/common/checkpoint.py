"""Resumable run state. One row per (question, arm, navigator, answerer).

Multi-day runs on the GB10 boxes must survive reboots and session ends;
every put() is its own committed transaction.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def row_key(qid: str, arm: str, navigator: str, answerer: str) -> str:
    return "|".join((qid, arm, navigator, answerer))


class Checkpoint:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS rows ("
            " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            " key TEXT UNIQUE NOT NULL,"
            " row TEXT NOT NULL)"
        )
        self._db.commit()

    def done(self, key: str) -> bool:
        return self._db.execute("SELECT 1 FROM rows WHERE key = ?", (key,)).fetchone() is not None

    def get(self, key: str) -> dict | None:
        found = self._db.execute("SELECT row FROM rows WHERE key = ?", (key,)).fetchone()
        return json.loads(found[0]) if found else None

    def put(self, key: str, row: dict) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO rows (key, row) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET row = excluded.row",
                (key, json.dumps(row, sort_keys=True)),
            )

    def rows(self) -> list[dict]:
        return [json.loads(r) for (r,) in self._db.execute("SELECT row FROM rows ORDER BY seq")]

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM rows").fetchone()[0]

    def close(self) -> None:
        self._db.close()
