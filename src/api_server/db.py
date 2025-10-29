"""SQLite persistence helpers for ALPR events.

Provides a lightweight event store tailored for Jetson deployments. The schema
tracks per-plate metadata plus a JSON snapshot of the event payload so future
updates can evolve without migrations.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Optional, Sequence, Dict


@dataclass(frozen=True)
class EventRecord:
    request_id: str
    camera_id: str
    ts: str
    plate: str
    plate_conf: float
    det_conf: float
    valid: bool
    bbox: Sequence[int]
    track_id: Optional[int]
    frame_id: Optional[int]
    snapshot_path: Optional[str]
    char_confs: Sequence[float]
    status: str
    raw_event: Dict[str, Any]


class EventStore:
    """Minimal SQLite wrapper with thread-safe inserts."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            status TEXT NOT NULL,
            plate TEXT,
            plate_conf REAL,
            det_conf REAL,
            valid INTEGER,
            bbox_x1 INTEGER,
            bbox_y1 INTEGER,
            bbox_x2 INTEGER,
            bbox_y2 INTEGER,
            track_id INTEGER,
            frame_id INTEGER,
            snapshot_path TEXT,
            char_confs TEXT,
            raw_event TEXT NOT NULL,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        """
        with self._conn:  # type: ignore[call-arg]
            self._conn.execute(ddl)

    def insert_many(self, records: Iterable[EventRecord]) -> int:
        rows = []
        for rec in records:
            bbox = list(rec.bbox)
            if len(bbox) != 4:
                raise ValueError("bbox must have four values")
            rows.append(
                (
                    rec.request_id,
                    rec.camera_id,
                    rec.ts,
                    rec.status,
                    rec.plate,
                    float(rec.plate_conf),
                    float(rec.det_conf),
                    1 if rec.valid else 0,
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    rec.track_id if rec.track_id is not None else None,
                    rec.frame_id if rec.frame_id is not None else None,
                    rec.snapshot_path,
                    json.dumps(list(rec.char_confs)),
                    json.dumps(rec.raw_event),
                )
            )

        if not rows:
            return 0

        with self._lock:
            with self._conn:  # type: ignore[call-arg]
                self._conn.executemany(
                    """
                    INSERT INTO events (
                        request_id, camera_id, ts, status, plate, plate_conf, det_conf,
                        valid, bbox_x1, bbox_y1, bbox_x2, bbox_y2, track_id, frame_id,
                        snapshot_path, char_confs, raw_event
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["EventRecord", "EventStore"]
