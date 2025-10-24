"""Seed a small demo events DB and ground-truth CSV for evaluator testing.

This script creates:
- export/events.sqlite with a minimal `events` table and 3 demo rows
- export/snapshots/* with placeholder JPEGs for those rows
- data/labels/e2e_gt_demo.csv as ground-truth

Run:
  python3 tools/seed_events_demo.py
Then:
  python3 tools/eval_e2e.py \
    --events export/events.sqlite \
    --ground-truth data/labels/e2e_gt_demo.csv \
    --low-conf-threshold 0.85 \
    --low-conf-dir export/low_confidence_demo \
    --output-json export/metrics/e2e_metrics_demo.json
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def main() -> None:
    db = Path("export/events.sqlite")
    snaps_dir = Path("export/snapshots")
    snaps_dir.mkdir(parents=True, exist_ok=True)
    db.parent.mkdir(parents=True, exist_ok=True)

    # Minimal schema aligned with src/api_server/db.py
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                camera_id TEXT,
                ts TEXT,
                status TEXT,
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
                raw_event TEXT,
                created_at TEXT
            )
            """
        )

        # Create placeholder snapshots
        s1 = snaps_dir / "req-001_seq0.jpg"
        s2 = snaps_dir / "req-002_seq0.jpg"
        s3 = snaps_dir / "req-003_seq0.jpg"
        for p in (s1, s2, s3):
            p.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG markers

        rows = [
            (
                "req-001",
                "cam01",
                "2025-10-24T10:00:00Z",
                "ok",
                "B 1234 CD",
                0.95,
                0.92,
                1,
                10,
                20,
                110,
                60,
                1,
                1001,
                str(s1),
                json.dumps([0.98, 0.97, 0.96]),
                json.dumps({}),
                "2025-10-24T10:00:01Z",
            ),
            (
                "req-002",
                "cam01",
                "2025-10-24T10:02:00Z",
                "ok",
                "B 12X4 CD",
                0.70,
                0.70,
                1,
                11,
                21,
                111,
                61,
                2,
                1002,
                str(s2),
                json.dumps([0.7, 0.6, 0.5]),
                json.dumps({}),
                "2025-10-24T10:02:01Z",
            ),
            (
                "req-003",
                "cam02",
                "2025-10-24T10:03:00Z",
                "ok",
                "A 8628 GL",
                0.93,
                0.90,
                1,
                5,
                15,
                100,
                55,
                3,
                1003,
                str(s3),
                json.dumps([0.95, 0.94, 0.93]),
                json.dumps({}),
                "2025-10-24T10:03:01Z",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO events (
                request_id, camera_id, ts, status, plate, plate_conf, det_conf, valid,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, track_id, frame_id, snapshot_path,
                char_confs, raw_event, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    # Ground-truth CSV
    gt = Path("data/labels/e2e_gt_demo.csv")
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_text(
        "request_id,plate\nreq-001,B 1234 CD\nreq-002,B 1234 CD\nreq-003,A 8628 GL\n",
        encoding="utf-8",
    )

    print("Seeded demo events at export/events.sqlite and GT at data/labels/e2e_gt_demo.csv")


if __name__ == "__main__":
    main()

