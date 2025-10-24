import csv
import json
import sqlite3
from pathlib import Path

import pytest

from tools.eval_e2e import (
    EvaluationResult,
    Prediction,
    char_error_rate,
    evaluate_predictions,
    levenshtein_distance,
    load_ground_truth_csv,
    load_predictions_from_sqlite,
)


def _make_sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "events.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE events (
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
        rows = [
            (
                "req-1",
                "cam01",
                "2025-10-24T10:00:00Z",
                "ok",
                "B 1234 CD",
                0.90,
                0.90,
                1,
                0,
                0,
                10,
                10,
                7,
                42,
                str(tmp_path / "snap1.jpg"),
                json.dumps([0.95, 0.93, 0.92]),
                json.dumps({}),
                "2025-10-24T10:00:00Z",
            ),
            (
                "req-2",
                "cam01",
                "2025-10-24T10:05:00Z",
                "ok",
                "B 12X4 CD",
                0.70,
                0.70,
                1,
                0,
                0,
                10,
                10,
                8,
                43,
                str(tmp_path / "snap2.jpg"),
                json.dumps([0.7, 0.6, 0.5]),
                json.dumps({}),
                "2025-10-24T10:05:00Z",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO events (
                request_id, camera_id, ts, status, plate, plate_conf, det_conf, valid,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, track_id, frame_id, snapshot_path,
                char_confs, raw_event, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    # Touch snapshot files for copy tests (empty but present)
    for idx in (1, 2):
        (tmp_path / f"snap{idx}.jpg").write_bytes(b"")

    return db_path


def _write_ground_truth(tmp_path: Path) -> Path:
    gt_path = tmp_path / "gt.csv"
    with gt_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "plate_gt"])
        writer.writeheader()
        writer.writerow({"request_id": "req-1", "plate_gt": "B 1234 CD"})
        writer.writerow({"request_id": "req-2", "plate_gt": "B 1234 CD"})
    return gt_path


def test_levenshtein_basic():
    assert levenshtein_distance("ABC", "ABC") == 0
    assert levenshtein_distance("ABC", "ABD") == 1
    assert levenshtein_distance("ABC", "ABCD") == 1


def test_char_error_rate_handles_empty_truth():
    assert char_error_rate("", "") == 0.0
    assert char_error_rate("ABC", "") == pytest.approx(3.0)


def test_loaders_and_evaluation(tmp_path):
    db_path = _make_sqlite_db(tmp_path)
    gt_path = _write_ground_truth(tmp_path)

    gt = load_ground_truth_csv(gt_path, id_column="request_id", plate_column="plate_gt")
    preds = load_predictions_from_sqlite(db_path)

    res = evaluate_predictions(preds, gt, low_conf_threshold=0.8)
    assert isinstance(res, EvaluationResult)
    assert res.total == 2
    # One exact match, one mismatch
    assert res.matched == 1
    assert res.exact_match_pct == pytest.approx(50.0)
    assert len(res.low_conf_predictions) == 1  # req-2 below threshold
    assert res.missing_gt == []
    assert "cam01" in res.per_camera
    assert res.per_camera["cam01"]["exact_match_pct"] == pytest.approx(50.0)
