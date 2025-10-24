import json
import sqlite3
from pathlib import Path

from api_server.db import EventRecord, EventStore


def test_event_store_insert(tmp_path):
    db_path = tmp_path / "events.sqlite"
    store = EventStore(db_path)

    rec = EventRecord(
        request_id="req-1",
        camera_id="cam01",
        ts="2025-10-24T12:00:00Z",
        plate="B 1234 CD",
        plate_conf=0.92,
        det_conf=0.85,
        valid=True,
        bbox=(10, 20, 110, 60),
        track_id=7,
        frame_id=123,
        snapshot_path="snapshots/cam01/2025/10/24/req-1_track7.jpg",
        char_confs=[0.95, 0.94, 0.91],
        status="ok",
        raw_event={"plate": "B 1234 CD", "plate_conf": 0.92},
    )

    inserted = store.insert_many([rec])
    assert inserted == 1

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT request_id, camera_id, plate, char_confs, raw_event FROM events").fetchone()
    assert row is not None
    assert row[0] == "req-1"
    assert row[1] == "cam01"
    assert row[2] == "B 1234 CD"
    assert json.loads(row[3]) == [0.95, 0.94, 0.91]
    assert json.loads(row[4])["plate"] == "B 1234 CD"
    conn.close()

    store.close()

