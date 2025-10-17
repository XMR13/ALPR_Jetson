from pipeline.track_aggregator import TrackAggregator


def test_track_aggregator_emits_after_consensus():
    agg = TrackAggregator(window=5, min_consensus=3)

    # Feed mixed texts for same track; only emit when 3x consensus is reached
    ev = agg.update(track_id=1, text="B 1234 CD", conf=0.8, bbox=(10, 20, 100, 40), frame_id=1)
    assert ev is None
    ev = agg.update(track_id=1, text="B 1234 CD", conf=0.9, bbox=(12, 22, 100, 40), frame_id=2)
    assert ev is None
    ev = agg.update(track_id=1, text="B 1234 CO", conf=0.6, bbox=(12, 22, 100, 40), frame_id=3)
    assert ev is None
    ev = agg.update(track_id=1, text="B 1234 CD", conf=0.85, bbox=(14, 24, 100, 40), frame_id=4)
    assert ev is not None
    assert ev["plate"] == "B 1234 CD"
    assert ev["track_id"] == 1


def test_track_aggregator_emits_on_change():
    agg = TrackAggregator(window=5, min_consensus=2)
    # reach consensus on first text
    _ = agg.update(track_id=7, text="A 1 B", conf=0.7, bbox=(0, 0, 1, 1), frame_id=1)
    ev = agg.update(track_id=7, text="A 1 B", conf=0.8, bbox=(0, 0, 1, 1), frame_id=2)
    assert ev is not None
    # now change to a new stable text
    ev2 = agg.update(track_id=7, text="A 1 C", conf=0.9, bbox=(0, 0, 1, 1), frame_id=3)
    assert ev2 is None
    ev3 = agg.update(track_id=7, text="A 1 C", conf=0.9, bbox=(0, 0, 1, 1), frame_id=4)
    assert ev3 is not None
    assert ev3["plate"] == "A 1 C"

