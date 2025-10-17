"""Track-level OCR aggregation and event emission.

Implements temporal majority voting per track and emits stabilized events that
match the JSON schema in plan.md §9. Designed to be reusable by DeepStream
bridge code or any CPU pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional, Tuple

from ocr_service.postprocess import MajorityVote


@dataclass
class _TrackState:
    voter: MajorityVote
    last_emitted_text: str = ""
    last_frame_id: int = -1
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_conf: float = 0.0
    updated_at_frame: int = -1


@dataclass
class TrackAggregator:
    """Aggregate OCR results per track and emit stabilized events.

    - Uses MajorityVote(window) to smooth OCR outputs over time.
    - Emits an event once `min_consensus` is reached for a text that differs
      from the last emitted value for the track.
    - Provides `evict_stale` to prune inactive tracks.
    """

    window: int = 8
    min_consensus: int = 3
    max_inactive_frames: int = 120
    state: Dict[int, _TrackState] = field(default_factory=dict)

    def update(
        self,
        *,
        track_id: int,
        text: str,
        conf: float,
        bbox: Tuple[int, int, int, int],  # x, y, w, h
        frame_id: int,
        camera_id: str = "cam01",
        ts_iso: Optional[str] = None,
        char_confs: Optional[list[float]] = None,
        det_ms: Optional[float] = None,
        ocr_ms: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        st = self.state.get(track_id)
        if st is None:
            st = _TrackState(voter=MajorityVote(window=self.window))
            self.state[track_id] = st

        st.voter.add(text, conf)
        st.last_frame_id = frame_id
        st.last_bbox = bbox
        st.last_conf = conf
        st.updated_at_frame = frame_id

        best = st.voter.best()
        if best is None:
            return None

        best_text, best_conf = best

        # Require at least min_consensus occurrences in the window
        # MajorityVote does not expose counts; approximate by checking the
        # frequency via internal buffer. This keeps it simple for now.
        # If API changes, adapt accordingly.
        #
        # Implementation detail: count occurrences of best_text in buffer.
        try:
            buf: Deque = st.voter.buf  # type: ignore[attr-defined]
            count = sum(1 for v in buf if getattr(v, "text", None) == best_text)
        except Exception:
            count = self.min_consensus  # fallback, emit if different

        if count < self.min_consensus:
            return None

        if best_text and best_text != st.last_emitted_text:
            ts = ts_iso or datetime.now(timezone.utc).isoformat()
            event = {
                "schema_version": "1.0",
                "camera_id": camera_id,
                "ts": ts,
                "plate": best_text,
                "plate_conf": float(best_conf),
                "char_confs": char_confs or [],
                "bbox": list(map(int, bbox)),
                "track_id": int(track_id),
                "frame_id": int(frame_id),
                "snapshots": {},  # filled by caller if desired
                "processing": {
                    "det_ms": float(det_ms) if det_ms is not None else None,
                    "ocr_ms": float(ocr_ms) if ocr_ms is not None else None,
                    "total_ms": None,
                },
            }
            st.last_emitted_text = best_text
            return event
        return None

    def evict_stale(self, current_frame: int) -> None:
        """Remove tracks that have been inactive for more than max_inactive_frames."""
        to_del = [
            tid
            for tid, st in self.state.items()
            if current_frame - (st.updated_at_frame if st.updated_at_frame >= 0 else current_frame)
            > self.max_inactive_frames
        ]
        for tid in to_del:
            self.state.pop(tid, None)

