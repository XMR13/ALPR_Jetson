"""OCR post-processing for Indonesian license plates.

Functions here implement:
- regex validation per plan (uppercase, loose spacing)
- heuristic ambiguous character fixes (O/0, I/1, S/5, B/8, G/6, Z/2)
- lightweight temporal majority voting over recent OCRs

These utilities are CPU-only and safe for Jetson NX.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Sequence, Tuple


DEFAULT_REGEX = r"^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$"
AMBIGUOUS_PAIRS: Sequence[Tuple[str, str]] = (
    ("O", "0"),
    ("I", "1"),
    ("S", "5"),
    ("B", "8"),
    ("G", "6"),
    ("Z", "2"),
)


def _clean(text: str) -> str:
    t = text.strip().upper()
    # keep alnum and space
    t = re.sub(r"[^A-Z0-9 ]+", "", t)
    # collapse spaces
    t = re.sub(r"\s+", " ", t)
    return t


def _split_segments(t: str) -> Tuple[str, str, str]:
    """Split into PREFIX letters, NUMBER digits, SUFFIX letters.

    Heuristic for Indonesian plates: [A-Z]{1,2} [0-9]{1,4} [A-Z]{0,3}
    Returns possibly empty suffix.
    """
    t = _clean(t)
    # If spaces present and grouping looks sane, trust them
    parts = t.split(" ") if " " in t else []
    if 2 <= len(parts) <= 3:
        prefix = re.sub(r"[^A-Z]", "", parts[0])
        number = re.sub(r"[^0-9]", "", parts[1])
        suffix = re.sub(r"[^A-Z]", "", parts[2]) if len(parts) == 3 else ""
        return prefix, number, suffix

    # Otherwise derive by boundary between letters and digits
    prefix = re.match(r"^[A-Z]{1,2}", t)
    p = prefix.group(0) if prefix else ""
    rest = t[len(p) :]
    number_match = re.match(r"^[0-9]{1,4}", rest)
    n = number_match.group(0) if number_match else ""
    suffix = re.sub(r"[^A-Z]", "", rest[len(n) :])
    return p, n, suffix


def _apply_ambiguity(prefix: str, number: str, suffix: str) -> Tuple[str, str, str]:
    """Fix ambiguous characters based on segment type (letter vs digit)."""
    # Map that converts digits to letters in letter segments
    to_letter = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z"}
    # Map that converts letters to digits in numeric segment
    to_digit = {"O": "0", "I": "1", "S": "5", "B": "8", "G": "6", "Z": "2"}

    p2 = "".join(to_letter.get(c, c) for c in prefix)
    n2 = "".join(to_digit.get(c, c) for c in number)
    s2 = "".join(to_letter.get(c, c) for c in suffix)
    return p2, n2, s2


def postprocess_indonesia(text: str, allowed_prefix: Optional[Iterable[str]] = None, regex: str = DEFAULT_REGEX) -> Tuple[str, bool]:
    """Normalize OCR text to Indonesian plate format and report validity.

    Returns (normalized_text, is_valid_by_regex_and_prefix)
    """
    t = _clean(text)
    p, n, s = _split_segments(t)
    p, n, s = _apply_ambiguity(p, n, s)
    out = " ".join([x for x in (p, n, s) if x])

    is_match = re.match(regex, out) is not None
    if allowed_prefix is not None and p:
        pref_ok = p in set(x.upper() for x in allowed_prefix)
        valid = is_match and pref_ok
    else:
        valid = is_match
    return out, valid


@dataclass
class VoteItem:
    text: str
    conf: float


class MajorityVote:
    """Keep a rolling window of OCR outputs and return the consensus.

    - Majority by frequency; ties broken by avg confidence.
    - Window defaults to 8 per plan.md.
    """

    def __init__(self, window: int = 8):
        if window <= 0:
            raise ValueError("window must be > 0")
        self.window: int = int(window)
        self.buf: Deque[VoteItem] = deque(maxlen=self.window)

    def add(self, text: str, conf: float) -> None:
        self.buf.append(VoteItem(text=text, conf=float(conf)))

    def best(self) -> Optional[Tuple[str, float]]:
        if not self.buf:
            return None
        counts = Counter(v.text for v in self.buf)
        # candidates sorted by count desc, then avg conf desc
        def avg_conf(t: str) -> float:
            vals = [v.conf for v in self.buf if v.text == t]
            return sum(vals) / max(1, len(vals))

        items = sorted(counts.items(), key=lambda kv: (kv[1], avg_conf(kv[0])), reverse=True)
        text_top = items[0][0]
        return text_top, avg_conf(text_top)

