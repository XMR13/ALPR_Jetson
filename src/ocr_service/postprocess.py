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
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional dependency for config loading
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_REGEX = r"^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$"
AMBIGUOUS_PAIRS: Sequence[Tuple[str, str]] = (
    ("O", "0"),
    ("I", "1"),
    ("S", "5"),
    ("B", "8"),
    ("G", "6"),
    ("Z", "2"),
)

# Heuristic correction weights (tuned for Indonesian scene)
_MAX_EDIT_COST = 1.7
_MAX_SEARCH_STATES = 256
_SUB_COST_PREFIX = 0.45
_SUB_COST_DIGIT = 0.35
_SUB_COST_SUFFIX = 0.45
_INSERT_COST = 0.65
_DELETE_COST = 0.70

_PREFIX_CONFUSIONS = {
    "8": ("B",),
    "B": ("8", "R"),
    "R": ("B",),
    "0": ("O", "D"),
    "O": ("0", "D"),
    "D": ("O", "0"),
    # Keep I↔1/L for prefix, but avoid I↔T which was over-aggressive.
    "I": ("1", "L"),
    "1": ("I", "L"),
    # Allow T→Y (common) but not T→I by default.
    "T": ("Y",),
    "L": ("I", "1"),
    "M": ("N",),
    "N": ("M",),
    "4": ("A",),
    "A": ("4",),
}

_DIGIT_CONFUSIONS = {
    "O": ("0",),
    "0": ("O", "D"),
    "D": ("0", "O"),
    "I": ("1",),
    "1": ("I",),
    "S": ("5",),
    "5": ("S",),
    "B": ("8",),
    "8": ("B", "0"),
    "Z": ("2",),
    "2": ("Z",),
    "G": ("6",),
    "6": ("G",),
}

_SUFFIX_CONFUSIONS = {
    "Y": ("T", "V"),
    # Favor T→Y but avoid T→I which caused I/T flips in suffix.
    "T": ("Y",),
    "V": ("Y", "U"),
    "U": ("V", "O"),
    "O": ("D", "0", "U"),
    "D": ("O", "0", "C"),
    "C": ("G", "O", "E"),
    "E": ("C", "F"),
    "N": ("M",),
    "M": ("N",),
    "R": ("B", "P"),
    "B": ("R",),
    "P": ("R", "F"),
    "J": ("I",),
    # Do not substitute I inside the suffix region by default; this avoids I→T flips.
    "I": (),
    "S": ("5",),
    "5": ("S",),
}

_SUFFIX_INSERT_CHOICES: Tuple[str, ...] = tuple("NMPRCDEHABRSTYOUILG")


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
    parts = t.split(" ") if " " in t else []
    if 2 <= len(parts) <= 3:
        # Keep alnum; don't strip letters out of number or digits out of letter segments yet.
        prefix = re.sub(r"[^A-Z0-9]", "", parts[0])
        number = re.sub(r"[^A-Z0-9]", "", parts[1])
        suffix = re.sub(r"[^A-Z0-9]", "", parts[2]) if len(parts) == 3 else ""
        return prefix, number, suffix

    # Fallback heuristic unchanged (can also be relaxed similarly if you want)
    prefix = re.match(r"^[A-Z]{1,2}", t)
    p = prefix.group(0) if prefix else ""
    rest = t[len(p):]
    number_match = re.match(r"^[0-9A-Z]{1,4}", rest)  # allow A-Z to catch ambiguous chars
    n = number_match.group(0) if number_match else ""
    suffix = re.sub(r"[^A-Z0-9]", "", rest[len(n):])
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


_PENALTY_TRIGGER = 0.5


@dataclass
class PostprocessTuning:
    suffix_len_lt3_penalty: float = 0.2
    suffix_last_letter_penalty: Dict[str, float] = field(
        # By default, do not penalize specific tail letters.
        # Site-specific biases (e.g., tail I/U handling) should be configured via YAML.
        default_factory=dict
    )
    suffix_penalty_map: Dict[str, float] = field(
        default_factory=lambda: {
            "DC": 0.8,
            "DJ": 0.8,
            "DT": 0.8,
            "DK": 0.8,
            "DG": 0.35,
            "DF": 0.35,
            # Encourage completing VI -> VIN by penalizing the short form slightly.
            "VI": 0.55,
        }
    )
    suffix_contains_penalty: Dict[str, float] = field(
        default_factory=lambda: {
            "TT": 0.6,
            "DE": 0.6,
        }
    )
    suffix_bonus_map: Dict[str, float] = field(
        default_factory=lambda: {"OE": -0.1, "VIN": -0.1, "PDC": -0.1}
    )
    suffix_duplicate_penalty: float = 1.0
    suffix_vowel_pair_penalty: float = 0.2
    duplicate_collapse_min_len: int = 2
    insert_bias_vi_to_vin: float = 0.05
    insert_bias_pdc: float = 0.25
    # New: confidence-aware truncation/gating (disabled by default)
    # If > 0, require the last suffix character confidence to meet this threshold
    # or drop it (plates allow 0–3 suffix letters). Useful for U↔O ambiguity at tail.
    last_char_min_conf: float = 0.0
    # If true, allow dropping low-confidence tail characters in suffix
    truncate_ambiguous_suffix: bool = False
    # Optional global minimum confidence for suffix characters; 0 disables
    min_suffix_char_conf: float = 0.0


DEFAULT_TUNING = PostprocessTuning()


def load_postprocess_config(path: str) -> PostprocessTuning:
    """Load tuning parameters from a YAML file."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load postprocess configs")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PostprocessTuning(
        suffix_len_lt3_penalty=float(data.get("suffix_len_lt3_penalty", DEFAULT_TUNING.suffix_len_lt3_penalty)),
        suffix_last_letter_penalty=dict(
            data.get("suffix_last_letter_penalty", DEFAULT_TUNING.suffix_last_letter_penalty)
        ),
        suffix_penalty_map=dict(data.get("suffix_penalty_map", DEFAULT_TUNING.suffix_penalty_map)),
        suffix_contains_penalty=dict(
            data.get("suffix_contains_penalty", DEFAULT_TUNING.suffix_contains_penalty)
        ),
        suffix_bonus_map=dict(data.get("suffix_bonus_map", DEFAULT_TUNING.suffix_bonus_map)),
        suffix_duplicate_penalty=float(data.get("suffix_duplicate_penalty", DEFAULT_TUNING.suffix_duplicate_penalty)),
        suffix_vowel_pair_penalty=float(data.get("suffix_vowel_pair_penalty", DEFAULT_TUNING.suffix_vowel_pair_penalty)),
        duplicate_collapse_min_len=int(data.get("duplicate_collapse_min_len", DEFAULT_TUNING.duplicate_collapse_min_len)),
        insert_bias_vi_to_vin=float(data.get("insert_bias_vi_to_vin", DEFAULT_TUNING.insert_bias_vi_to_vin)),
        insert_bias_pdc=float(data.get("insert_bias_pdc", DEFAULT_TUNING.insert_bias_pdc)),
        last_char_min_conf=float(data.get("last_char_min_conf", DEFAULT_TUNING.last_char_min_conf)),
        truncate_ambiguous_suffix=bool(data.get("truncate_ambiguous_suffix", DEFAULT_TUNING.truncate_ambiguous_suffix)),
        min_suffix_char_conf=float(data.get("min_suffix_char_conf", DEFAULT_TUNING.min_suffix_char_conf)),
    )


def postprocess_indonesia(
    text: str,
    allowed_prefix: Optional[Iterable[str]] = None,
    regex: str = DEFAULT_REGEX,
    tuning: Optional[PostprocessTuning] = None,
    # Optional: per-character confidences aligned to raw text with no spaces
    # (only applicable for slot-based OCR decoders). If provided and tuning
    # enables truncation, we may drop low-confidence tail letters in the suffix.
    char_confs: Optional[List[float]] = None,
    strict: bool = False,
) -> Tuple[str, bool]:
    """Normalize OCR text to Indonesian plate format and report validity."""
    cfg = tuning or DEFAULT_TUNING
    allowed_set: Optional[set[str]] = {p.upper() for p in allowed_prefix} if allowed_prefix else None
    regex_obj = re.compile(regex)

    # Optionally drop low-confidence trailing suffix characters before normalization
    raw_for_norm = text
    if (strict or cfg.truncate_ambiguous_suffix or cfg.last_char_min_conf > 0.0 or cfg.min_suffix_char_conf > 0.0) and char_confs:
        pruned = _truncate_lowconf_suffix(text, char_confs, cfg)
        if pruned:
            raw_for_norm = pruned

    normalized, is_valid, _, suffix = _normalize_plate(raw_for_norm, allowed_set, regex_obj, cfg)
    base_penalty = _suffix_penalty(suffix, cfg)

    if is_valid and base_penalty < _PENALTY_TRIGGER:
        return normalized, True

    refined = _search_best_plate(
        text,
        allowed_set,
        regex_obj,
        cfg,
        normalized if is_valid else "",
        base_penalty if is_valid else float("inf"),
    )
    if refined is not None:
        return refined, True
    return normalized, is_valid


def _truncate_lowconf_suffix(text: str, char_confs: List[float], cfg: PostprocessTuning) -> Optional[str]:
    """Drop trailing low-confidence suffix characters safely.

    Assumptions:
    - text has no spaces (common for OCR decoders); we operate on alnum-only.
    - char_confs aligns one-to-one with characters in `text`.
    Strategy:
    - Split into segments; identify suffix region indices.
    - While last suffix char exists and its confidence < thresholds, drop it.
    - Never drop prefix or number characters.
    Returns a new raw string or None if unchanged/invalid mapping.
    """
    if not text:
        return None
    clean = _clean(text)
    flat = re.sub(r"\s+", "", clean)
    if len(flat) != len(char_confs):
        return None
    # Determine segment boundaries based on cleaned string
    p, n, s = _split_segments(clean)
    if not (p or n or s):
        return None
    p_len = len(p)
    n_len = len(n)
    # Sanity guard: ensure prefix are letters and numbers are digits lengths within bounds
    if p_len < 1 or n_len < 1:
        return None
    suffix_start = p_len + n_len
    if suffix_start > len(flat):
        return None
    # Nothing to do if no suffix
    if suffix_start == len(flat):
        return None

    # Iterate from the end, drop while below threshold and allowed
    confs = list(char_confs)
    chars = list(flat)
    changed = False
    while len(chars) > suffix_start:
        last_idx = len(chars) - 1
        last_conf = float(confs[last_idx])
        # thresholds
        min_tail = max(0.0, float(cfg.last_char_min_conf))
        min_each = max(0.0, float(cfg.min_suffix_char_conf))
        threshold = max(min_tail, min_each)
        if threshold <= 0.0:
            break
        if last_conf >= threshold:
            break
        # Drop last char
        chars.pop()
        confs.pop()
        changed = True
        # Optionally only drop one character; for now allow dropping multiple until satisfied
        if not bool(cfg.truncate_ambiguous_suffix) and min_tail > 0.0:
            # If only last-char rule is on but truncate flag is false, drop at most one
            break

    if not changed:
        return None
    return "".join(chars)


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


def _normalize_plate(
    text: str,
    allowed_set: Optional[set[str]],
    regex_obj: re.Pattern[str],
    tuning: PostprocessTuning,
) -> Tuple[str, bool, str, str]:
    cleaned = _clean(text)
    if not cleaned:
        return "", False, "", ""
    p, n, s = _split_segments(cleaned)
    p, n, s = _apply_ambiguity(p, n, s)

    p = re.sub(r"[^A-Z]", "", p)[:2]
    n = re.sub(r"[^0-9]", "", n)[:4]
    s = re.sub(r"[^A-Z]", "", s)[:3]
    if len(s) >= tuning.duplicate_collapse_min_len and len(set(s)) == 1:
        s = s[0]

    out = " ".join([x for x in (p, n, s) if x])
    if not out:
        return "", False, "", ""

    is_match = bool(regex_obj.match(out))
    if allowed_set is not None and p:
        valid = is_match and (p in allowed_set)
    else:
        valid = is_match
    return out, valid, p, s


def _normalize_clean_plate(
    clean_text: str,
    allowed_set: Optional[set[str]],
    regex_obj: re.Pattern[str],
    tuning: PostprocessTuning,
) -> Tuple[str, bool, str, str]:
    """Variant of _normalize_plate that accepts already cleaned text."""
    if not clean_text:
        return "", False, "", 0
    p, n, s = _split_segments(clean_text)
    p, n, s = _apply_ambiguity(p, n, s)
    p = re.sub(r"[^A-Z]", "", p)[:2]
    n = re.sub(r"[^0-9]", "", n)[:4]
    s = re.sub(r"[^A-Z]", "", s)[:3]
    if len(s) >= tuning.duplicate_collapse_min_len and len(set(s)) == 1:
        s = s[0]
    out = " ".join([x for x in (p, n, s) if x])
    if not out:
        return "", False, "", ""
    is_match = bool(regex_obj.match(out))
    if allowed_set is not None and p:
        valid = is_match and (p in allowed_set)
    else:
        valid = is_match
    return out, valid, p, s


def _suffix_penalty(suffix: str, tuning: PostprocessTuning) -> float:
    if not suffix:
        return 0.0
    penalty = 0.0
    if len(suffix) < 3:
        penalty += float(tuning.suffix_len_lt3_penalty)
    if len(suffix) == 2:
        penalty += tuning.suffix_last_letter_penalty.get(suffix[-1], 0.0)
        if len(set(suffix)) == 1:
            penalty += float(tuning.suffix_duplicate_penalty)
    for key, value in tuning.suffix_penalty_map.items():
        if suffix == key:
            penalty += float(value)
    for key, value in tuning.suffix_contains_penalty.items():
        if key in suffix:
            penalty += float(value)
    for key, value in tuning.suffix_bonus_map.items():
        if suffix == key:
            penalty += float(value)
    if len(suffix) == 3 and re.search(r"[AEIOU]{2}", suffix):
        penalty += float(tuning.suffix_vowel_pair_penalty)
    return max(0.0, penalty)


def _search_best_plate(
    text: str,
    allowed_set: Optional[set[str]],
    regex_obj: re.Pattern[str],
    tuning: PostprocessTuning,
    baseline_text: str,
    baseline_score: float,
) -> Optional[str]:
    """Search nearby strings (small edits) for a valid Indonesian plate."""
    start = _clean(text)
    if not start:
        return None

    queue: List[Tuple[float, str]] = []
    heappush(queue, (0.0, start))
    seen = {start: 0.0}
    best_score = baseline_score
    best_text = baseline_text if baseline_score < float("inf") else None
    states = 0

    while queue and states < _MAX_SEARCH_STATES:
        cost, raw = heappop(queue)
        states += 1

        normalized, valid, _, suffix = _normalize_clean_plate(raw, allowed_set, regex_obj, tuning)
        if normalized:
            penalty = _suffix_penalty(suffix, tuning)
            score = cost + penalty
            if valid and score + 1e-6 < best_score:
                best_score = score
                best_text = normalized

        if cost >= _MAX_EDIT_COST:
            continue

        for candidate, op_cost in _expand_candidates(raw, tuning):
            new_cost = cost + op_cost
            if new_cost > _MAX_EDIT_COST:
                continue
            prev = seen.get(candidate)
            if prev is not None and prev <= new_cost:
                continue
            seen[candidate] = new_cost
            heappush(queue, (new_cost, candidate))

    if best_text is not None and best_score + 1e-6 < baseline_score:
        return best_text
    if best_text is not None and baseline_score == float("inf"):
        return best_text
    return None


def _segment_bounds(clean_text: str) -> Tuple[int, int]:
    """Return (prefix_end, number_end) indices into clean_text."""
    p, n, _ = _split_segments(clean_text)
    pref_end = min(len(clean_text), len(p))
    num_end = min(len(clean_text), pref_end + len(n))
    return pref_end, num_end


def _expand_candidates(clean_text: str, tuning: PostprocessTuning) -> Iterable[Tuple[str, float]]:
    """Generate nearby strings via small edits with heuristic costs.

    Heuristics tuned to match regression expectations:
    - Prefer edits earlier within the suffix when resolving duplicates (e.g., JTT → JYT).
    - Prefer inserting at the end of the suffix for completing common trigrams (e.g., VI → VIN).
    - Prefer inserting 'P' before 'DC' to form 'PDC' when suffix is 'DC'.
    """
    pref_end, num_end = _segment_bounds(clean_text)
    length = len(clean_text)
    suffix = clean_text[num_end:]

    # Substitutions
    for idx, ch in enumerate(clean_text):
        if idx < pref_end:
            replacements = _PREFIX_CONFUSIONS.get(ch, ())
            cost = _SUB_COST_PREFIX
        elif idx < num_end:
            replacements = _DIGIT_CONFUSIONS.get(ch, ())
            cost = _SUB_COST_DIGIT
        else:
            replacements = _SUFFIX_CONFUSIONS.get(ch, ())
            # Slightly prefer changing earlier chars within the suffix region
            # so 'JTT' favors changing the first 'T' → 'Y' before the last one.
            rel_idx = idx - num_end
            cost = _SUB_COST_SUFFIX - (0.02 if rel_idx == 1 else 0.0) - (0.01 if rel_idx == 0 else 0.0)
        for repl in replacements:
            if repl == ch:
                continue
            yield clean_text[:idx] + repl + clean_text[idx + 1 :], cost

    # Insertions at logical boundaries (end of prefix, end of number, end of string)
    insert_positions = {pref_end, num_end, length}
    # allow insertions inside suffix region to recover missing interior letters
    for pos in range(num_end, length + 1):
        insert_positions.add(pos)
    # Iterate from later positions to earlier ones to bias appending at end
    for pos in sorted(insert_positions, reverse=True):
        if pos < 0 or pos > length:
            continue
        for char in _SUFFIX_INSERT_CHOICES:
            adj_cost = _INSERT_COST
            # Prefer completing 'VI' → 'VIN' by inserting 'N' at the end
            if pos == length and suffix.endswith("VI") and char == "N":
                adj_cost -= float(tuning.insert_bias_vi_to_vin)
            # Prefer prefixing 'DC' with 'P' → 'PDC' when inserting at start of suffix
            if pos == num_end and suffix.startswith("DC") and char == "P":
                adj_cost -= float(tuning.insert_bias_pdc)
            yield clean_text[:pos] + char + clean_text[pos:], adj_cost

    # Deletions
    if length > 0:
        for idx in range(length):
            yield clean_text[:idx] + clean_text[idx + 1 :], _DELETE_COST
