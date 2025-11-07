from ocr_service.postprocess import (
    DEFAULT_REGEX,
    MajorityVote,
    postprocess_indonesia,
)


def test_postprocess_indonesia_basic_valid():
    text, valid = postprocess_indonesia("B 9418 QW", allowed_prefix=["A", "B"], regex=DEFAULT_REGEX)
    assert text == "B 9418 QW"
    assert valid is True


def test_postprocess_indonesia_ambiguous_fix_letters_and_digits():
    # Digit mistaken as letter and vice-versa
    text, valid = postprocess_indonesia("B 9O16 PEN", allowed_prefix=["B"])  # O -> 0 in number segment
    assert text == "B 9016 PEN"
    assert valid is True

    text2, valid2 = postprocess_indonesia("8 9418 QW", allowed_prefix=["B"])  # 8 -> B in letter prefix
    assert text2 == "B 9418 QW"
    assert valid2 is True


def test_postprocess_indonesia_recovers_missing_suffix_letter():
    text, valid = postprocess_indonesia("B9048VI", allowed_prefix=["B"])
    assert valid is True
    assert text == "B 9048 VIN"


def test_postprocess_indonesia_recovers_missing_suffix_prefix_mix():
    text, valid = postprocess_indonesia("R9105DC", allowed_prefix=["A", "B"])
    assert valid is True
    assert text == "B 9105 PDC"


def test_postprocess_indonesia_suffix_confusions():
    text, valid = postprocess_indonesia("B9470JTT", allowed_prefix=["B"])
    assert valid is True
    assert text == "B 9470 JYT"

    text2, valid2 = postprocess_indonesia("B9922DE", allowed_prefix=["B"])
    assert valid2 is True
    assert text2 == "B 9922 OE"

    text3, valid3 = postprocess_indonesia("A9513SS", allowed_prefix=["A", "B"])
    assert valid3 is True
    assert text3 == "A 9513 S"


def test_majority_vote_prefers_frequent_then_conf():
    mv = MajorityVote(window=8)
    mv.add("B 9418 QW", 0.90)
    mv.add("B 9418 QW", 0.80)
    mv.add("B 9418 QW", 0.60)
    mv.add("B 941B QW", 0.85)  # one-off typo
    best = mv.best()
    assert best is not None
    assert best[0] == "B 9418 QW"
