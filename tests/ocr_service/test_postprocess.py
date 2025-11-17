from ocr_service.postprocess import (
    DEFAULT_REGEX,
    MajorityVote,
    PostprocessTuning,
    load_postprocess_config,
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


def test_postprocess_indonesia_tail_u_not_flipped_to_o():
    # Regression: a valid suffix ending in U should not be "corrected" to O.
    text, valid = postprocess_indonesia("B9048ZU", allowed_prefix=["B"])
    assert valid is True
    assert text == "B 9048 ZU"


def test_majority_vote_prefers_frequent_then_conf():
    mv = MajorityVote(window=8)
    mv.add("B 9418 QW", 0.90)
    mv.add("B 9418 QW", 0.80)
    mv.add("B 9418 QW", 0.60)
    mv.add("B 941B QW", 0.85)  # one-off typo
    best = mv.best()
    assert best is not None
    assert best[0] == "B 9418 QW"


def test_load_postprocess_config(tmp_path):
    cfg_file = tmp_path / "postproc.yaml"
    cfg_file.write_text(
        """
suffix_len_lt3_penalty: 0.5
insert_bias_pdc: 0.1
suffix_last_letter_penalty:
  I: 0.2
""",
        encoding="utf-8",
    )
    cfg = load_postprocess_config(str(cfg_file))
    assert cfg.suffix_len_lt3_penalty == 0.5
    assert cfg.insert_bias_pdc == 0.1
    assert cfg.suffix_last_letter_penalty["I"] == 0.2


def test_postprocess_indonesia_respects_custom_tuning():
    base = PostprocessTuning()
    custom = PostprocessTuning(
        suffix_penalty_map={**base.suffix_penalty_map, "DG": 0.0},
        suffix_bonus_map={"DG": -0.5},
        insert_bias_pdc=0.0,
    )
    # Without custom tuning, R9105DC → PDC
    text_default, _ = postprocess_indonesia("R9105DC", allowed_prefix=["B"])
    assert text_default == "B 9105 PDC"
    # With the custom config, prefer DG to demonstrate configurability
    text_custom, _ = postprocess_indonesia("R9105DC", allowed_prefix=["B"], tuning=custom)
    assert text_custom != "B 9105 PDC"


def test_postprocess_indonesia_yaml_prefers_zu_over_zo():
    # Ensure the shipped YAML config biases ZO -> ZU in the suffix.
    cfg = load_postprocess_config("configs/ocr/postproc_indonesia.yaml")
    text, valid = postprocess_indonesia("B9048ZO", allowed_prefix=["B"], tuning=cfg)
    assert valid is True
    assert text == "B 9048 ZU"
