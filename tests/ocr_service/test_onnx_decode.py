import numpy as np

from ocr_service.onnx_infer import _decode_logits


def test_decode_logits_3d_removes_pad():
    # Vocab: A,B,_ (pad)
    alphabet = "AB_"
    pad_char = "_"
    # Two samples, S=3, V=3
    logits = np.array([
        # sample 0 -> A B _ => "AB"
        [[10, 1, 0], [1, 9, 0], [0, 0, 5]],
        # sample 1 -> _ A A => "AA"
        [[0, 0, 7], [5, 0, 0], [8, 1, 0]],
    ], dtype=np.float32)
    out = _decode_logits(logits, max_slots=3, alphabet=alphabet, pad_char=pad_char)
    assert out == ["AB", "AA"]


def test_decode_logits_2d_flattened():
    alphabet = "01_"
    pad_char = "_"
    # N=1, S=4 => flattened shape (4, V)
    logits = np.array(
        [
            [9, 1, 0],  # 0
            [1, 9, 0],  # 1
            [1, 9, 0],  # 1
            [0, 0, 5],  # pad
        ],
        dtype=np.float32,
    )
    out = _decode_logits(logits, max_slots=4, alphabet=alphabet, pad_char=pad_char)
    assert out == ["011"]

