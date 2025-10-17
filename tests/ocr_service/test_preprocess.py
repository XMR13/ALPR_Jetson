import numpy as np

from ocr_service.preprocess import PreprocConfig, prepare_ocr_batch


def test_prepare_ocr_batch_three_channels_shape():
    img = np.random.randint(0, 255, (48, 160, 3), dtype=np.uint8)
    cfg = PreprocConfig(input_width=160, input_height=32, channels=3)
    batch = prepare_ocr_batch([img], cfg)
    assert batch.shape == (1, 3, 32, 160)
    assert batch.dtype == np.float32

