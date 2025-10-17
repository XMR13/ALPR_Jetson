import numpy as np

from ocr_service.preprocess import PreprocConfig
from ocr_service.trt_infer import OCRService


class _DummyRunner:
    def infer(self, x):
        # Return logits with layout [N, C, T]; expects transpose in service.
        n = x.shape[0]
        c = 37  # <blank> + 36 charset symbols
        t = 4
        logits = np.zeros((n, c, t), dtype=np.float32)
        # Time step 0 -> 'A' (index 11)
        logits[:, 11, 0] = 5.0
        # Time step 1 -> '1' (digit index 2)
        logits[:, 2, 1] = 5.0
        # Time steps 2,3 -> blank (index 0)
        logits[:, 0, 2:] = 5.0
        return logits


def test_ocr_service_transposes_nct_logits():
    svc = OCRService(engine_path=None, logits_layout="NCT", preproc=PreprocConfig(channels=3))
    svc._runner = _DummyRunner()  # inject dummy runner
    img = np.zeros((32, 160, 3), dtype=np.uint8)
    preds = svc.infer_batch([img])
    assert preds == ["A1"]

