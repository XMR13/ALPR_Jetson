def test_trt_infer_module_importable():
    import importlib

    mod = importlib.import_module("ocr_service.trt_infer")
    assert hasattr(mod, "__file__")

