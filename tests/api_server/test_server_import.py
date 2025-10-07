def test_api_server_importable():
    import importlib

    mod = importlib.import_module("api_server.server")
    assert hasattr(mod, "__file__")

