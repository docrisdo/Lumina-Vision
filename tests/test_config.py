from lumina_vision.config import AppConfig


def test_config_defaults(monkeypatch):
    monkeypatch.delenv("LUMINA_CAMERA_WIDTH", raising=False)
    monkeypatch.delenv("LUMINA_ENABLE_OCR", raising=False)

    config = AppConfig.load()

    assert config.camera_width == 1280
    assert config.enable_ocr is True
    assert config.detection_run_every_n_frames >= 1
