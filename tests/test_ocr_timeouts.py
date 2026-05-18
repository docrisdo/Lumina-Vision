import numpy as np

from lumina_vision.config import AppConfig
from lumina_vision.ocr import OCRService


def _ocr_service(monkeypatch) -> OCRService:
    monkeypatch.delenv("LUMINA_OCR_FAST_MODE", raising=False)
    config = AppConfig.load()
    config.ocr_fast_mode = True
    return OCRService(config)


def test_large_text_timeout_is_skipped(monkeypatch):
    service = _ocr_service(monkeypatch)

    def timeout(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr("lumina_vision.ocr.pytesseract.image_to_string", timeout)

    image = np.zeros((80, 240), dtype=np.uint8)

    assert service._extract_large_text([image], "texto") is None


def test_data_timeout_returns_empty_text(monkeypatch):
    service = _ocr_service(monkeypatch)

    def timeout(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr("lumina_vision.ocr.pytesseract.image_to_data", timeout)

    image = np.zeros((80, 240), dtype=np.uint8)

    assert service._text_from_data(image, 6) == ("", 0.0)
    assert service._text_lines_from_data(image, 6) == ("", 0.0)
