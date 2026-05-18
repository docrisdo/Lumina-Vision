import cv2
import numpy as np

from lumina_vision.config import AppConfig
from lumina_vision.ocr import OCRService


def _ocr_service(monkeypatch) -> OCRService:
    monkeypatch.delenv("LUMINA_OCR_FAST_MODE", raising=False)
    config = AppConfig.load()
    config.ocr_fast_mode = True
    config.ocr_min_sharpness = 0.0
    return OCRService(config)


def test_text_region_uses_roi_and_ignores_background(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)

    cv2.rectangle(frame, (10, 20), (170, 95), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "pudo beber",
        (180, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "agua",
        (180, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    box = service.reading_box(frame)

    assert box is not None
    assert box[0] >= int(frame.shape[1] * service.config.ocr_roi_x1)
    assert box[1] >= int(frame.shape[0] * service.config.ocr_roi_y1)


def test_fast_mode_does_not_try_full_frame(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    sources = [source for source, _region in service._ocr_regions(frame)]

    assert "frame" not in sources
