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


def test_reading_region_prefers_detected_document_outside_roi(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((480, 640, 3), 35, dtype=np.uint8)

    paper = np.array([[35, 70], [325, 45], [355, 330], [55, 360]], dtype=np.int32)
    cv2.fillConvexPoly(frame, paper, (240, 240, 235))
    cv2.putText(
        frame,
        "pudo beber",
        (80, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "agua",
        (85, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    region = service.reading_region(frame)

    assert region is not None
    assert region.source == "documento"
    assert region.box[0] < int(frame.shape[1] * service.config.ocr_roi_x1)


def test_fast_mode_does_not_try_full_frame(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    sources = [source for source, _region in service._ocr_regions(frame)]

    assert "frame" not in sources


def test_document_text_trim_keeps_lower_story_text(monkeypatch):
    service = _ocr_service(monkeypatch)
    page = np.full((1000, 700, 3), 255, dtype=np.uint8)
    for index, y in enumerate(range(120, 820, 80)):
        cv2.putText(
            page,
            f"linea de cuento {index}",
            (70, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
    cv2.rectangle(page, (80, 850), (620, 980), (0, 0, 0), -1)

    trimmed = service._trim_document_text_area(page)

    assert trimmed.shape[0] >= 760
    assert trimmed.shape[0] < page.shape[0]


def test_text_region_prefers_text_lines_over_solid_picture(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((700, 900, 3), 245, dtype=np.uint8)
    cv2.rectangle(frame, (80, 440), (810, 650), (0, 0, 0), -1)
    for index, y in enumerate((120, 190, 260, 330)):
        cv2.putText(
            frame,
            f"cuento linea {index}",
            (110, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    region = service._text_region(frame)

    assert region is not None
    assert region.box[3] < 440
