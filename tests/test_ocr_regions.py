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
    assert region.box[3] < 340


def test_fast_mode_does_not_try_full_frame(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    sources = [source for source, _region in service._ocr_regions(frame)]

    assert "frame" not in sources
    assert "pagina" in sources
    assert "centro" not in sources


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


def test_document_region_box_targets_text_area(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((900, 700, 3), 35, dtype=np.uint8)
    paper = np.array([[70, 40], [630, 45], [620, 850], [60, 845]], dtype=np.int32)
    cv2.fillConvexPoly(frame, paper, (245, 245, 245))
    for index, y in enumerate(range(150, 600, 80)):
        cv2.putText(
            frame,
            f"cuento linea {index}",
            (120, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(frame, (170, 690), (540, 810), (0, 0, 0), -1)

    region = service.reading_region(frame)

    assert region is not None
    assert region.source == "documento"
    assert region.box[1] > 60
    assert region.box[3] < 720


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


def test_text_region_groups_multiple_separate_lines(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((900, 700, 3), 235, dtype=np.uint8)
    y_positions = (130, 220, 310, 400, 490, 580, 670)
    for index, y in enumerate(y_positions):
        cv2.putText(
            frame,
            f"linea cuento {index}",
            (90, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    region = service._text_region(frame)

    assert region is not None
    assert region.source == "texto"
    assert region.box[1] < min(y_positions)
    assert region.box[3] > max(y_positions)


def test_text_region_detects_narrow_page_inside_frame(monkeypatch):
    service = _ocr_service(monkeypatch)
    frame = np.full((864, 1536, 3), 70, dtype=np.uint8)
    cv2.rectangle(frame, (480, 80), (1060, 820), (235, 235, 235), -1)
    for index, y in enumerate(range(170, 760, 70)):
        cv2.putText(
            frame,
            f"cuento linea {index}",
            (520, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    region = service._text_region(frame)

    assert region is not None
    assert region.source == "texto"
    assert region.box[1] < 170
    assert region.box[3] > 730


def test_long_text_repair_uses_story_vocabulary(monkeypatch):
    service = _ocr_service(monkeypatch)

    repaired = service._repair_long_text(
        "cuervo la jarra seyo sediento encon ra con un poco de ondo "
        "nsando rapidamente menzo echar piedre",
    )

    assert "cuervo" in repaired
    assert "jarra" in repaired
    assert "sediento" in repaired
    assert "fondo" in repaired
    assert "pensando" in repaired
    assert "echar" in repaired


def test_long_text_repair_restores_common_y(monkeypatch):
    service = _ocr_service(monkeypatch)

    repaired = service._repair_long_text(
        "El cuervo la jarra. Este cuento ensena sobre la inteligencia la perseverancia",
    )

    assert "El cuervo y la jarra" in repaired
    assert "inteligencia y la perseverancia" in repaired
