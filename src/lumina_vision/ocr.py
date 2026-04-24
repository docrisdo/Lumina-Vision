from __future__ import annotations

from dataclasses import dataclass

import cv2
import pytesseract

from lumina_vision.config import AppConfig
from lumina_vision.utils import clean_ocr_text


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence_hint: float


class OCRService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def extract_text(self, frame) -> OCRResult | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        text = pytesseract.image_to_string(
            processed,
            lang=self.config.ocr_language,
            config="--oem 1 --psm 6",
            timeout=3,
        )
        cleaned = clean_ocr_text(text)
        if len(cleaned) < self.config.ocr_min_text_length:
            return None

        return OCRResult(text=cleaned, confidence_hint=0.5)
