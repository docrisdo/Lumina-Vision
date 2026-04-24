from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
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

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x1 = int(width * 0.15)
        x2 = int(width * 0.85)
        y1 = int(height * 0.15)
        y2 = int(height * 0.85)
        return frame[y1:y2, x1:x2]

    def _preprocess_variants(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)

        sharpened = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 2.0), -0.6, 0)
        otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        return [gray, sharpened, otsu, adaptive]

    def _score_text(self, text: str) -> tuple[int, int]:
        letters = sum(char.isalpha() for char in text)
        return letters, len(text)

    def extract_text(self, frame: np.ndarray) -> OCRResult | None:
        candidates: list[str] = []

        for region in (frame, self._center_crop(frame)):
            for variant in self._preprocess_variants(region):
                for psm in (6, 7, 11):
                    text = pytesseract.image_to_string(
                        variant,
                        lang=self.config.ocr_language,
                        config=f"--oem 1 --psm {psm}",
                        timeout=3,
                    )
                    cleaned = clean_ocr_text(text)
                    if len(cleaned) >= self.config.ocr_min_text_length:
                        candidates.append(cleaned)

        if not candidates:
            return None

        candidates.sort(key=self._score_text, reverse=True)
        return OCRResult(text=candidates[0], confidence_hint=0.5)
