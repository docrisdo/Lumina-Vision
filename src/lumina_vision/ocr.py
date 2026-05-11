from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from lumina_vision.config import AppConfig
from lumina_vision.utils import clean_ocr_text


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence_hint: float
    sharpness: float


class OCRService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _limit_width(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.config.ocr_max_width:
            return frame
        scale = self.config.ocr_max_width / float(width)
        return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x1 = int(width * 0.04)
        x2 = int(width * 0.96)
        y1 = int(height * 0.08)
        y2 = int(height * 0.92)
        return frame[y1:y2, x1:x2]

    def sharpness(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _rotate(self, frame: np.ndarray, angle: float) -> np.ndarray:
        if angle == 0:
            return frame
        height, width = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _preprocess_variants(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)

        sharpened = cv2.addWeighted(gray, 1.8, cv2.GaussianBlur(gray, (0, 0), 1.2), -0.8, 0)
        denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
        otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        adaptive_inv = cv2.bitwise_not(adaptive)

        return [otsu, adaptive, adaptive_inv, sharpened, denoised, gray]

    def best_debug_variant(self, frame: np.ndarray) -> np.ndarray:
        variants = self._preprocess_variants(self._center_crop(self._limit_width(frame)))
        return variants[0]

    def _score_text(self, text: str) -> tuple[int, int, int]:
        letters = sum(char.isalpha() for char in text)
        words = len(re.findall(r"[^\W\d_]{2,}", text, flags=re.UNICODE))
        noise = sum(char in "|[]{}_=~^" for char in text)
        return words, letters, len(text) - noise * 4

    def _ocr_config(self, psm: int, *, large_text: bool = False) -> str:
        whitelist = ""
        if large_text:
            whitelist = (
                " -c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            )
        return (
            f"--oem 1 --psm {psm} "
            "-c preserve_interword_spaces=1 "
            "-c user_defined_dpi=300"
            f"{whitelist}"
        )

    def _text_from_data(self, image: np.ndarray, psm: int) -> tuple[str, float]:
        data = pytesseract.image_to_data(
            image,
            lang=self.config.ocr_language,
            config=self._ocr_config(psm),
            output_type=Output.DICT,
            timeout=4,
        )
        words: list[str] = []
        confidences: list[float] = []
        for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", []), strict=False):
            text = clean_ocr_text(str(raw_text))
            if not text:
                continue
            try:
                confidence = float(raw_conf)
            except ValueError:
                confidence = -1.0
            if confidence >= 25 or len(text) >= self.config.ocr_min_text_length:
                words.append(text)
                if confidence >= 0:
                    confidences.append(confidence)
        if not words:
            return "", 0.0
        return clean_ocr_text(" ".join(words)), float(np.mean(confidences)) if confidences else 0.0

    def _extract_large_text(self, variants: list[np.ndarray]) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        for variant in variants:
            for psm in (8, 7, 13):
                text = pytesseract.image_to_string(
                    variant,
                    lang=self.config.ocr_language,
                    config=self._ocr_config(psm, large_text=True),
                    timeout=2,
                )
                cleaned = clean_ocr_text(text)
                letters = sum(char.isalpha() for char in cleaned)
                if letters < max(2, self.config.ocr_min_text_length - 1):
                    continue
                score = float(letters * 10 + len(cleaned))
                if best is None or score > best[1]:
                    best = (cleaned, score)
        return best

    def _extract_regular_text(self, variants: list[np.ndarray]) -> list[tuple[str, float]]:
        candidates: list[tuple[str, float]] = []
        for variant in variants:
            for psm in (6, 11, 4):
                cleaned, confidence = self._text_from_data(variant, psm)
                if len(cleaned) >= self.config.ocr_min_text_length:
                    candidates.append((cleaned, confidence))
        return candidates

    def extract_text(self, frame: np.ndarray) -> OCRResult | None:
        candidates: list[tuple[str, float]] = []
        frame = self._limit_width(frame)
        frame_sharpness = self.sharpness(frame)
        if frame_sharpness < self.config.ocr_min_sharpness:
            return None

        regions = [self._center_crop(frame), frame] if self.config.ocr_prefer_center_crop else [frame]
        for region in regions:
            for rotation_index, rotated in enumerate((self._rotate(region, 0), self._rotate(region, -2.0), self._rotate(region, 2.0))):
                variants = self._preprocess_variants(rotated)
                large_text = self._extract_large_text(variants)
                if large_text is not None:
                    candidates.append(large_text)
                candidates.extend(self._extract_regular_text(variants[:4]))

                if candidates and rotation_index == 0:
                    break
            if candidates:
                break

        if not candidates:
            return None

        candidates.sort(key=lambda item: (*self._score_text(item[0]), item[1]), reverse=True)
        text, confidence = candidates[0]
        return OCRResult(text=text, confidence_hint=confidence, sharpness=frame_sharpness)
