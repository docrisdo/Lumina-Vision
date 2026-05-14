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


@dataclass(slots=True)
class OCRCandidate:
    text: str
    confidence_hint: float
    priority: int
    source: str


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

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1)
        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]
        return rect

    def _four_point_transform(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        rect = self._order_points(points.reshape(4, 2).astype("float32"))
        top_left, top_right, bottom_right, bottom_left = rect
        width_a = np.linalg.norm(bottom_right - bottom_left)
        width_b = np.linalg.norm(top_right - top_left)
        height_a = np.linalg.norm(top_right - bottom_right)
        height_b = np.linalg.norm(top_left - bottom_left)
        max_width = max(1, int(max(width_a, width_b)))
        max_height = max(1, int(max(height_a, height_b)))
        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, destination)
        return cv2.warpPerspective(frame, matrix, (max_width, max_height))

    def _document_crop(self, frame: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        area = float(height * width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < area * 0.18:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4:
                warped = self._four_point_transform(frame, approx)
                warped_height, warped_width = warped.shape[:2]
                if warped_width >= 220 and warped_height >= 220:
                    return warped
        return None

    def _ocr_regions(self, frame: np.ndarray) -> list[tuple[str, np.ndarray]]:
        regions: list[tuple[str, np.ndarray]] = []
        document = self._document_crop(frame)
        if document is not None:
            regions.append(("documento", document))
        center = self._center_crop(frame)
        regions.append(("centro", center))
        if not self.config.ocr_prefer_center_crop:
            regions.insert(0, ("frame", frame))
        else:
            regions.append(("frame", frame))
        return regions

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

    def _preprocess_page_variants(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        sharpened = cv2.addWeighted(contrast, 1.5, cv2.GaussianBlur(contrast, (0, 0), 1.0), -0.5, 0)
        otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            15,
        )
        return [otsu, adaptive, sharpened, contrast]

    def best_debug_variant(self, frame: np.ndarray) -> np.ndarray:
        variants = self._preprocess_variants(self._center_crop(self._limit_width(frame)))
        return variants[0]

    def _score_text(self, text: str) -> tuple[int, int, int]:
        letters = sum(char.isalpha() for char in text)
        words = len(re.findall(r"[^\W\d_]{2,}", text, flags=re.UNICODE))
        noise = sum(char in "|[]{}_=~^" for char in text)
        return words, letters, len(text) - noise * 4

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"[^\W\d_]{2,}", text, flags=re.UNICODE))

    def _looks_like_text(self, text: str) -> bool:
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        if not tokens:
            return False

        letters = sum(char.isalpha() for char in text)
        if letters < max(3, self.config.ocr_min_text_length - 1):
            return False

        single_letter_tokens = sum(1 for token in tokens if len(token) == 1 and token.isalpha())
        alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
        useful_tokens = [token for token in alpha_tokens if len(token) >= 2]

        if len(tokens) == 1:
            return len(tokens[0]) >= max(3, self.config.ocr_min_text_length - 1)

        if not useful_tokens:
            return False

        # OCR noise often appears as many isolated letters: "i A a 4".
        return single_letter_tokens <= max(2, len(useful_tokens))

    def _clean_large_text_candidate(self, text: str) -> str:
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        if not tokens:
            return ""

        strong_tokens: list[str] = []
        for token in tokens:
            letters = [char for char in token if char.isalpha()]
            if len(letters) < max(3, self.config.ocr_min_text_length - 1):
                continue
            uppercase_letters = sum(char.isupper() for char in letters)
            uppercase_ratio = uppercase_letters / float(len(letters))
            if token.isupper() or uppercase_ratio >= 0.75:
                strong_tokens.append(token)

        if strong_tokens:
            return clean_ocr_text(" ".join(strong_tokens))

        useful_tokens = [token for token in tokens if len(token) >= self.config.ocr_min_text_length]
        if len(useful_tokens) == 1:
            return clean_ocr_text(useful_tokens[0])

        return clean_ocr_text(text)

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

    def _text_lines_from_data(self, image: np.ndarray, psm: int) -> tuple[str, float]:
        data = pytesseract.image_to_data(
            image,
            lang=self.config.ocr_language,
            config=self._ocr_config(psm),
            output_type=Output.DICT,
            timeout=6,
        )
        lines: dict[tuple[int, int, int], list[str]] = {}
        confidences: list[float] = []
        for index, raw_text in enumerate(data.get("text", [])):
            text = clean_ocr_text(str(raw_text))
            if not text:
                continue
            try:
                confidence = float(data.get("conf", [])[index])
            except (ValueError, IndexError):
                confidence = -1.0
            if confidence < 30 and len(text) < self.config.ocr_min_text_length:
                continue
            key = (
                int(data.get("block_num", [0])[index]),
                int(data.get("par_num", [0])[index]),
                int(data.get("line_num", [0])[index]),
            )
            lines.setdefault(key, []).append(text)
            if confidence >= 0:
                confidences.append(confidence)

        ordered_lines = [clean_ocr_text(" ".join(words)) for _key, words in sorted(lines.items())]
        ordered_lines = [line for line in ordered_lines if self._looks_like_text(line)]
        text = clean_ocr_text(". ".join(ordered_lines))
        return text, float(np.mean(confidences)) if confidences else 0.0

    def _extract_page_text(self, region: np.ndarray, source: str) -> list[OCRCandidate]:
        candidates: list[OCRCandidate] = []
        for rotated in (self._rotate(region, 0), self._rotate(region, -1.5), self._rotate(region, 1.5)):
            variants = self._preprocess_page_variants(rotated)
            for variant in variants:
                for psm in (6, 4, 3):
                    text, confidence = self._text_lines_from_data(variant, psm)
                    if self._word_count(text) >= 5 and self._looks_like_text(text):
                        candidates.append(OCRCandidate(text, confidence, 3, source))
            if candidates:
                break
        return candidates

    def _extract_large_text(self, variants: list[np.ndarray], source: str) -> OCRCandidate | None:
        best: OCRCandidate | None = None
        for variant in variants:
            for psm in (8, 7, 13):
                text = pytesseract.image_to_string(
                    variant,
                    lang=self.config.ocr_language,
                    config=self._ocr_config(psm, large_text=True),
                    timeout=2,
                )
                cleaned = self._clean_large_text_candidate(clean_ocr_text(text))
                letters = sum(char.isalpha() for char in cleaned)
                if letters < max(2, self.config.ocr_min_text_length - 1):
                    continue
                if not self._looks_like_text(cleaned):
                    continue
                score = float(letters * 10 + len(cleaned))
                if best is None or score > best.confidence_hint:
                    best = OCRCandidate(cleaned, score, 2, source)
        return best

    def _extract_regular_text(self, variants: list[np.ndarray], source: str) -> list[OCRCandidate]:
        candidates: list[OCRCandidate] = []
        for variant in variants:
            for psm in (6, 11, 4):
                cleaned, confidence = self._text_from_data(variant, psm)
                if len(cleaned) >= self.config.ocr_min_text_length and self._looks_like_text(cleaned):
                    candidates.append(OCRCandidate(cleaned, confidence, 1, source))
        return candidates

    def extract_candidates(self, frame: np.ndarray) -> tuple[list[OCRCandidate], float]:
        candidates: list[OCRCandidate] = []
        frame = self._limit_width(frame)
        frame_sharpness = self.sharpness(frame)
        if frame_sharpness < self.config.ocr_min_sharpness:
            return candidates, frame_sharpness

        for source, region in self._ocr_regions(frame):
            if self.config.ocr_page_mode:
                candidates.extend(self._extract_page_text(region, source))
                if candidates:
                    break

            for rotation_index, rotated in enumerate((self._rotate(region, 0), self._rotate(region, -2.0), self._rotate(region, 2.0))):
                variants = self._preprocess_variants(rotated)
                large_text = self._extract_large_text(variants, source)
                if large_text is not None:
                    candidates.append(large_text)
                candidates.extend(self._extract_regular_text(variants[:4], source))

                if candidates and rotation_index == 0:
                    break
            if candidates:
                break

        candidates.sort(
            key=lambda item: (item.priority, *self._score_text(item.text), item.confidence_hint),
            reverse=True,
        )
        return candidates, frame_sharpness

    def debug_variants(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        frame = self._limit_width(frame)
        debug: dict[str, np.ndarray] = {"ocr_original": frame}
        for source, region in self._ocr_regions(frame):
            debug[f"ocr_region_{source}"] = region
            for index, variant in enumerate(self._preprocess_variants(region)):
                debug[f"ocr_{source}_variant_{index}"] = variant
            for index, variant in enumerate(self._preprocess_page_variants(region)):
                debug[f"ocr_{source}_page_variant_{index}"] = variant
        debug["ocr_best_for_tesseract"] = self.best_debug_variant(frame)
        return debug

    def extract_text(self, frame: np.ndarray) -> OCRResult | None:
        candidates, frame_sharpness = self.extract_candidates(frame)
        if not candidates:
            return None

        candidate = candidates[0]
        return OCRResult(text=candidate.text, confidence_hint=candidate.confidence_hint, sharpness=frame_sharpness)
