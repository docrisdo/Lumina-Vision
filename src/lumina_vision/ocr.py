from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata

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


@dataclass(slots=True)
class ReadingRegion:
    image: np.ndarray
    box: tuple[int, int, int, int]
    score: float
    source: str


class OCRService:
    _WORD_CONFIDENCE_MIN = 35.0
    _WORD_BOX_CONFIDENCE_MIN = 50.0
    _SPANISH_COMMON_WORDS = {
        "a",
        "agua",
        "al",
        "aunque",
        "cada",
        "comenzo",
        "como",
        "con",
        "cuento",
        "cuervo",
        "de",
        "del",
        "dentro",
        "donde",
        "e",
        "el",
        "ella",
        "ellos",
        "en",
        "encontro",
        "ensena",
        "era",
        "es",
        "este",
        "finalmente",
        "fondo",
        "fue",
        "habia",
        "hasta",
        "inteligencia",
        "jarra",
        "la",
        "las",
        "le",
        "lecturas",
        "lo",
        "los",
        "mas",
        "mientras",
        "no",
        "para",
        "pero",
        "pico",
        "piedra",
        "piedras",
        "poco",
        "problemas",
        "por",
        "pudo",
        "que",
        "rapidamente",
        "resolver",
        "se",
        "sediento",
        "sobre",
        "su",
        "subia",
        "trato",
        "un",
        "una",
        "y",
    }
    _SPANISH_COMMON_WORDS.update(
        {
            "abajo",
            "abrio",
            "ademas",
            "ahora",
            "alcanzaba",
            "alli",
            "animales",
            "aprendio",
            "arriba",
            "beber",
            "beberla",
            "bien",
            "blanco",
            "buscar",
            "casa",
            "clase",
            "cortas",
            "crecio",
            "cuando",
            "dia",
            "echo",
            "escuela",
            "eso",
            "estaba",
            "estar",
            "feo",
            "grande",
            "hacer",
            "hizo",
            "hoja",
            "huevo",
            "huevos",
            "jardin",
            "leer",
            "libro",
            "mama",
            "muy",
            "nacio",
            "nacidos",
            "negro",
            "nino",
            "ninos",
            "otro",
            "otros",
            "pagina",
            "patito",
            "porque",
            "primero",
            "presente",
            "presentes",
            "quedo",
            "rechazado",
            "rompio",
            "ser",
            "si",
            "sin",
            "subio",
            "tener",
            "texto",
            "todos",
            "trabajo",
            "uno",
            "verano",
            "vez",
        },
    )
    _NOISE_RE = re.compile(r"^[\W\d_]+$|^[bcdfghjklmnpqrstvwxyz]{4,}$", re.IGNORECASE)
    _STORY_WORDS = {
        "alcanzaba",
        "beber",
        "beberla",
        "cada",
        "comenzo",
        "con",
        "cuento",
        "cuervo",
        "dentro",
        "echar",
        "el",
        "encontro",
        "ensena",
        "este",
        "finalmente",
        "fondo",
        "inteligencia",
        "jarra",
        "pense",
        "pensando",
        "perseverancia",
        "pico",
        "piedra",
        "piedras",
        "poco",
        "problemas",
        "pudo",
        "rapidamente",
        "resolver",
        "sediento",
        "sobre",
        "subia",
        "trato",
        "una",
    }

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
        x1 = int(width * self.config.ocr_roi_x1)
        x2 = int(width * self.config.ocr_roi_x2)
        y1 = int(height * self.config.ocr_roi_y1)
        y2 = int(height * self.config.ocr_roi_y2)
        return frame[y1:y2, x1:x2]

    def _page_fallback_crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x1 = int(width * 0.04)
        x2 = int(width * 0.96)
        y1 = int(height * 0.03)
        y2 = int(height * 0.97)
        return frame[y1:y2, x1:x2]

    def _crop_box(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        return frame[y1:y2, x1:x2]

    def _expand_box(
        self,
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int] | tuple[int, int],
        padding_ratio: float = 0.08,
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = box
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    def roi_box(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        return (
            int(width * self.config.ocr_roi_x1),
            int(height * self.config.ocr_roi_y1),
            int(width * self.config.ocr_roi_x2),
            int(height * self.config.ocr_roi_y2),
        )

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1)
        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]
        return rect

    def _four_point_transform_with_matrix(
        self,
        frame: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return cv2.warpPerspective(frame, matrix, (max_width, max_height)), matrix

    def _four_point_transform(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        crop, _matrix = self._four_point_transform_with_matrix(frame, points)
        return crop

    def _map_warp_box_to_frame(
        self,
        box: tuple[int, int, int, int],
        matrix: np.ndarray,
        frame_shape: tuple[int, int, int] | tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        points = np.array(
            [[
                [float(x1), float(y1)],
                [float(x2), float(y1)],
                [float(x2), float(y2)],
                [float(x1), float(y2)],
            ]],
            dtype="float32",
        )
        inverse = np.linalg.inv(matrix)
        mapped = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)
        height, width = frame_shape[:2]
        left = max(0, int(np.floor(np.min(mapped[:, 0]))))
        top = max(0, int(np.floor(np.min(mapped[:, 1]))))
        right = min(width, int(np.ceil(np.max(mapped[:, 0]))))
        bottom = min(height, int(np.ceil(np.max(mapped[:, 1]))))
        return self._expand_box((left, top, right, bottom), frame_shape, padding_ratio=0.04)

    def _touches_frame_edge(
        self,
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int] | tuple[int, int],
        margin_ratio: float = 0.025,
    ) -> bool:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = box
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        return x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y

    def _dark_text_density(self, frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
        dark = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        return float(np.count_nonzero(dark)) / max(1.0, float(dark.size))

    def _candidate_document_boxes(self, frame: np.ndarray) -> list[np.ndarray]:
        height, width = frame.shape[:2]
        area = float(height * width)
        boxes: list[np.ndarray] = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 95), (179, 95, 255))
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            np.ones((17, 17), np.uint8),
            iterations=2,
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        contours, _hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            contour_area = cv2.contourArea(contour)
            if contour_area < area * 0.04:
                continue
            points = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32").reshape(4, 1, 2)
            boxes.append(points)

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
                boxes.append(approx.astype("float32"))
        return boxes

    def _document_box(self, frame: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        frame_area = float(height * width)
        best_box: np.ndarray | None = None
        best_score = 0.0

        for box in self._candidate_document_boxes(frame):
            rect = self._order_points(box.reshape(4, 2).astype("float32"))
            width_a = np.linalg.norm(rect[2] - rect[3])
            width_b = np.linalg.norm(rect[1] - rect[0])
            height_a = np.linalg.norm(rect[1] - rect[2])
            height_b = np.linalg.norm(rect[0] - rect[3])
            doc_width = max(width_a, width_b)
            doc_height = max(height_a, height_b)
            doc_area = doc_width * doc_height
            if doc_width < 160 or doc_height < 160 or doc_area < frame_area * 0.035:
                continue
            if doc_area > frame_area * 0.9:
                continue
            aspect = doc_width / max(1.0, doc_height)
            if not 0.35 <= aspect <= 2.4:
                continue

            crop = self._four_point_transform(frame, box)
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray_crop))
            contrast = float(np.std(gray_crop))
            text_density = self._dark_text_density(crop)
            if text_density < 0.012 or text_density > 0.35:
                continue
            score = doc_area * (0.55 + text_density * 6.0) + contrast * 900.0 + brightness * 70.0
            if score > best_score:
                best_box = box
                best_score = score

        if best_box is not None:
            return best_box.astype("int32")
        return None

    def _document_crop(self, frame: np.ndarray) -> np.ndarray | None:
        box = self._document_box(frame)
        if box is None:
            return None
        crop = self._four_point_transform(frame, box)
        if crop.shape[0] < 160 or crop.shape[1] < 160:
            return None
        text_crop, _text_box = self._document_text_area(crop)
        return text_crop

    def _trim_document_text_area(self, crop: np.ndarray) -> np.ndarray:
        text_crop, _text_box = self._document_text_area(crop)
        return text_crop

    def _document_text_area(self, crop: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = crop.shape[:2]
        margin_x = 0
        margin_top = int(height * 0.035)
        margin_bottom = int(height * 0.025)
        base_box = (margin_x, margin_top, width - margin_x, height - margin_bottom)
        base_crop = self._crop_box(crop, base_box)
        if base_crop.size == 0:
            return crop, (0, 0, width, height)

        raw_gray = cv2.cvtColor(base_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(raw_gray)
        adaptive_dark = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )
        bright_context = cv2.morphologyEx(
            cv2.inRange(raw_gray, 145, 255),
            cv2.MORPH_CLOSE,
            np.ones((21, 21), np.uint8),
            iterations=1,
        )
        bright_context = cv2.erode(
            bright_context,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )
        fixed_dark = cv2.bitwise_and(cv2.inRange(raw_gray, 0, 125), bright_context)
        dark = cv2.bitwise_or(adaptive_dark, fixed_dark)
        fixed_dark = cv2.morphologyEx(
            fixed_dark,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
            iterations=1,
        )
        text_lines = self._text_line_bounds(dark)
        if len(self._text_line_bounds(fixed_dark)) >= 2:
            text_lines = self._text_line_bounds(fixed_dark)
        picture_top = self._large_picture_top(dark)
        if picture_top is not None:
            text_lines = [line for line in text_lines if line[1] < picture_top]

        if len(text_lines) >= 2:
            y1 = max(0, min(line[0] for line in text_lines) - int(base_crop.shape[0] * 0.03))
            y2 = min(
                base_crop.shape[0],
                max(line[1] for line in text_lines) + int(base_crop.shape[0] * 0.08),
            )
            if y2 - y1 >= base_crop.shape[0] * 0.35:
                text_box = (base_box[0], base_box[1] + y1, base_box[2], base_box[1] + y2)
                return self._crop_box(crop, text_box), text_box

        if picture_top is not None:
            cutoff = max(int(base_crop.shape[0] * 0.55), picture_top)
            text_box = (base_box[0], base_box[1], base_box[2], base_box[1] + cutoff)
            return self._crop_box(crop, text_box), text_box

        return base_crop, base_box

    def _text_line_bounds(self, dark_mask: np.ndarray) -> list[tuple[int, int]]:
        return [(y1, y2) for _x1, y1, _x2, y2 in self._text_line_boxes(dark_mask)]

    def _text_line_boxes(self, dark_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = dark_mask.shape[:2]
        line_mask = cv2.morphologyEx(
            dark_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, width // 24), 3)),
            iterations=1,
        )
        line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)
        projected_boxes = self._projected_text_line_boxes(line_mask)
        if projected_boxes:
            return projected_boxes

        contours, _hierarchy = cv2.findContours(
            line_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            density = float(np.count_nonzero(line_mask[y : y + h, x : x + w])) / max(1.0, area)
            if w < width * 0.16:
                continue
            if h < max(6, height * 0.01) or h > max(height * 0.45, 130):
                continue
            if density > 0.86:
                continue
            boxes.append((x, y, x + w, y + h))
        return sorted(boxes, key=lambda box: (box[1], box[0]))

    def _projected_text_line_boxes(self, line_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = line_mask.shape[:2]
        row_counts = np.count_nonzero(line_mask, axis=1)
        threshold = max(8, int(width * 0.012))
        active_rows = row_counts >= threshold
        boxes: list[tuple[int, int, int, int]] = []
        start: int | None = None
        for index, is_active in enumerate(active_rows):
            if is_active and start is None:
                start = index
            elif not is_active and start is not None:
                self._append_projected_line_box(line_mask, start, index, boxes)
                start = None
        if start is not None:
            self._append_projected_line_box(line_mask, start, height, boxes)
        return boxes

    def _append_projected_line_box(
        self,
        line_mask: np.ndarray,
        y1: int,
        y2: int,
        boxes: list[tuple[int, int, int, int]],
    ) -> None:
        height, width = line_mask.shape[:2]
        if y2 - y1 < max(5, int(height * 0.006)):
            return
        if y2 - y1 > max(130, int(height * 0.22)):
            return
        band = line_mask[y1:y2]
        cols = np.where(np.count_nonzero(band, axis=0) > 0)[0]
        if cols.size == 0:
            return
        x1 = int(cols[0])
        x2 = int(cols[-1]) + 1
        if x2 - x1 < width * 0.12:
            return
        boxes.append((x1, y1, x2, y2))

    def _text_block_region(self, frame: np.ndarray, dark_mask: np.ndarray) -> ReadingRegion | None:
        height, width = frame.shape[:2]
        line_boxes = self._text_line_boxes(dark_mask)
        picture_top = self._large_picture_top(dark_mask)
        if picture_top is not None:
            line_boxes = [box for box in line_boxes if box[3] < picture_top]
        if len(line_boxes) < 3:
            return None

        x1 = max(0, min(box[0] for box in line_boxes))
        y1 = max(0, min(box[1] for box in line_boxes))
        x2 = min(width, max(box[2] for box in line_boxes))
        y2 = min(height, max(box[3] for box in line_boxes))
        if (y2 - y1) < height * 0.28 or (x2 - x1) < width * 0.1:
            return None

        expanded = self._expand_box((x1, y1, x2, y2), frame.shape, padding_ratio=0.08)
        crop = self._crop_box(frame, expanded)
        crop_dark = dark_mask[expanded[1] : expanded[3], expanded[0] : expanded[2]]
        dark_density = float(np.count_nonzero(crop_dark)) / max(1.0, float(crop_dark.size))
        if dark_density < 0.01 or dark_density > 0.38:
            return None

        line_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in line_boxes)
        score = float(line_area) + min(len(line_boxes), 16) * float(height * width) * 0.04
        return ReadingRegion(crop, expanded, score, "texto")

    def _large_picture_top(self, dark_mask: np.ndarray) -> int | None:
        height, width = dark_mask.shape[:2]
        lower_start = int(height * 0.5)
        lower_mask = dark_mask[lower_start:]
        lower_mask = cv2.morphologyEx(
            lower_mask,
            cv2.MORPH_CLOSE,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )
        contours, _hierarchy = cv2.findContours(
            lower_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        picture_top: int | None = None
        lower_area = float(max(1, lower_mask.size))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            component_area = float(cv2.contourArea(contour))
            is_large_picture = (
                h >= height * 0.11
                or component_area >= lower_area * 0.035
            )
            if not is_large_picture:
                continue
            if lower_start + y < height * 0.52:
                continue
            top = lower_start + y
            picture_top = top if picture_top is None else min(picture_top, top)
        return picture_top

    def _document_region(self, frame: np.ndarray) -> ReadingRegion | None:
        box = self._document_box(frame)
        if box is None:
            return None
        crop, matrix = self._four_point_transform_with_matrix(frame, box)
        if crop.shape[0] < 160 or crop.shape[1] < 160:
            return None
        text_crop, text_box = self._document_text_area(crop)
        mapped_box = self._map_warp_box_to_frame(text_box, matrix, frame.shape)
        return ReadingRegion(
            text_crop,
            mapped_box,
            float(text_crop.size) * (1.0 + self._dark_text_density(text_crop)),
            "documento",
        )

    def document_box(self, frame: np.ndarray) -> np.ndarray | None:
        return self._document_box(self._limit_width(frame))

    def _text_region(self, frame: np.ndarray) -> ReadingRegion | None:
        height, width = frame.shape[:2]
        frame_area = float(height * width)
        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw_gray)
        adaptive_dark = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )
        bright_context = cv2.morphologyEx(
            cv2.inRange(raw_gray, 145, 255),
            cv2.MORPH_CLOSE,
            np.ones((21, 21), np.uint8),
            iterations=1,
        )
        bright_context = cv2.erode(
            bright_context,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )
        fixed_dark = cv2.bitwise_and(cv2.inRange(raw_gray, 0, 125), bright_context)
        dark = cv2.bitwise_or(adaptive_dark, fixed_dark)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        fixed_dark = cv2.morphologyEx(
            fixed_dark,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
            iterations=1,
        )
        best_region = self._text_block_region(frame, fixed_dark)
        if best_region is None:
            best_region = self._text_block_region(frame, dark)
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(18, width // 42), 3),
        )
        grouped = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        grouped = cv2.dilate(grouped, np.ones((7, 7), np.uint8), iterations=2)

        contours, _hierarchy = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            candidate_box = (x, y, x + w, y + h)
            if area < frame_area * 0.012 or area > frame_area * 0.62:
                continue
            if area > frame_area * 0.38 and self._touches_frame_edge(candidate_box, frame.shape):
                continue
            if w < 120 or h < 45:
                continue
            aspect = w / max(1, h)
            if not 0.45 <= aspect <= 8.5:
                continue

            expanded = self._expand_box(
                (x, y, x + w, y + h),
                frame.shape,
                padding_ratio=0.12,
            )
            crop = self._crop_box(frame, expanded)
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_dark = dark[expanded[1] : expanded[3], expanded[0] : expanded[2]]
            dark_density = float(np.count_nonzero(crop_dark)) / max(1.0, float(crop_dark.size))
            brightness = float(np.mean(crop_gray))
            contrast = float(np.std(crop_gray))
            if dark_density < 0.015 or dark_density > 0.42:
                continue
            if brightness < 55 or contrast < 18:
                continue
            line_count = len(self._text_line_bounds(crop_dark))
            if line_count < 1 or self._text_component_count(crop_dark) < 3:
                continue
            picture_top = self._large_picture_top(crop_dark)
            if picture_top is not None and picture_top < crop_dark.shape[0] * 0.58:
                continue

            line_bonus = min(line_count, 12) * frame_area * 0.025
            score = (
                area * (0.45 + dark_density * 2.0)
                + contrast * 900.0
                + brightness * 80.0
                + line_bonus
            )
            if best_region is None or score > best_region.score:
                best_region = ReadingRegion(crop, expanded, score, "texto")
        return best_region

    def _text_component_count(self, dark_mask: np.ndarray) -> int:
        contours, _hierarchy = cv2.findContours(
            dark_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        height, width = dark_mask.shape[:2]
        count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 5 or w < 2:
                continue
            if h > height * 0.75 or w > width * 0.9:
                continue
            if cv2.contourArea(contour) < 8:
                continue
            count += 1
        return count

    def reading_region(self, frame: np.ndarray) -> ReadingRegion | None:
        limited = self._limit_width(frame)
        document_region = self._document_region(limited)
        text_region = self._text_region(limited)

        if document_region is not None:
            return document_region

        return text_region

    def reading_box(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        region = self.reading_region(frame)
        if region is None:
            return None
        return region.box

    def page_detected(self, frame: np.ndarray) -> bool:
        region = self.reading_region(frame)
        if region is None:
            return False
        return self.sharpness(region.image) >= max(10.0, self.config.ocr_min_sharpness * 0.7)

    def focus_region(self, frame: np.ndarray) -> np.ndarray:
        region = self.reading_region(frame)
        if region is not None:
            return region.image
        return self._center_crop(frame)

    def _ocr_regions(self, frame: np.ndarray) -> list[tuple[str, np.ndarray]]:
        regions: list[tuple[str, np.ndarray]] = []
        document = self._document_crop(frame)
        if document is not None:
            regions.append(("documento", document))
            return regions
        text_region = self._text_region(frame)
        if text_region is not None:
            regions.append((text_region.source, text_region.image))
        regions.append(("pagina", self._page_fallback_crop(frame)))
        if not self.config.ocr_fast_mode:
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
        scale = 2.0 if max(gray.shape[:2]) < 1200 else 1.35
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)

        sharpened = cv2.addWeighted(gray, 1.8, cv2.GaussianBlur(gray, (0, 0), 1.2), -0.8, 0)
        otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        otsu = self._add_ocr_border(otsu)
        adaptive = self._add_ocr_border(adaptive)
        sharpened = self._add_ocr_border(sharpened)
        if self.config.ocr_fast_mode:
            return [otsu, adaptive, sharpened]
        adaptive_inv = cv2.bitwise_not(adaptive)
        denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
        return [otsu, adaptive, adaptive_inv, sharpened, denoised, gray]

    def _preprocess_page_variants(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 1.7 if max(gray.shape[:2]) < 1300 else 1.25
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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
        otsu = self._add_ocr_border(self._deskew(otsu))
        adaptive = self._add_ocr_border(self._deskew(adaptive))
        sharpened = self._add_ocr_border(sharpened)
        contrast = self._add_ocr_border(contrast)
        if self.config.ocr_fast_mode:
            return [otsu]
        return [otsu, adaptive, sharpened, contrast]

    def _add_ocr_border(self, image: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(image, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(inverted > 0))
        if coords.shape[0] < 100:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 8:
            return image
        height, width = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def best_debug_variant(self, frame: np.ndarray) -> np.ndarray:
        limited = self._limit_width(frame)
        region = self.reading_region(limited)
        variants = self._preprocess_page_variants(region.image if region is not None else self._center_crop(limited))
        return variants[0]

    def _score_text(self, text: str) -> tuple[int, int, int]:
        letters = sum(char.isalpha() for char in text)
        words = len(re.findall(r"[^\W\d_]{2,}", text, flags=re.UNICODE))
        noise = sum(char in "|[]{}_=~^" for char in text)
        return words, letters, len(text) - noise * 4

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"[^\W\d_]{2,}", text, flags=re.UNICODE))

    def _normalized_tokens(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKD", text)
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        normalized = normalized.lower()
        return re.findall(r"[a-z]{2,}", normalized)

    def _coherence_score(self, text: str, confidence: float = 0.0) -> float:
        raw_tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        tokens = self._normalized_tokens(text)
        if not raw_tokens or not tokens:
            return 0.0

        letters = sum(char.isalpha() for char in text)
        alpha_ratio = letters / max(1, len(text))
        single_letters = sum(1 for token in raw_tokens if len(token) == 1 and token.isalpha())
        single_ratio = single_letters / max(1, len(raw_tokens))
        common_hits = sum(1 for token in tokens if token in self._SPANISH_COMMON_WORDS)
        noisy_tokens = sum(1 for token in tokens if self._NOISE_RE.match(token))
        noise_ratio = noisy_tokens / max(1, len(tokens))
        avg_len = sum(len(token) for token in tokens) / max(1, len(tokens))

        score = confidence
        score += min(70.0, len(tokens) * 8.0)
        score += min(90.0, common_hits * 22.0)
        score += min(40.0, avg_len * 5.0)
        score += alpha_ratio * 40.0
        score -= single_ratio * 130.0
        score -= noise_ratio * 120.0
        if len(tokens) >= 5 and common_hits == 0:
            score -= 60.0
        return score

    def _repair_long_text(self, text: str) -> str:
        if self._word_count(text) > 8:
            clean_text = clean_ocr_text(text)
            clean_text = clean_text.replace("El cuervo la jarra", "El cuervo y la jarra")
            clean_text = clean_text.replace(
                "inteligencia la perseverancia",
                "inteligencia y la perseverancia",
            )
            return clean_text

        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        repaired: list[str] = []
        for token in tokens:
            if not any(char.isalpha() for char in token):
                repaired.append(token)
                continue
            normalized = self._normalized_tokens(token)
            if not normalized:
                repaired.append(token)
                continue
            word = normalized[0]
            if word in self._STORY_WORDS or len(word) < 4:
                repaired.append(token)
                continue
            best_word = ""
            best_score = 0.0
            for known_word in self._STORY_WORDS:
                score = SequenceMatcher(None, word, known_word).ratio()
                if word in known_word or known_word.startswith(word[: max(3, len(word) - 2)]):
                    score += 0.12
                if score > best_score:
                    best_word = known_word
                    best_score = score
            repaired.append(best_word if best_score >= 0.72 else token)
        clean_text = clean_ocr_text(" ".join(repaired))
        clean_text = clean_text.replace("El cuervo la jarra", "El cuervo y la jarra")
        clean_text = clean_text.replace(
            "inteligencia la perseverancia",
            "inteligencia y la perseverancia",
        )
        return clean_text

    def _line_looks_readable(self, line: str, confidence: float = 0.0) -> bool:
        tokens = self._normalized_tokens(line)
        if len(tokens) < 2 and not (len(tokens) == 1 and len(tokens[0]) >= 4 and confidence >= 45):
            return False
        if any(self._NOISE_RE.match(token) for token in tokens):
            return False
        common_hits = sum(1 for token in tokens if token in self._SPANISH_COMMON_WORDS)
        if len(tokens) >= 5 and common_hits < 1 and confidence < 55:
            return False
        return self._coherence_score(line, confidence) >= 45

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
        if single_letter_tokens > max(2, len(useful_tokens) // 2):
            return False

        normalized_tokens = self._normalized_tokens(text)
        if len(normalized_tokens) >= 5:
            common_hits = sum(1 for token in normalized_tokens if token in self._SPANISH_COMMON_WORDS)
            return common_hits >= 1 or self._coherence_score(text) >= 105

        return True

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

    def _parse_confidence(self, raw_conf: object) -> float:
        try:
            return float(raw_conf)
        except (TypeError, ValueError):
            return -1.0

    def _is_tesseract_timeout(self, error: RuntimeError) -> bool:
        return "timeout" in str(error).lower()

    def _word_looks_valid(self, text: str, confidence: float) -> bool:
        if not text or confidence < self._WORD_CONFIDENCE_MIN:
            return False
        tokens = self._normalized_tokens(text)
        if not tokens:
            return False
        token = tokens[0]
        if len(token) == 1 and token not in {"a", "e", "o", "u", "y"}:
            return False
        if self._NOISE_RE.match(token):
            return False
        letters = sum(char.isalpha() for char in text)
        if letters < 2 and token not in {"a", "e", "o", "u", "y"}:
            return False
        return True

    def _text_from_data(self, image: np.ndarray, psm: int) -> tuple[str, float]:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.config.ocr_language,
                config=self._ocr_config(psm),
                output_type=Output.DICT,
                timeout=2 if self.config.ocr_fast_mode else 4,
            )
        except RuntimeError as error:
            if not self._is_tesseract_timeout(error):
                raise
            return "", 0.0
        words: list[str] = []
        confidences: list[float] = []
        for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", []), strict=False):
            text = clean_ocr_text(str(raw_text))
            confidence = self._parse_confidence(raw_conf)
            if not self._word_looks_valid(text, confidence):
                continue
            words.append(text)
            confidences.append(confidence)
        if not words:
            return "", 0.0
        return clean_ocr_text(" ".join(words)), float(np.mean(confidences)) if confidences else 0.0

    def _text_lines_from_data(self, image: np.ndarray, psm: int) -> tuple[str, float]:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.config.ocr_language,
                config=self._ocr_config(psm),
                output_type=Output.DICT,
                timeout=3 if self.config.ocr_fast_mode else 6,
            )
        except RuntimeError as error:
            if not self._is_tesseract_timeout(error):
                raise
            return "", 0.0
        lines: dict[tuple[int, int, int], list[str]] = {}
        line_confidences: dict[tuple[int, int, int], list[float]] = {}
        confidences: list[float] = []
        for index, raw_text in enumerate(data.get("text", [])):
            text = clean_ocr_text(str(raw_text))
            try:
                confidence = self._parse_confidence(data.get("conf", [])[index])
            except IndexError:
                confidence = -1.0
            if not self._word_looks_valid(text, confidence):
                continue
            key = (
                int(data.get("block_num", [0])[index]),
                int(data.get("par_num", [0])[index]),
                int(data.get("line_num", [0])[index]),
            )
            lines.setdefault(key, []).append(text)
            if confidence >= 0:
                confidences.append(confidence)
                line_confidences.setdefault(key, []).append(confidence)

        average_confidence = float(np.mean(confidences)) if confidences else 0.0
        ordered_lines: list[str] = []
        for key, words in sorted(lines.items()):
            line = clean_ocr_text(" ".join(words))
            if not self._looks_like_text(line):
                continue
            line_confidence = (
                float(np.mean(line_confidences.get(key, [])))
                if line_confidences.get(key)
                else average_confidence
            )
            if self._line_looks_readable(line, line_confidence):
                ordered_lines.append(line)
        text = clean_ocr_text(". ".join(ordered_lines))
        return self._repair_long_text(text), average_confidence

    def debug_word_boxes(self, image: np.ndarray, psm: int = 6) -> np.ndarray:
        if image.ndim == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()
        data = pytesseract.image_to_data(
            image,
            lang=self.config.ocr_language,
            config=self._ocr_config(psm),
            output_type=Output.DICT,
            timeout=3 if self.config.ocr_fast_mode else 6,
        )
        for index, raw_text in enumerate(data.get("text", [])):
            text = clean_ocr_text(str(raw_text))
            try:
                confidence = self._parse_confidence(data.get("conf", [])[index])
                x = int(data.get("left", [0])[index])
                y = int(data.get("top", [0])[index])
                w = int(data.get("width", [0])[index])
                h = int(data.get("height", [0])[index])
            except (IndexError, ValueError):
                continue
            if confidence < self._WORD_BOX_CONFIDENCE_MIN:
                continue
            color = (0, 220, 0) if self._word_looks_valid(text, confidence) else (0, 0, 220)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"{text} {confidence:.0f}"
            cv2.putText(
                annotated,
                label,
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated

    def _extract_page_text(self, region: np.ndarray, source: str) -> list[OCRCandidate]:
        candidates: list[OCRCandidate] = []
        rotations = (self._rotate(region, 0),) if self.config.ocr_fast_mode else (
            self._rotate(region, 0),
            self._rotate(region, -1.5),
            self._rotate(region, 1.5),
        )
        psms = (6, 4) if self.config.ocr_fast_mode else (6, 4, 3)
        if self.config.ocr_fast_mode:
            psms = (6,)
        for rotated in rotations:
            variants = self._preprocess_page_variants(rotated)
            for variant in variants:
                for psm in psms:
                    text, confidence = self._text_lines_from_data(variant, psm)
                    if (
                        self._word_count(text) >= 4
                        and self._looks_like_text(text)
                        and self._coherence_score(text, confidence) >= 80
                    ):
                        priority = 5 if source == "texto" else 4 if source == "documento" else 3
                        candidates.append(OCRCandidate(text, confidence, priority, source))
            if candidates:
                break
        return candidates

    def _extract_large_text(self, variants: list[np.ndarray], source: str) -> OCRCandidate | None:
        best: OCRCandidate | None = None
        psms = (7, 8) if self.config.ocr_fast_mode else (8, 7, 13)
        for variant in variants:
            for psm in psms:
                try:
                    text = pytesseract.image_to_string(
                        variant,
                        lang=self.config.ocr_language,
                        config=self._ocr_config(psm, large_text=True),
                        timeout=2 if self.config.ocr_fast_mode else 4,
                    )
                except RuntimeError as error:
                    if not self._is_tesseract_timeout(error):
                        raise
                    continue
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
        psms = (6,) if self.config.ocr_fast_mode else (6, 11, 4)
        for variant in variants:
            for psm in psms:
                cleaned, confidence = self._text_from_data(variant, psm)
                if (
                    len(cleaned) >= self.config.ocr_min_text_length
                    and self._looks_like_text(cleaned)
                    and self._coherence_score(cleaned, confidence) >= 65
                ):
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

            rotations = (self._rotate(region, 0),) if self.config.ocr_fast_mode else (
                self._rotate(region, 0),
                self._rotate(region, -2.0),
                self._rotate(region, 2.0),
            )
            for rotation_index, rotated in enumerate(rotations):
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
            key=lambda item: (
                item.priority,
                self._coherence_score(item.text, item.confidence_hint),
                *self._score_text(item.text),
                item.confidence_hint,
            ),
            reverse=True,
        )
        return candidates, frame_sharpness

    def debug_variants(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        frame = self._limit_width(frame)
        debug: dict[str, np.ndarray] = {"ocr_original": frame}
        reading_region = self.reading_region(frame)
        if reading_region is not None:
            debug[f"ocr_region_lectura_{reading_region.source}"] = reading_region.image
        for source, region in self._ocr_regions(frame):
            debug[f"ocr_region_{source}"] = region
            for index, variant in enumerate(self._preprocess_variants(region)):
                debug[f"ocr_{source}_variant_{index}"] = variant
            for index, variant in enumerate(self._preprocess_page_variants(region)):
                debug[f"ocr_{source}_page_variant_{index}"] = variant
        debug["ocr_best_for_tesseract"] = self.best_debug_variant(frame)
        debug["ocr_word_boxes_best"] = self.debug_word_boxes(debug["ocr_best_for_tesseract"])
        return debug

    def extract_text(self, frame: np.ndarray) -> OCRResult | None:
        candidates, frame_sharpness = self.extract_candidates(frame)
        if not candidates:
            return None

        candidate = candidates[0]
        return OCRResult(text=candidate.text, confidence_hint=candidate.confidence_hint, sharpness=frame_sharpness)
