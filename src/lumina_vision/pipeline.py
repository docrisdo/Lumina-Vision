from __future__ import annotations

from collections import Counter
from pathlib import Path
import threading

import cv2
from loguru import logger

from lumina_vision.camera import CameraManager
from lumina_vision.config import AppConfig
from lumina_vision.detectors.tflite_detector import Detection, ObjectDetector
from lumina_vision.ocr import OCRService
from lumina_vision.speech import SpeechEngine
from lumina_vision.utils import CooldownGate, now_monotonic


SCHOOL_OBJECT_PRIORITY = {
    "libro",
    "mochila",
    "laptop",
    "celular",
    "botella",
    "tijeras",
    "teclado",
    "raton",
    "silla",
    "mesa",
    "reloj",
}


class LuminaPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.camera = CameraManager(config)
        self.detector = ObjectDetector(config)
        self.ocr = OCRService(config)
        self.speech = SpeechEngine(config)
        self._speech_gate = CooldownGate(config.speech_cooldown_seconds)
        self._last_ocr_at = 0.0
        self._frame_index = 0
        self._last_ocr_text = ""
        self._last_object_signature = ""
        self._latest_ocr_text = ""
        self._ocr_lock = threading.Lock()
        self._ocr_worker: threading.Thread | None = None

    def run(self) -> int:
        self.camera.start()

        if self.config.enable_tts:
            self.speech.start()

        detection_ready = False
        if self.config.enable_object_detection:
            try:
                self.detector.load()
                detection_ready = True
            except Exception as exc:
                logger.warning("Deteccion de objetos deshabilitada: {}", exc)

        debug_dir = Path("debug_frames")
        if self.config.save_debug_frames:
            debug_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                frame = self.camera.read()
                detections: list[Detection] = []
                ocr_text = ""

                self._frame_index += 1

                if detection_ready and self._frame_index % self.config.detection_run_every_n_frames == 0:
                    detections = self.detector.detect(frame)

                if self.config.enable_ocr:
                    self._maybe_start_ocr(frame)
                    with self._ocr_lock:
                        ocr_text = self._latest_ocr_text

                annotated = self._annotate_frame(frame, detections, ocr_text)
                self._handle_speech(detections, ocr_text)

                if self.config.save_debug_frames and self._frame_index % 30 == 0:
                    file_path = debug_dir / f"frame_{self._frame_index:06d}.jpg"
                    cv2.imwrite(str(file_path), annotated)

                if self.config.show_preview:
                    cv2.imshow("Lumina Vision", self._resize_preview(annotated))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            self.camera.stop()
            self.speech.stop()
            if self.config.show_preview:
                cv2.destroyAllWindows()

        return 0

    def _should_run_ocr(self) -> bool:
        return (now_monotonic() - self._last_ocr_at) >= self.config.ocr_run_interval_seconds

    def _maybe_start_ocr(self, frame) -> None:
        if not self._should_run_ocr():
            return
        if self._ocr_worker is not None and self._ocr_worker.is_alive():
            return

        frame_copy = frame.copy()
        self._last_ocr_at = now_monotonic()
        self._ocr_worker = threading.Thread(
            target=self._run_ocr_job,
            args=(frame_copy,),
            name="lumina-ocr",
            daemon=True,
        )
        self._ocr_worker.start()

    def _run_ocr_job(self, frame) -> None:
        self.camera.refocus()
        result = self.ocr.extract_text(frame)
        with self._ocr_lock:
            self._latest_ocr_text = result.text if result is not None else ""
        if result is not None:
            logger.info("OCR detecto texto: {}", result.text[:160])
        else:
            logger.debug("OCR no detecto texto util.")

    def _resize_preview(self, frame):
        height, width = frame.shape[:2]
        if width <= self.config.preview_max_width:
            return frame
        scale = self.config.preview_max_width / float(width)
        return cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    def _handle_speech(self, detections: list[Detection], ocr_text: str) -> None:
        if not self.config.enable_tts or not self._speech_gate.ready():
            return

        if self.config.speech_enable_ocr and ocr_text and ocr_text != self._last_ocr_text:
            logger.info("Anunciando texto por voz: {}", ocr_text[:160])
            self.speech.speak(f"Texto detectado: {ocr_text}")
            self._last_ocr_text = ocr_text
            self._speech_gate.mark()
            return

        if self.config.speech_enable_objects and detections:
            ordered_detections = sorted(
                detections,
                key=lambda detection: (
                    detection.label not in SCHOOL_OBJECT_PRIORITY,
                    -detection.score,
                ),
            )
            labels = [d.label for d in ordered_detections]
            counter = Counter(labels)
            signature = "|".join(
                f"{label}:{count}"
                for label, count in counter.most_common(3)
            )
            if signature == self._last_object_signature:
                return
            message = ", ".join(
                f"{count} {label}" if count > 1 else f"un {label}"
                for label, count in counter.most_common(3)
            )
            logger.info("Anunciando objetos por voz: {}", message)
            self.speech.speak(message)
            self._last_object_signature = signature
            self._speech_gate.mark()

    def _annotate_frame(self, frame, detections: list[Detection], ocr_text: str):
        annotated = frame.copy()
        for detection in detections:
            left, top, right, bottom = detection.box
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 220, 0), 2)
            label = f"{detection.label} {detection.score:.2f}"
            cv2.putText(
                annotated,
                label,
                (left, max(20, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

        if ocr_text:
            display_text = ocr_text[:100]
            cv2.rectangle(annotated, (10, 10), (annotated.shape[1] - 10, 70), (10, 10, 10), -1)
            cv2.putText(
                annotated,
                display_text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        status = (
            f"OBJ={'ON' if self.config.enable_object_detection else 'OFF'} "
            f"OCR={'ON' if self.config.enable_ocr else 'OFF'} "
            f"TTS={'ON' if self.config.enable_tts else 'OFF'}"
        )
        cv2.putText(
            annotated,
            status,
            (20, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 200, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated
