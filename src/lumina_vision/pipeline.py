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
from lumina_vision.ultrasonic import UltrasonicMonitor
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
    "persona",
}


class LuminaPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.camera = CameraManager(config)
        self.detector = ObjectDetector(config)
        self.ocr = OCRService(config)
        self.speech = SpeechEngine(config)
        self.ultrasonic = UltrasonicMonitor(config)
        self._object_speech_gate = CooldownGate(config.speech_object_cooldown_seconds)
        self._ocr_speech_gate = CooldownGate(config.speech_ocr_cooldown_seconds)
        self._ultrasonic_speech_gate = CooldownGate(config.ultrasonic_alert_cooldown_seconds)
        self._last_ocr_at = 0.0
        self._last_ocr_speech_at = 0.0
        self._last_object_speech_at = 0.0
        self._frame_index = 0
        self._last_ocr_text = ""
        self._last_object_signature = ""
        self._latest_detections: list[Detection] = []
        self._latest_detection_at = 0.0
        self._latest_ocr_text = ""
        self._latest_ocr_sharpness = 0.0
        self._latest_ocr_status = "OCR esperando texto"
        self._ocr_candidate_text = ""
        self._ocr_candidate_count = 0
        self._force_ocr = False
        self._ocr_lock = threading.Lock()
        self._ocr_worker: threading.Thread | None = None

    def run(self) -> int:
        self.camera.start()

        if self.config.enable_tts:
            self.speech.start()
            if self.config.enable_ultrasonic:
                logger.info("Preparando frases de alerta ultrasonica en segundo plano.")
                self.speech.warmup_async(self._ultrasonic_warmup_phrases())
            if self.config.tts_warmup:
                self.speech.warmup_async(
                    [
                        "Veo una persona",
                        "Veo un libro",
                        "Veo una mochila",
                        "Veo una laptop",
                        "Veo un celular",
                        "Veo una botella",
                        "Veo una silla",
                        "Veo una mesa",
                    ],
                )

        self.ultrasonic.start()

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
                ocr_text = ""

                self._frame_index += 1

                if detection_ready and self._frame_index % self.config.detection_run_every_n_frames == 0:
                    self._latest_detections = self.detector.detect(frame)
                    self._latest_detection_at = now_monotonic()

                detections = self._visible_detections()

                if self.config.enable_ocr and self.config.ocr_auto_read:
                    self._maybe_start_ocr(frame)
                    with self._ocr_lock:
                        ocr_text = self._latest_ocr_text

                annotated = self._annotate_frame(frame, detections, ocr_text)
                self._handle_ultrasonic_alert()
                if not self.ultrasonic.close_obstacle_confirmed():
                    self._handle_speech(detections, ocr_text)

                if self.config.save_debug_frames and self._frame_index % 30 == 0:
                    file_path = debug_dir / f"frame_{self._frame_index:06d}.jpg"
                    cv2.imwrite(str(file_path), annotated)

                if self.config.show_preview:
                    cv2.imshow("Lumina Vision", self._resize_preview(annotated))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("r"):
                        self._force_ocr = True
                        logger.info("Lectura OCR manual solicitada.")
                    if key == ord("f"):
                        self.camera.refocus(force=True)
                        logger.info("Reenfoque manual solicitado.")
        finally:
            self.camera.stop()
            self.ultrasonic.stop()
            self.speech.stop()
            if self.config.show_preview:
                cv2.destroyAllWindows()

        return 0

    def _should_run_ocr(self) -> bool:
        return (now_monotonic() - self._last_ocr_at) >= self.config.ocr_run_interval_seconds

    def _maybe_start_ocr(self, frame) -> None:
        if not self._force_ocr and not self._should_run_ocr():
            return
        if self._ocr_worker is not None and self._ocr_worker.is_alive():
            return

        frame_copy = frame.copy()
        force_ocr = self._force_ocr
        self._force_ocr = False
        self._last_ocr_at = now_monotonic()
        self._ocr_worker = threading.Thread(
            target=self._run_ocr_job,
            args=(frame_copy, force_ocr),
            name="lumina-ocr",
            daemon=True,
        )
        self._ocr_worker.start()

    def _run_ocr_job(self, frame, force_ocr: bool) -> None:
        self.camera.refocus(force=force_ocr)
        try:
            frame = self.camera.read()
        except Exception as exc:
            logger.debug("OCR usara el frame anterior porque no pudo capturar uno nuevo: {}", exc)
        result = self.ocr.extract_text(frame)
        with self._ocr_lock:
            if result is None:
                self._latest_ocr_status = "OCR: texto borroso o no detectado"
                self._latest_ocr_text = ""
                self._latest_ocr_sharpness = 0.0
                return

            self._latest_ocr_sharpness = result.sharpness
            if result.text == self._ocr_candidate_text:
                self._ocr_candidate_count += 1
            else:
                self._ocr_candidate_text = result.text
                self._ocr_candidate_count = 1

            if force_ocr or self._ocr_candidate_count >= self.config.ocr_stable_reads:
                self._latest_ocr_text = result.text
                self._latest_ocr_status = f"OCR: {result.text[:80]}"
            else:
                self._latest_ocr_status = "OCR: confirmando texto"
        if result is not None:
            logger.info("OCR detecto texto: {} | nitidez {:.1f}", result.text[:160], result.sharpness)

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

    def _visible_detections(self) -> list[Detection]:
        if not self._latest_detections:
            return []
        if (now_monotonic() - self._latest_detection_at) > 2.0:
            return []
        return self._latest_detections

    def _handle_speech(self, detections: list[Detection], ocr_text: str) -> None:
        if not self.config.enable_tts:
            return

        if (
            self.config.speech_enable_ocr
            and ocr_text
            and ocr_text != self._last_ocr_text
            and self._ocr_speech_gate.ready()
        ):
            logger.info("Anunciando texto por voz: {}", ocr_text[:160])
            self.speech.speak(f"Leo: {ocr_text}")
            self._last_ocr_text = ocr_text
            self._last_ocr_speech_at = now_monotonic()
            self._ocr_speech_gate.mark()
            return

        if self._objects_suppressed_by_ocr():
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
            same_object_recently_spoken = (
                signature == self._last_object_signature
                and (now_monotonic() - self._last_object_speech_at)
                < self.config.speech_repeat_same_object_seconds
            )
            if same_object_recently_spoken or not self._object_speech_gate.ready():
                return
            message = self._format_object_message(counter)
            logger.info("Anunciando objetos por voz: {}", message)
            self.speech.speak(message)
            self._last_object_signature = signature
            self._last_object_speech_at = now_monotonic()
            self._object_speech_gate.mark()

    def _handle_ultrasonic_alert(self) -> None:
        if not self.config.enable_tts or not self.config.enable_ultrasonic:
            return
        if not self.ultrasonic.close_obstacle_confirmed():
            return
        if not self._ultrasonic_speech_gate.ready():
            return

        message = self._format_ultrasonic_message()

        logger.info("Alerta ultrasonica: {}", message)
        self.speech.speak(message, priority=True)
        self._ultrasonic_speech_gate.mark()

    def _format_ultrasonic_message(self) -> str:
        distance = self.ultrasonic.latest_distance_cm
        if distance is None:
            return "Cuidado. Hay un objeto muy cerca."

        distance_cm = max(1, int(round(distance)))
        unit = "centimetro" if distance_cm == 1 else "centimetros"
        return f"Cuidado. Hay un objeto a {distance_cm} {unit}."

    def _ultrasonic_warmup_phrases(self) -> list[str]:
        max_cm = max(1, int(round(self.config.ultrasonic_alert_distance_cm)))
        return [
            f"Cuidado. Hay un objeto a {distance_cm} {'centimetro' if distance_cm == 1 else 'centimetros'}."
            for distance_cm in range(1, max_cm + 1)
        ]

    def _objects_suppressed_by_ocr(self) -> bool:
        return (
            self._last_ocr_speech_at > 0
            and (now_monotonic() - self._last_ocr_speech_at)
            < self.config.ocr_suppress_objects_seconds
        )

    def _format_object_message(self, counter: Counter[str]) -> str:
        labels = counter.most_common(2 if self.config.wearable_mode else 3)
        if not labels:
            return ""
        parts = []
        for label, count in labels:
            if count > 1:
                parts.append(f"{count} {label}")
            else:
                article = "una" if label.endswith("a") else "un"
                parts.append(f"{article} {label}")
        return f"Veo {', '.join(parts)}"

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
            f"MODO={'LENTES' if self.config.wearable_mode else 'PRUEBA'} "
            f"OBJ={'ON' if self.config.enable_object_detection else 'OFF'} "
            f"OCR={'ON' if self.config.enable_ocr else 'OFF'} "
            f"TTS={'ON' if self.config.enable_tts else 'OFF'} "
            f"US={'ON' if self.config.enable_ultrasonic else 'OFF'}"
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
        if not self.config.wearable_mode:
            cv2.putText(
                annotated,
                "Diagnostico: R leer texto | F reenfocar | Q salir",
                (20, annotated.shape[0] - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (50, 200, 255),
                2,
                cv2.LINE_AA,
            )
        if self._latest_ocr_status:
            cv2.putText(
                annotated,
                self._latest_ocr_status[:90],
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 230, 60),
                2,
                cv2.LINE_AA,
            )
        return annotated
