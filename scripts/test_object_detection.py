from __future__ import annotations

import argparse
from collections import Counter
import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.camera import CameraManager
from lumina_vision.config import AppConfig
from lumina_vision.detectors.tflite_detector import Detection, ObjectDetector
from lumina_vision.speech import SpeechEngine


SCHOOL_OBJECT_PRIORITY = {
    "libro",
    "libreta",
    "cuaderno",
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


def _draw_detections(frame, detections, fps: float) -> None:
    for detection in detections:
        left, top, right, bottom = detection.box
        label = f"{detection.label} {detection.score:.2f}"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Objetos: {len(detections)} | FPS aprox: {fps:.1f} | Q=salir",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _detection_signature(detections: list[Detection]) -> str:
    labels = [detection.label for detection in detections]
    counter = Counter(labels)
    return "|".join(f"{label}:{count}" for label, count in counter.most_common(3))


def _format_detection_message(detections: list[Detection]) -> str:
    ordered_detections = sorted(
        detections,
        key=lambda detection: (
            detection.label not in SCHOOL_OBJECT_PRIORITY,
            -detection.score,
        ),
    )
    counter = Counter(detection.label for detection in ordered_detections)
    parts = []
    for label, count in counter.most_common(3):
        if count > 1:
            parts.append(f"{count} {label}")
        else:
            article = "una" if label.endswith("a") else "un"
            parts.append(f"{article} {label}")
    return f"Veo {', '.join(parts)}" if parts else ""


def _maybe_speak_detections(
    speech: SpeechEngine,
    detections: list[Detection],
    *,
    last_signature: str,
    last_speech_at: float,
    repeat_seconds: float,
) -> tuple[str, float]:
    if not detections:
        return "", last_speech_at

    signature = _detection_signature(detections)
    now = time.monotonic()
    if signature == last_signature and (now - last_speech_at) < repeat_seconds:
        return last_signature, last_speech_at

    message = _format_detection_message(detections)
    if message:
        print(f"[Lumina] Voz objetos: {message}")
        speech.speak(message, priority=True)
        return signature, now
    return last_signature, last_speech_at


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba unitaria de deteccion de objetos.")
    parser.add_argument("--no-preview", action="store_true", help="No abre ventana; guarda una imagen de diagnostico.")
    parser.add_argument("--no-speech", action="store_true", help="No dice los objetos detectados por voz.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Tiempo maximo de prueba. 0 = hasta presionar Q.")
    args = parser.parse_args()

    config = AppConfig.load()
    config.enable_tts = True
    if config.tts_engine.lower() == "auto":
        config.tts_engine = "piper"
    detector = ObjectDetector(config)
    if not detector.available():
        print("[Lumina] Detector no disponible.")
        print(f"[Lumina] Modelo: {config.detection_model_path}")
        print("[Lumina] Instala tflite_runtime, ai-edge-litert o tensorflow y descarga el modelo TFLite.")
        return 1

    camera = CameraManager(config)
    speech = SpeechEngine(config)
    debug_dir = ROOT_DIR / "debug_objects"
    last_detections = []
    last_signature = ""
    last_speech_at = 0.0
    frames = 0
    started_at = time.perf_counter()
    fps = 0.0

    try:
        detector.load()
        if not args.no_speech:
            speech.start()
        camera.start()
        print("[Lumina] Prueba de objetos iniciada. Presiona Q para salir.")
        while True:
            frame = camera.read()
            frames += 1
            loop_started = time.perf_counter()
            detections = detector.detect(frame)
            last_detections = detections
            elapsed = max(0.001, time.perf_counter() - started_at)
            fps = frames / elapsed
            if not args.no_speech:
                last_signature, last_speech_at = _maybe_speak_detections(
                    speech,
                    detections,
                    last_signature=last_signature,
                    last_speech_at=last_speech_at,
                    repeat_seconds=config.speech_repeat_same_object_seconds,
                )

            if args.no_preview:
                debug_dir.mkdir(parents=True, exist_ok=True)
                _draw_detections(frame, detections, fps)
                output_path = debug_dir / "object_detection_test.jpg"
                cv2.imwrite(str(output_path), frame)
                print(f"[Lumina] Detecciones: {[item.label for item in detections]}")
                print(f"[Lumina] Imagen guardada en: {output_path}")
                if not args.no_speech:
                    speech.wait_until_done()
                return 0

            _draw_detections(frame, detections, fps)
            cv2.imshow("Lumina Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                break
            if args.seconds > 0 and (time.perf_counter() - started_at) >= args.seconds:
                break

            sleep_time = max(0.0, (1.0 / max(1.0, config.camera_framerate)) - (time.perf_counter() - loop_started))
            if sleep_time:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        speech.stop()
        cv2.destroyAllWindows()

    print(f"[Lumina] FPS aprox: {fps:.1f}")
    print(f"[Lumina] Ultimas detecciones: {[item.label for item in last_detections]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
