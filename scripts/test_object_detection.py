from __future__ import annotations

import argparse
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
from lumina_vision.detectors.tflite_detector import ObjectDetector


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba unitaria de deteccion de objetos.")
    parser.add_argument("--no-preview", action="store_true", help="No abre ventana; guarda una imagen de diagnostico.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Tiempo maximo de prueba. 0 = hasta presionar Q.")
    args = parser.parse_args()

    config = AppConfig.load()
    detector = ObjectDetector(config)
    if not detector.available():
        print("[Lumina] Detector no disponible.")
        print(f"[Lumina] Modelo: {config.detection_model_path}")
        print("[Lumina] Instala tflite_runtime, ai-edge-litert o tensorflow y descarga el modelo TFLite.")
        return 1

    camera = CameraManager(config)
    debug_dir = ROOT_DIR / "debug_objects"
    last_detections = []
    frames = 0
    started_at = time.perf_counter()
    fps = 0.0

    try:
        detector.load()
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

            if args.no_preview:
                debug_dir.mkdir(parents=True, exist_ok=True)
                _draw_detections(frame, detections, fps)
                output_path = debug_dir / "object_detection_test.jpg"
                cv2.imwrite(str(output_path), frame)
                print(f"[Lumina] Detecciones: {[item.label for item in detections]}")
                print(f"[Lumina] Imagen guardada en: {output_path}")
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
        cv2.destroyAllWindows()

    print(f"[Lumina] FPS aprox: {fps:.1f}")
    print(f"[Lumina] Ultimas detecciones: {[item.label for item in last_detections]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
