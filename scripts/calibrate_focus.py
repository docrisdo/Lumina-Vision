from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.camera import CameraManager
from lumina_vision.config import AppConfig
from lumina_vision.ocr import OCRService


def main() -> int:
    config = AppConfig.load()
    config.camera_width = 1536
    config.camera_height = 864
    config.camera_framerate = 5.0
    config.camera_buffer_count = 2
    config.camera_lens_position = -1.0
    config.camera_focus_settle_seconds = 0.8

    debug_dir = ROOT_DIR / "debug_focus"
    debug_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraManager(config)
    ocr = OCRService(config)
    camera.start()

    results: list[tuple[float, float]] = []
    try:
        print("[Lumina] Coloca una hoja impresa a la distancia real de lectura.")
        print("[Lumina] No muevas la hoja durante el escaneo de enfoque.")
        time.sleep(1.0)

        for lens_position in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]:
            camera.set_lens_position(lens_position)
            time.sleep(0.8)

            best_frame = None
            best_sharpness = -1.0
            for _ in range(6):
                frame = camera.read()
                sharpness = ocr.sharpness(ocr._center_crop(frame))
                if sharpness > best_sharpness:
                    best_frame = frame
                    best_sharpness = sharpness
                time.sleep(0.08)

            results.append((lens_position, best_sharpness))
            if best_frame is not None:
                cv2.imwrite(str(debug_dir / f"focus_{lens_position:.1f}_{best_sharpness:.1f}.jpg"), best_frame)
            print(f"[Lumina] LensPosition={lens_position:.1f} nitidez={best_sharpness:.1f}")

        best_lens, best_score = max(results, key=lambda item: item[1])
        print("")
        print(f"[Lumina] Mejor enfoque: LUMINA_CAMERA_LENS_POSITION={best_lens:.1f}")
        print(f"[Lumina] Nitidez maxima: {best_score:.1f}")
        print("[Lumina] Agrega ese valor a tu .env para lectura de texto.")
        return 0
    finally:
        camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
