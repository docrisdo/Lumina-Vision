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
    debug_dir = ROOT_DIR / "debug_ocr"
    debug_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraManager(config)
    ocr = OCRService(config)

    camera.start()
    try:
        time.sleep(1.0)
        camera.refocus(force=True)
        frame = camera.read()
        cv2.imwrite(str(debug_dir / "ocr_original.jpg"), frame)
        for index, variant in enumerate(ocr._preprocess_variants(frame)):
            cv2.imwrite(str(debug_dir / f"ocr_variant_{index}.jpg"), variant)

        result = ocr.extract_text(frame)
        if result is None:
            print("[Lumina] OCR no detecto texto util.")
            print(f"[Lumina] Imagen guardada en: {debug_dir / 'ocr_original.jpg'}")
            print(f"[Lumina] Variantes guardadas en: {debug_dir / 'ocr_variant_*.jpg'}")
            print("[Lumina] Prueba con texto grande, bien iluminado, a 20-50 cm y sin mover la hoja.")
            return 1

        print("[Lumina] OCR detecto:")
        print(result.text)
        print(f"[Lumina] Nitidez: {result.sharpness:.1f} | confianza aprox: {result.confidence_hint:.1f}")
        print(f"[Lumina] Imagen guardada en: {debug_dir / 'ocr_original.jpg'}")
        print(f"[Lumina] Variantes guardadas en: {debug_dir / 'ocr_variant_*.jpg'}")
        return 0
    finally:
        camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
