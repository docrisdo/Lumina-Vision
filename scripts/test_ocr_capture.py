from __future__ import annotations

import argparse
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


def _preview_capture(camera: CameraManager, ocr: OCRService):
    print("[Lumina] Acomoda el texto dentro de la ventana.")
    print("[Lumina] Presiona ESPACIO para capturar, F para reenfocar, Q para salir.")
    while True:
        frame = camera.read()
        preview = frame.copy()
        height, width = preview.shape[:2]
        cv2.rectangle(
            preview,
            (int(width * 0.08), int(height * 0.12)),
            (int(width * 0.92), int(height * 0.88)),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "ESPACIO=capturar | F=enfocar | Q=salir",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Lumina OCR Capture", preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord("f"):
            camera.refocus(force=True)
            time.sleep(0.3)
        if key == 32:
            camera.refocus(force=True)
            time.sleep(0.3)
            return camera.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura una imagen y prueba OCR.")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Captura directo sin ventana. Util para pruebas sin escritorio.",
    )
    args = parser.parse_args()

    config = AppConfig.load()
    debug_dir = ROOT_DIR / "debug_ocr"
    debug_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraManager(config)
    ocr = OCRService(config)

    camera.start()
    try:
        time.sleep(1.0)
        if args.no_preview:
            camera.refocus(force=True)
            time.sleep(0.3)
            frame = camera.read()
        else:
            frame = _preview_capture(camera, ocr)
        cv2.imwrite(str(debug_dir / "ocr_original.jpg"), frame)
        for index, variant in enumerate(ocr._preprocess_variants(ocr._center_crop(frame))):
            cv2.imwrite(str(debug_dir / f"ocr_variant_{index}.jpg"), variant)
        cv2.imwrite(str(debug_dir / "ocr_best_for_tesseract.jpg"), ocr.best_debug_variant(frame))

        result = ocr.extract_text(frame)
        if result is None:
            print("[Lumina] OCR no detecto texto util.")
            print(f"[Lumina] Imagen guardada en: {debug_dir / 'ocr_original.jpg'}")
            print(f"[Lumina] Variantes guardadas en: {debug_dir / 'ocr_variant_*.jpg'}")
            print(f"[Lumina] Mejor variante guardada en: {debug_dir / 'ocr_best_for_tesseract.jpg'}")
            print("[Lumina] Prueba con texto grande, bien iluminado, a 20-50 cm y sin mover la hoja.")
            print('[Lumina] Diagnostico: tesseract debug_ocr/ocr_best_for_tesseract.jpg stdout -l spa+eng --psm 8')
            return 1

        print("[Lumina] OCR detecto:")
        print(result.text)
        print(f"[Lumina] Nitidez: {result.sharpness:.1f} | confianza aprox: {result.confidence_hint:.1f}")
        print(f"[Lumina] Imagen guardada en: {debug_dir / 'ocr_original.jpg'}")
        print(f"[Lumina] Variantes guardadas en: {debug_dir / 'ocr_variant_*.jpg'}")
        print(f"[Lumina] Mejor variante guardada en: {debug_dir / 'ocr_best_for_tesseract.jpg'}")
        return 0
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
