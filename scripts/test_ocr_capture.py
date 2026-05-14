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


def _configure_for_page_test(config: AppConfig) -> None:
    config.camera_width = 1536
    config.camera_height = 864
    config.camera_framerate = 5.0
    config.camera_buffer_count = 2
    config.camera_focus_settle_seconds = max(config.camera_focus_settle_seconds, 0.8)
    config.ocr_max_width = 1536
    config.ocr_page_mode = True
    config.ocr_fast_mode = True


def _sharpness_label(sharpness: float) -> tuple[str, tuple[int, int, int]]:
    if sharpness >= 80:
        return "ENFOQUE BUENO", (0, 220, 0)
    if sharpness >= 35:
        return "ENFOQUE ACEPTABLE", (0, 200, 255)
    return "BORROSO: acerca/aleja y presiona F", (0, 0, 255)


def _best_sharp_frame(camera: CameraManager, ocr: OCRService, samples: int = 8):
    best_frame = None
    best_sharpness = -1.0
    for _ in range(samples):
        frame = camera.read()
        sharpness = ocr.sharpness(ocr._center_crop(frame))
        if sharpness > best_sharpness:
            best_frame = frame
            best_sharpness = sharpness
        time.sleep(0.08)
    return best_frame, best_sharpness


def _preview_capture(camera: CameraManager, ocr: OCRService):
    print("[Lumina] Acomoda el texto dentro de la ventana.")
    print("[Lumina] Presiona ESPACIO para capturar, F para reenfocar, Q para salir.")
    while True:
        frame = camera.read()
        preview = frame.copy()
        height, width = preview.shape[:2]
        page_x1, page_y1, page_x2, page_y2 = ocr.roi_box(frame)
        live_sharpness = ocr.sharpness(ocr._center_crop(frame))
        focus_text, focus_color = _sharpness_label(live_sharpness)
        cv2.rectangle(
            preview,
            (page_x1, page_y1),
            (page_x2, page_y2),
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
        cv2.putText(
            preview,
            f"{focus_text} | nitidez={live_sharpness:.1f}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            focus_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            "Solo se procesa el rectangulo amarillo. Evita pantalla/reflejos/fondo.",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Lumina OCR Capture", preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord("f"):
            camera.autofocus_cycle()
        if key == 32:
            camera.autofocus_cycle()
            frame, sharpness = _best_sharp_frame(camera, ocr)
            if sharpness < 35:
                print(f"[Lumina] Captura borrosa: nitidez={sharpness:.1f}. No conviene procesar OCR.")
                print("[Lumina] Acerca/aleja la hoja, mejora luz y vuelve a presionar F.")
                continue
            print(f"[Lumina] Captura elegida con nitidez: {sharpness:.1f}")
            return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura una imagen y prueba OCR.")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Captura directo sin ventana. Util para pruebas sin escritorio.",
    )
    args = parser.parse_args()

    config = AppConfig.load()
    _configure_for_page_test(config)
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
        for name, image in ocr.debug_variants(frame).items():
            cv2.imwrite(str(debug_dir / f"{name}.jpg"), image)

        candidates, sharpness = ocr.extract_candidates(frame)
        result = ocr.extract_text(frame)
        if result is None:
            print("[Lumina] OCR no detecto texto util.")
            print(f"[Lumina] Imagen guardada en: {debug_dir / 'ocr_original.jpg'}")
            print(f"[Lumina] Nitidez: {sharpness:.1f}")
            print(f"[Lumina] Variantes guardadas en: {debug_dir / 'ocr_*_variant_*.jpg'}")
            print(f"[Lumina] Regiones guardadas en: {debug_dir / 'ocr_region_*.jpg'}")
            print(f"[Lumina] Mejor variante guardada en: {debug_dir / 'ocr_best_for_tesseract.jpg'}")
            print("[Lumina] Prueba con hoja bien iluminada, centrada, a 25-45 cm y sin moverla.")
            print("[Lumina] Diagnostico pagina: tesseract debug_ocr/ocr_best_for_tesseract.jpg stdout -l spa+eng --psm 6")
            return 1

        print("[Lumina] OCR detecto:")
        print(result.text)
        print(f"[Lumina] Nitidez: {result.sharpness:.1f} | confianza aprox: {result.confidence_hint:.1f}")
        print("[Lumina] Mejores candidatos:")
        for index, candidate in enumerate(candidates[:5], start=1):
            print(
                f"  {index}. fuente={candidate.source} prioridad={candidate.priority} "
                f"conf={candidate.confidence_hint:.1f}: {candidate.text[:120]}",
            )
        print(f"[Lumina] Imagen guardada en: {debug_dir / 'ocr_original.jpg'}")
        print(f"[Lumina] Variantes guardadas en: {debug_dir / 'ocr_*_variant_*.jpg'}")
        print(f"[Lumina] Regiones guardadas en: {debug_dir / 'ocr_region_*.jpg'}")
        print(f"[Lumina] Mejor variante guardada en: {debug_dir / 'ocr_best_for_tesseract.jpg'}")
        return 0
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
