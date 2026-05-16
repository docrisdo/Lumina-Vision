from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
import unicodedata

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.camera import CameraManager
from lumina_vision.config import AppConfig
from lumina_vision.ocr import OCRCandidate, OCRService


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return text


def _expected_bonus(text: str, expected: str) -> float:
    if not expected:
        return 0.0
    normalized_text = _normalize(text)
    normalized_expected = _normalize(expected)
    if not normalized_expected:
        return 0.0
    if normalized_expected in normalized_text:
        return 300.0
    expected_tokens = set(normalized_expected.split())
    text_tokens = set(normalized_text.split())
    if not expected_tokens:
        return 0.0
    matches = len(expected_tokens & text_tokens)
    return 200.0 * (matches / len(expected_tokens))


def _candidate_score(candidate: OCRCandidate | None, sharpness: float, expected: str) -> float:
    if candidate is None:
        return sharpness * 0.15
    letters = sum(char.isalpha() for char in candidate.text)
    words = len(re.findall(r"[^\W\d_]{2,}", candidate.text, flags=re.UNICODE))
    return (
        candidate.priority * 80.0
        + candidate.confidence_hint
        + words * 18.0
        + letters * 1.5
        + sharpness * 0.25
        + _expected_bonus(candidate.text, expected)
    )


def _configure(config: AppConfig) -> None:
    config.camera_width = 1536
    config.camera_height = 864
    config.camera_framerate = 5.0
    config.camera_buffer_count = 2
    config.camera_focus_settle_seconds = 0.9
    config.camera_refocus_before_ocr = False
    config.ocr_max_width = 1536
    config.ocr_page_mode = True
    config.ocr_prefer_center_crop = True
    config.ocr_fast_mode = True
    config.ocr_min_text_length = 2


def _capture_best(camera: CameraManager, ocr: OCRService, samples: int) -> tuple[object, float]:
    best_frame = None
    best_sharpness = -1.0
    for _ in range(samples):
        frame = camera.read()
        sharpness = ocr.sharpness(ocr.focus_region(frame))
        if sharpness > best_sharpness:
            best_frame = frame
            best_sharpness = sharpness
        time.sleep(0.08)
    if best_frame is None:
        raise RuntimeError("No se pudo capturar frame para OCR.")
    return best_frame, best_sharpness


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibra enfoque y OCR para texto escolar.")
    parser.add_argument(
        "--expected",
        default="",
        help="Texto esperado para calibrar, por ejemplo: HOLA o El patito feo.",
    )
    parser.add_argument(
        "--positions",
        default="2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0,8.0",
        help="Lista de LensPosition separada por comas.",
    )
    parser.add_argument("--samples", type=int, default=5, help="Frames por posicion de lente.")
    args = parser.parse_args()

    lens_positions = [float(value.strip()) for value in args.positions.split(",") if value.strip()]
    if not lens_positions:
        raise SystemExit("[Lumina] No hay posiciones de lente para probar.")

    config = AppConfig.load()
    _configure(config)
    debug_dir = ROOT_DIR / "debug_ocr_calibration"
    debug_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraManager(config)
    ocr = OCRService(config)
    results: list[tuple[float, float, float, OCRCandidate | None, object]] = []

    print("[Lumina] Coloca el texto dentro de la guia, bien iluminado y sin moverlo.")
    if args.expected:
        print(f"[Lumina] Texto esperado para la calibracion: {args.expected}")
    print("[Lumina] Probando posiciones de lente. Esto puede tardar 1-2 minutos.")

    camera.start()
    try:
        time.sleep(1.0)
        for lens_position in lens_positions:
            camera.set_lens_position(lens_position)
            time.sleep(config.camera_focus_settle_seconds)
            frame, sharpness = _capture_best(camera, ocr, args.samples)
            candidates, _frame_sharpness = ocr.extract_candidates(frame)
            best_candidate = candidates[0] if candidates else None
            score = _candidate_score(best_candidate, sharpness, args.expected)
            results.append((score, lens_position, sharpness, best_candidate, frame))

            cv2.imwrite(str(debug_dir / f"lens_{lens_position:.1f}_original.jpg"), frame)
            if best_candidate is None:
                print(f"[Lumina] LensPosition={lens_position:.1f} nitidez={sharpness:.1f} score={score:.1f} texto=NO")
            else:
                print(
                    f"[Lumina] LensPosition={lens_position:.1f} nitidez={sharpness:.1f} "
                    f"score={score:.1f} texto={best_candidate.text[:100]}",
                )

        best_score, best_lens, best_sharpness, best_candidate, best_frame = max(results, key=lambda item: item[0])
        for name, image in ocr.debug_variants(best_frame).items():
            cv2.imwrite(str(debug_dir / f"best_{name}.jpg"), image)

        print("")
        print(f"[Lumina] Mejor calibracion OCR: LUMINA_CAMERA_LENS_POSITION={best_lens:.1f}")
        print(f"[Lumina] Nitidez: {best_sharpness:.1f} | score OCR: {best_score:.1f}")
        if best_candidate is not None:
            print(f"[Lumina] Texto leido: {best_candidate.text}")
        else:
            print("[Lumina] No se leyo texto util en ninguna posicion. Revisa luz, distancia y que la hoja llene la guia.")
        print(f"[Lumina] Diagnosticos guardados en: {debug_dir}")
        print("[Lumina] Agrega el valor ganador a tu .env y vuelve a probar scripts/test_ocr_capture.py.")
        return 0 if best_candidate is not None else 1
    finally:
        camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
