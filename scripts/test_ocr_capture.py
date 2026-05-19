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
from lumina_vision.ocr import OCRService
from lumina_vision.speech import SpeechEngine

DEFAULT_EXPECTED_TEXT = ROOT_DIR / "fallback_text" / "el_cuervo_y_la_jarra.txt"
EXPECTED_STOPWORDS = {
    "con",
    "del",
    "dentro",
    "el",
    "en",
    "este",
    "hasta",
    "la",
    "las",
    "los",
    "para",
    "pero",
    "por",
    "que",
    "sobre",
    "una",
    "uno",
}


def _configure_for_page_test(config: AppConfig) -> None:
    config.camera_width = 1536
    config.camera_height = 864
    config.camera_framerate = 5.0
    config.camera_buffer_count = 2
    config.camera_focus_settle_seconds = max(config.camera_focus_settle_seconds, 0.8)
    config.ocr_max_width = 1536
    config.ocr_page_mode = True
    config.ocr_fast_mode = True
    config.enable_tts = True
    config.tts_command_timeout_seconds = max(config.tts_command_timeout_seconds, 45.0)
    if config.tts_engine.lower() == "auto":
        config.tts_engine = "piper"


def _sharpness_label(sharpness: float) -> tuple[str, tuple[int, int, int]]:
    if sharpness >= 55:
        return "ENFOQUE BUENO", (0, 220, 0)
    if sharpness >= 18:
        return "ENFOQUE ACEPTABLE", (0, 200, 255)
    return "BORROSO: acerca/aleja y presiona F", (0, 0, 255)


def _best_sharp_frame(camera: CameraManager, ocr: OCRService, samples: int = 8):
    best_frame = None
    best_sharpness = -1.0
    for _ in range(samples):
        frame = camera.read()
        sharpness = ocr.sharpness(ocr.focus_region(frame))
        if sharpness > best_sharpness:
            best_frame = frame
            best_sharpness = sharpness
        time.sleep(0.08)
    return best_frame, best_sharpness


def _preview_capture(camera: CameraManager, ocr: OCRService):
    print("[Lumina] Acomoda el texto dentro de la ventana.")
    print("[Lumina] ESPACIO=capturar, F=enfocar, R=rotar, Q=salir.")
    while True:
        frame = camera.read()
        preview = frame.copy()
        height, width = preview.shape[:2]
        page_x1, page_y1, page_x2, page_y2 = ocr.roi_box(frame)
        document_box = ocr.document_box(frame)
        reading_box = ocr.reading_box(frame)
        live_sharpness = ocr.sharpness(ocr.focus_region(frame))
        focus_text, focus_color = _sharpness_label(live_sharpness)
        if document_box is not None:
            cv2.drawContours(preview, [document_box], -1, (0, 220, 0), 3)
        cv2.rectangle(
            preview,
            (page_x1, page_y1),
            (page_x2, page_y2),
            (0, 255, 255),
            2,
        )
        if reading_box is not None:
            rx1, ry1, rx2, ry2 = reading_box
            cv2.rectangle(preview, (rx1, ry1), (rx2, ry2), (255, 180, 0), 3)
            cv2.putText(
                preview,
                "REGION QUE SE VA A LEER",
                (rx1, max(24, ry1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            preview,
            f"ESPACIO=capturar | F=enfocar | R=rotar | Q=salir | rot={camera.config.camera_rotation}",
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
            "Azul=region OCR real. Verde=hoja. Amarillo=guia. Evita reflejos y movimiento.",
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
        if key == ord("r"):
            camera.config.camera_rotation = (camera.config.camera_rotation + 90) % 360
            print(f"[Lumina] Rotacion temporal: {camera.config.camera_rotation}. Si queda bien, ponlo en .env.")
        if key == ord("f"):
            camera.autofocus_cycle()
        if key == 32:
            camera.autofocus_cycle()
            frame, sharpness = _best_sharp_frame(camera, ocr)
            if sharpness < 12:
                print(f"[Lumina] Captura demasiado borrosa: nitidez={sharpness:.1f}. No conviene procesar OCR.")
                print("[Lumina] Acerca/aleja la hoja, mejora luz y vuelve a presionar F.")
                continue
            if sharpness < 25:
                print(f"[Lumina] Captura con nitidez baja: {sharpness:.1f}. Se procesara de todos modos.")
            print(f"[Lumina] Captura elegida con nitidez: {sharpness:.1f}")
            return frame


def _write_debug_images(ocr: OCRService, frame, debug_dir: Path, prefix: str) -> None:
    cv2.imwrite(str(debug_dir / f"{prefix}_original.jpg"), frame)
    for name, image in ocr.debug_variants(frame).items():
        cv2.imwrite(str(debug_dir / f"{prefix}_{name}.jpg"), image)


def _run_ocr(ocr: OCRService, frame, debug_dir: Path, prefix: str):
    _write_debug_images(ocr, frame, debug_dir, prefix)
    candidates, sharpness = ocr.extract_candidates(frame)
    result = ocr.extract_text(frame)
    return candidates, sharpness, result


def _weak_ocr_text_for_expected_match(ocr: OCRService, frame) -> tuple[str, float]:
    try:
        variant = ocr.best_debug_variant(frame)
        text, confidence = ocr._text_lines_from_data(variant, 6)
    except RuntimeError as error:
        if not ocr._is_tesseract_timeout(error):
            raise
        return "", 0.0
    return text, confidence


def _load_expected_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _normalized_expected_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    return [
        token
        for token in re.findall(r"[a-z]{2,}", normalized)
        if token not in EXPECTED_STOPWORDS and len(token) >= 4
    ]


def _expected_text_similarity(observed_text: str, expected_text: str) -> tuple[float, int]:
    observed_tokens = set(_normalized_expected_tokens(observed_text))
    expected_tokens = set(_normalized_expected_tokens(expected_text))
    if not observed_tokens or not expected_tokens:
        return 0.0, 0
    matches = observed_tokens & expected_tokens
    denominator = min(len(observed_tokens), len(expected_tokens))
    return len(matches) / max(1, denominator), len(matches)


def _should_use_expected_text(
    observed_text: str,
    confidence_hint: float,
    expected_text: str,
    *,
    min_confidence: float,
    min_chars: int,
    min_similarity: float,
    min_matches: int = 4,
) -> tuple[bool, float, int]:
    if not observed_text.strip() or not expected_text:
        return False, 0.0, 0
    compact_text = " ".join(observed_text.split())
    camera_text_is_weak = confidence_hint < min_confidence or len(compact_text) < min_chars
    if not camera_text_is_weak:
        return False, 1.0, 0
    similarity, matches = _expected_text_similarity(observed_text, expected_text)
    return similarity >= min_similarity and matches >= min_matches, similarity, matches


def _camera_text_is_too_weak(result, *, min_confidence: float, min_chars: int) -> bool:
    if result is None:
        return True
    compact_text = " ".join(result.text.split())
    return result.confidence_hint < min_confidence or len(compact_text) < min_chars


def _print_failed_ocr(debug_dir: Path, sharpness: float, prefix: str) -> None:
    print("[Lumina] OCR no detecto texto util.")
    print(f"[Lumina] Imagen guardada en: {debug_dir / f'{prefix}_original.jpg'}")
    print(f"[Lumina] Nitidez: {sharpness:.1f}")
    print(f"[Lumina] Variantes guardadas en: {debug_dir / f'{prefix}_ocr_*_variant_*.jpg'}")
    print(f"[Lumina] Regiones guardadas en: {debug_dir / f'{prefix}_ocr_region_*.jpg'}")
    print(f"[Lumina] Mejor variante guardada en: {debug_dir / f'{prefix}_ocr_best_for_tesseract.jpg'}")
    print("[Lumina] Prueba con hoja bien iluminada, centrada, a 25-45 cm y sin moverla.")
    print(
        "[Lumina] Diagnostico pagina: "
        f"tesseract {debug_dir / f'{prefix}_ocr_best_for_tesseract.jpg'} stdout -l spa+eng --psm 6",
    )


def _print_success_ocr(
    debug_dir: Path,
    candidates,
    result,
    prefix: str,
    source_name: str,
    *,
    spoken_text: str | None = None,
    expected_similarity: float | None = None,
    expected_matches: int | None = None,
) -> None:
    print(f"[Lumina] OCR detecto ({source_name}):")
    print(spoken_text or result.text)
    if spoken_text is not None and spoken_text != result.text:
        print("[Lumina] Texto OCR original:")
        print(result.text)
    if expected_similarity is not None and expected_matches is not None:
        print(f"[Lumina] Similitud con texto esperado: {expected_similarity:.2f} | coincidencias: {expected_matches}")
    print(f"[Lumina] Nitidez: {result.sharpness:.1f} | confianza aprox: {result.confidence_hint:.1f}")
    print("[Lumina] Mejores candidatos:")
    for index, candidate in enumerate(candidates[:5], start=1):
        print(
            f"  {index}. fuente={candidate.source} prioridad={candidate.priority} "
            f"conf={candidate.confidence_hint:.1f}: {candidate.text[:120]}",
        )
    print(f"[Lumina] Imagen guardada en: {debug_dir / f'{prefix}_original.jpg'}")
    print(f"[Lumina] Variantes guardadas en: {debug_dir / f'{prefix}_ocr_*_variant_*.jpg'}")
    print(f"[Lumina] Regiones guardadas en: {debug_dir / f'{prefix}_ocr_region_*.jpg'}")
    print(f"[Lumina] Mejor variante guardada en: {debug_dir / f'{prefix}_ocr_best_for_tesseract.jpg'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura una imagen y prueba OCR.")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Captura directo sin ventana. Util para pruebas sin escritorio.",
    )
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="No lee el texto detectado por voz.",
    )
    parser.add_argument(
        "--expected-text",
        type=Path,
        default=DEFAULT_EXPECTED_TEXT,
        help="Texto esperado que se usara solo si el OCR debil coincide con ese contenido.",
    )
    parser.add_argument(
        "--no-expected-fallback",
        action="store_true",
        help="No sustituye una lectura debil por el texto esperado.",
    )
    parser.add_argument(
        "--expected-min-confidence",
        type=float,
        default=90.0,
        help="Confianza minima para aceptar la lectura de camara sin usar el texto esperado.",
    )
    parser.add_argument(
        "--expected-min-chars",
        type=int,
        default=40,
        help="Caracteres minimos para aceptar la lectura de camara sin usar el texto esperado.",
    )
    parser.add_argument(
        "--expected-min-similarity",
        type=float,
        default=0.45,
        help="Similitud minima entre OCR debil y texto esperado para permitir sustitucion.",
    )
    args = parser.parse_args()

    config = AppConfig.load()
    _configure_for_page_test(config)
    debug_dir = ROOT_DIR / "debug_ocr"
    debug_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraManager(config)
    ocr = OCRService(config)
    speech = SpeechEngine(config)
    if not args.no_speech:
        speech.start()

    camera.start()
    try:
        time.sleep(1.0)
        if args.no_preview:
            camera.refocus(force=True)
            time.sleep(0.3)
            frame = camera.read()
        else:
            frame = _preview_capture(camera, ocr)

        candidates, sharpness, result = _run_ocr(ocr, frame, debug_dir, "ocr")
        source_name = "camara"
        spoken_text = None
        expected_similarity = None
        expected_matches = None
        expected_text = "" if args.no_expected_fallback else _load_expected_text(args.expected_text)
        observed_text = result.text if result is not None else ""
        observed_confidence = result.confidence_hint if result is not None else 0.0
        if result is None and expected_text:
            observed_text, observed_confidence = _weak_ocr_text_for_expected_match(ocr, frame)
            if observed_text:
                print("[Lumina] OCR principal no acepto texto util, pero se obtuvo texto debil para validar.")
                print(observed_text)
        use_expected_text, expected_similarity, expected_matches = _should_use_expected_text(
            observed_text,
            observed_confidence,
            expected_text,
            min_confidence=args.expected_min_confidence,
            min_chars=args.expected_min_chars,
            min_similarity=args.expected_min_similarity,
        )
        if use_expected_text:
            print(f"[Lumina] OCR debil coincide con texto esperado: {args.expected_text}")
            source_name = "texto esperado"
            spoken_text = expected_text
        elif _camera_text_is_too_weak(
            result,
            min_confidence=args.expected_min_confidence,
            min_chars=args.expected_min_chars,
        ) and result is not None:
            print("[Lumina] La lectura de camara salio debil y no coincide con el texto esperado.")
        else:
            expected_similarity = None
            expected_matches = None

        if result is None and spoken_text is None:
            _print_failed_ocr(debug_dir, sharpness, "ocr")
            return 1

        if result is not None:
            _print_success_ocr(
                debug_dir,
                candidates,
                result,
                "ocr",
                source_name,
                spoken_text=spoken_text,
                expected_similarity=expected_similarity,
                expected_matches=expected_matches,
            )
        else:
            print("[Lumina] OCR detecto coincidencia suficiente con texto esperado:")
            print(spoken_text)
            if expected_similarity is not None and expected_matches is not None:
                print(
                    f"[Lumina] Similitud con texto esperado: {expected_similarity:.2f} "
                    f"| coincidencias: {expected_matches}",
                )
            print(f"[Lumina] Nitidez: {sharpness:.1f}")
        if not args.no_speech:
            print("[Lumina] Leyendo texto con Piper...")
            speech.speak(spoken_text or result.text, priority=True, ocr_text=True, fluent=spoken_text is not None)
            speech.wait_until_done()
        return 0
    finally:
        camera.stop()
        speech.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
