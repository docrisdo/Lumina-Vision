from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.config import AppConfig


def main() -> int:
    config = AppConfig.load()
    errors: list[str] = []
    warnings: list[str] = []

    if config.enable_object_detection and not config.detection_model_path.exists():
        errors.append(
            f"Falta el modelo de deteccion: {config.detection_model_path}",
        )

    if not config.labels_path.exists():
        errors.append(f"Falta el archivo de etiquetas: {config.labels_path}")

    if shutil.which("tesseract") is None:
        warnings.append("Tesseract no esta en PATH. OCR puede fallar.")

    if config.enable_tts and shutil.which("espeak-ng") is None:
        warnings.append("espeak-ng no esta en PATH. Se intentara usar pyttsx3.")

    if config.tts_output.lower() == "aplay" and shutil.which("aplay") is None:
        warnings.append("LUMINA_TTS_OUTPUT=aplay pero aplay no esta instalado.")

    if config.show_preview and os.environ.get("DISPLAY") is None:
        warnings.append(
            "LUMINA_SHOW_PREVIEW=true pero no hay DISPLAY. Considera usar false en modo headless.",
        )

    if errors:
        print("[Lumina] Validacion fallida:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print("[Lumina] Validacion correcta.")
    for item in warnings:
        print(f"[Lumina] Aviso: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
