from __future__ import annotations

import sys
import time
from argparse import ArgumentParser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.config import AppConfig
from lumina_vision.speech import SpeechEngine


def main() -> int:
    parser = ArgumentParser(description="Prueba de audio de Lumina Vision.")
    parser.add_argument(
        "--include-alert",
        action="store_true",
        help="Tambien prueba la alerta urgente del ultrasonico con espeak-ng.",
    )
    args = parser.parse_args()

    config = AppConfig.load()
    speech = SpeechEngine(config)
    speech.start()
    print("[Lumina] Probando voz normal con Piper.")
    speech.speak("Prueba de voz de Lumina Vision.")
    speech._queue.join()

    if args.include_alert:
        print("[Lumina] Probando alerta ultrasonica con espeak-ng.")
        speech.speak_alert("Cuidado. Hay un objeto a diez centimetros.")
        time.sleep(2.0)

    speech.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
