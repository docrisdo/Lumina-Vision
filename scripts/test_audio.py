from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.config import AppConfig
from lumina_vision.speech import SpeechEngine


def main() -> int:
    config = AppConfig.load()
    speech = SpeechEngine(config)
    speech.start()
    speech.speak("Prueba de voz de Lumina Vision.")
    speech._queue.join()
    speech.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
