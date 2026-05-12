from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.config import AppConfig
from lumina_vision.speech import SpeechEngine
from lumina_vision.ultrasonic import UltrasonicMonitor
from lumina_vision.utils import CooldownGate


def main() -> int:
    config = AppConfig.load()
    config.enable_ultrasonic = True
    config.enable_tts = True

    speech = SpeechEngine(config)
    ultrasonic = UltrasonicMonitor(config)
    gate = CooldownGate(config.ultrasonic_alert_cooldown_seconds)

    speech.start()
    speech.warmup_async(
        [
            "Cuidado. Hay un objeto muy cerca.",
            "Cuidado. Hay un objeto demasiado cerca.",
        ],
    )
    if not ultrasonic.start():
        speech.stop()
        return 1

    print("[Lumina] Prueba de ultrasonico con audio. Deten con Ctrl+C.")
    try:
        while True:
            distance = ultrasonic.latest_distance_cm
            if distance is None:
                print("[Lumina] Sin lectura valida")
            else:
                print(f"[Lumina] Distancia: {distance:6.1f} cm")

            if ultrasonic.close_obstacle_confirmed() and gate.ready():
                speech.speak("Cuidado. Hay un objeto muy cerca.", priority=True)
                gate.mark()
            time.sleep(config.ultrasonic_poll_interval_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        ultrasonic.stop()
        speech.stop()


if __name__ == "__main__":
    raise SystemExit(main())
