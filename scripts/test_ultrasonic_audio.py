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
            "Cuidado. Hay un objeto a 15 centimetros.",
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
                distance_cm = max(1, int(round(distance or config.ultrasonic_alert_distance_cm)))
                unit = "centimetro" if distance_cm == 1 else "centimetros"
                speech.speak(f"Cuidado. Hay un objeto a {distance_cm} {unit}.", priority=True)
                gate.mark()
            time.sleep(config.ultrasonic_poll_interval_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        ultrasonic.stop()
        speech.stop()


if __name__ == "__main__":
    raise SystemExit(main())
