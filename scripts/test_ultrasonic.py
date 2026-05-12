from __future__ import annotations

import os
import signal
import sys
import time


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return int(raw_value) if raw_value is not None else default


def main() -> int:
    try:
        from gpiozero import DistanceSensor
    except ImportError:
        print("[Lumina] Falta gpiozero.")
        print("[Lumina] Instala en Raspberry con: sudo apt install -y python3-gpiozero")
        return 1

    trigger_pin = _env_int("LUMINA_ULTRASONIC_TRIGGER_PIN", 23)
    echo_pin = _env_int("LUMINA_ULTRASONIC_ECHO_PIN", 24)
    max_distance_m = float(os.getenv("LUMINA_ULTRASONIC_MAX_DISTANCE_M", "3.0"))

    print("[Lumina] Prueba HC-SR04")
    print(f"[Lumina] Usando numeracion BCM: TRIG=GPIO{trigger_pin}, ECHO=GPIO{echo_pin}")
    print("[Lumina] Deten con Ctrl+C.")
    print("[Lumina] IMPORTANTE: ECHO debe llegar a la Raspberry como 3.3V, no 5V.")

    stop = False

    def _stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    sensor = DistanceSensor(
        echo=echo_pin,
        trigger=trigger_pin,
        max_distance=max_distance_m,
        queue_len=5,
        threshold_distance=0.5,
    )

    try:
        while not stop:
            distance_cm = sensor.distance * 100.0
            if distance_cm <= 0 or distance_cm >= max_distance_m * 100:
                print("[Lumina] Fuera de rango o sin eco")
            else:
                print(f"[Lumina] Distancia: {distance_cm:6.1f} cm")
            time.sleep(0.35)
    finally:
        sensor.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
