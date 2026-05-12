from __future__ import annotations

import threading
import time

from loguru import logger

from lumina_vision.config import AppConfig

try:
    from gpiozero import DistanceSensor
except ImportError:
    DistanceSensor = None


class UltrasonicMonitor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._sensor = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_distance_cm: float | None = None
        self._close_since: float | None = None

    @property
    def latest_distance_cm(self) -> float | None:
        with self._lock:
            return self._latest_distance_cm

    def start(self) -> bool:
        if not self.config.enable_ultrasonic:
            return False
        if DistanceSensor is None:
            logger.warning("Sensor ultrasonico deshabilitado: falta gpiozero.")
            return False

        try:
            self._sensor = DistanceSensor(
                echo=self.config.ultrasonic_echo_pin,
                trigger=self.config.ultrasonic_trigger_pin,
                max_distance=self.config.ultrasonic_max_distance_m,
                queue_len=3,
            )
        except Exception as exc:
            logger.warning("No se pudo iniciar el sensor ultrasonico: {}", exc)
            return False

        self._running = True
        self._thread = threading.Thread(target=self._worker, name="lumina-ultrasonic", daemon=True)
        self._thread.start()
        logger.info(
            "Sensor ultrasonico iniciado. TRIG=GPIO{} ECHO=GPIO{} alerta={}cm",
            self.config.ultrasonic_trigger_pin,
            self.config.ultrasonic_echo_pin,
            self.config.ultrasonic_alert_distance_cm,
        )
        return True

    def _worker(self) -> None:
        while self._running and self._sensor is not None:
            try:
                distance_cm = float(self._sensor.distance * 100.0)
            except Exception as exc:
                logger.debug("No se pudo leer el sensor ultrasonico: {}", exc)
                distance_cm = 0.0

            if 0.0 < distance_cm < self.config.ultrasonic_max_distance_m * 100.0:
                with self._lock:
                    self._latest_distance_cm = distance_cm
            else:
                with self._lock:
                    self._latest_distance_cm = None
                    self._close_since = None

            time.sleep(self.config.ultrasonic_poll_interval_seconds)

    def close_obstacle_confirmed(self) -> bool:
        distance = self.latest_distance_cm
        if distance is None or distance > self.config.ultrasonic_alert_distance_cm:
            self._close_since = None
            return False

        now = time.monotonic()
        if self._close_since is None:
            self._close_since = now
            return False

        return (now - self._close_since) >= 0.35

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sensor is not None:
            self._sensor.close()
            self._sensor = None
            logger.info("Sensor ultrasonico detenido.")
