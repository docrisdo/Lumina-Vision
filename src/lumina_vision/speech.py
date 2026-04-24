from __future__ import annotations

import queue
import shutil
import subprocess
import threading

from loguru import logger

from lumina_vision.config import AppConfig

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class SpeechEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._engine_name = self._resolve_engine()
        self._pyttsx3_engine = None

        if self._engine_name == "pyttsx3" and pyttsx3 is not None:
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", self.config.speech_rate)
            self._pyttsx3_engine.setProperty("volume", self.config.speech_volume)
            self._configure_spanish_voice()

    def _resolve_engine(self) -> str:
        configured = self.config.tts_engine.lower()
        if configured in {"espeak-ng", "pyttsx3"}:
            return configured
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if pyttsx3 is not None:
            return "pyttsx3"
        return "none"

    def start(self) -> None:
        if self._engine_name == "none":
            logger.warning("No se encontro motor TTS disponible.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Motor TTS iniciado con {}.", self._engine_name)

    def _configure_spanish_voice(self) -> None:
        if self._pyttsx3_engine is None:
            return

        for voice in self._pyttsx3_engine.getProperty("voices"):
            voice_id = getattr(voice, "id", "").lower()
            voice_name = getattr(voice, "name", "").lower()
            languages = " ".join(str(item).lower() for item in getattr(voice, "languages", []))
            if (
                "spanish" in voice_name
                or "es_" in voice_id
                or "es-" in voice_id
                or "mex" in voice_name
                or "spanish" in languages
                or "es" in languages
            ):
                self._pyttsx3_engine.setProperty("voice", voice.id)
                logger.info("Voz en espanol seleccionada: {}", getattr(voice, "name", voice.id))
                return

        logger.warning("No se encontro una voz explicita en espanol para pyttsx3.")

    def speak(self, text: str) -> None:
        if not self._running or not text.strip():
            return
        self._queue.put(text.strip())

    def _worker(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if self._engine_name == "espeak-ng":
                    subprocess.run(
                        ["espeak-ng", "-v", "es", "-s", str(self.config.speech_rate), text],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif self._engine_name == "pyttsx3" and self._pyttsx3_engine is not None:
                    self._pyttsx3_engine.say(text)
                    self._pyttsx3_engine.runAndWait()
            except Exception as exc:
                logger.exception("Error reproduciendo voz: {}", exc)
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
