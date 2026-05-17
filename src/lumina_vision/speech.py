from __future__ import annotations

import hashlib
import queue
import shutil
import subprocess
import threading
import tempfile
from pathlib import Path

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
        if configured == "piper" and self._piper_tts_available():
            return "piper"
        if configured == "piper":
            logger.warning("Piper TTS no esta disponible. La voz normal no usara espeak-ng.")
            return "none"
        if configured == "espeak-ng":
            logger.warning("espeak-ng queda reservado para la alerta ultrasonica; voz normal desactivada.")
            return "none"
        if configured == "pyttsx3":
            return configured
        if self._piper_tts_available():
            return "piper"
        logger.warning("Piper no esta disponible; voz normal desactivada.")
        return "none"

    def _piper_tts_available(self) -> bool:
        piper_path = self._resolve_piper_command()
        if not piper_path or not self.config.piper_model_path.exists():
            return False

        return True

    def _resolve_piper_command(self) -> str | None:
        candidates = [
            Path(".venv/bin/piper"),
            Path("venv/bin/piper"),
            shutil.which("piper-tts"),
            shutil.which("piper"),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            piper_path = str(candidate)
            if not Path(piper_path).exists() and shutil.which(piper_path) is None:
                continue
            if self._is_piper_tts_command(piper_path):
                return piper_path
        return None

    def _is_piper_tts_command(self, piper_path: str) -> bool:
        try:
            help_result = subprocess.run(
                [piper_path, "--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config.tts_command_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Piper no respondio a tiempo al validar el comando.")
            return False
        help_text = f"{help_result.stdout}\n{help_result.stderr}"
        if "--model" not in help_text and "-m" not in help_text:
            logger.warning(
                "El comando piper encontrado no parece ser Piper TTS: {}",
                piper_path,
            )
            return False
        return True

    def start(self) -> None:
        if self._engine_name == "none":
            logger.warning("No se encontro motor TTS disponible.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Motor TTS iniciado con {}.", self._engine_name)

        if self.config.tts_startup_test:
            self.speak("Sistema de voz listo.")

    def warmup(self, phrases: list[str]) -> None:
        if self._engine_name != "piper":
            return
        for phrase in phrases:
            try:
                self._ensure_piper_audio(phrase.strip())
            except Exception as exc:
                logger.warning("No se pudo precalentar frase TTS '{}': {}", phrase, exc)

    def warmup_async(self, phrases: list[str]) -> None:
        thread = threading.Thread(
            target=self.warmup,
            args=(phrases,),
            name="lumina-tts-warmup",
            daemon=True,
        )
        thread.start()

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

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def speak(self, text: str, *, priority: bool = False) -> None:
        if not self._running or not text.strip():
            return
        clean_text = text.strip()
        if priority:
            self._clear_queue()
        while self._queue.qsize() >= self.config.speech_max_queue_size:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        logger.info("Voz en cola{}: {}", " prioritaria" if priority else "", clean_text)
        self._queue.put(clean_text)

    def speak_alert(self, text: str) -> None:
        if not self._running or not text.strip():
            return

        clean_text = text.strip()
        self._clear_queue()
        if shutil.which("espeak-ng"):
            thread = threading.Thread(
                target=self._speak_with_espeak,
                args=(clean_text,),
                kwargs={"rate": self.config.ultrasonic_speech_rate},
                name="lumina-urgent-tts",
                daemon=True,
            )
            thread.start()
            logger.info("Voz urgente con espeak-ng: {}", clean_text)
            return

        self.speak(clean_text, priority=True)

    def _speak_with_espeak(self, text: str, *, rate: int | None = None) -> None:
        amplitude = max(0, min(200, int(self.config.speech_volume * 200)))
        speech_rate = self.config.speech_rate if rate is None else rate
        base_cmd = [
            "espeak-ng",
            "-v",
            "es",
            "-s",
            str(speech_rate),
            "-a",
            str(amplitude),
        ]

        if self.config.tts_output.lower() != "direct" or rate is not None:
            with tempfile.NamedTemporaryFile(prefix="lumina_espeak_", suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            result = subprocess.run(
                [*base_cmd, "-w", str(temp_path), text],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config.tts_command_timeout_seconds,
            )
            if result.returncode != 0:
                logger.warning("espeak-ng fallo: {}", result.stderr.strip())
                temp_path.unlink(missing_ok=True)
                return
            try:
                self._play_audio_file(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
            return

        result = subprocess.run(
            [*base_cmd, text],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.config.tts_command_timeout_seconds,
        )
        if result.returncode != 0:
            logger.warning("espeak-ng fallo: {}", result.stderr.strip())

    def _speak_with_piper(self, text: str) -> None:
        cached_output = self._ensure_piper_audio(text)
        if cached_output is None:
            logger.warning("Piper no genero audio. No se usara espeak-ng para voz normal.")
            return

        self._play_audio_file(cached_output)

    def _wake_audio_sink(self) -> None:
        if not shutil.which("pactl"):
            return
        commands = [
            ["pactl", "suspend-sink", "@DEFAULT_SINK@", "false"],
            ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "false"],
        ]
        for command in commands:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )

    def _audio_players(self) -> list[list[str]]:
        players: list[list[str]] = []
        if shutil.which("pw-play"):
            players.append(["pw-play"])
        if shutil.which("paplay"):
            players.append(["paplay"])
        if shutil.which("aplay"):
            players.append(["aplay", "-q"])
        return players

    def _play_audio_file(self, audio_path: Path) -> None:
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            logger.warning("No se puede reproducir audio: archivo inexistente o vacio: {}", audio_path)
            return

        self._wake_audio_sink()
        last_error = ""
        for player_cmd in self._audio_players():
            play_result = subprocess.run(
                [*player_cmd, str(audio_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config.tts_command_timeout_seconds,
            )
            if play_result.returncode == 0:
                return
            last_error = play_result.stderr.strip()
            logger.warning("{} fallo: {}", player_cmd[0], last_error)

        logger.warning("No se pudo reproducir audio con ningun reproductor. Ultimo error: {}", last_error)

    def _ensure_piper_audio(self, text: str):
        if not text:
            return None
        if not self.config.piper_model_path.exists():
            logger.warning("No se encontro el modelo Piper: {}", self.config.piper_model_path)
            return None
        piper_path = self._resolve_piper_command()
        if not piper_path:
            logger.warning("No se encontro un comando Piper TTS valido.")
            return None

        self.config.piper_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        cached_output = self.config.piper_cache_dir / f"{cache_key}.wav"

        if cached_output.exists():
            if cached_output.stat().st_size > 44:
                return cached_output
            logger.warning("Cache Piper invalido, se regenerara: {}", cached_output)
            cached_output.unlink(missing_ok=True)

        piper_commands = [
            [
                piper_path,
                "--model",
                str(self.config.piper_model_path),
                "--output_file",
                str(cached_output),
            ],
            [
                piper_path,
                "--model",
                str(self.config.piper_model_path),
                "--output-file",
                str(cached_output),
            ],
            [
                piper_path,
                "-m",
                str(self.config.piper_model_path),
                "-f",
                str(cached_output),
            ],
        ]
        last_stdout = ""
        last_stderr = ""
        try:
            for command in piper_commands:
                cached_output.unlink(missing_ok=True)
                result = subprocess.run(
                    command,
                    input=f"{text.strip()}\n",
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.config.tts_command_timeout_seconds,
                )
                last_stdout = result.stdout.strip()
                last_stderr = result.stderr.strip()
                if cached_output.exists() and cached_output.stat().st_size > 44:
                    return cached_output
                logger.warning(
                    "Piper intento fallo: {} | exit={} | stdout='{}' stderr='{}'",
                    " ".join(command),
                    result.returncode,
                    last_stdout,
                    last_stderr,
                )
        except subprocess.TimeoutExpired:
            logger.warning("Piper tardo demasiado generando audio para: {}", text)
            cached_output.unlink(missing_ok=True)
            return None

        logger.warning(
            "Piper no genero audio valido para '{}'. Ultimo stdout='{}' stderr='{}'",
            text,
            last_stdout,
            last_stderr,
        )
        cached_output.unlink(missing_ok=True)
        return None

    def _worker(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if self._engine_name == "piper":
                    self._speak_with_piper(text)
                elif self._engine_name == "espeak-ng":
                    self._speak_with_espeak(text)
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
