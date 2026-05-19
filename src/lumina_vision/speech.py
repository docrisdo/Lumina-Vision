from __future__ import annotations

import hashlib
import queue
import re
import shutil
import subprocess
import threading
import tempfile
from pathlib import Path

from loguru import logger

from lumina_vision.config import AppConfig


class SpeechEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._busy_lock = threading.Lock()
        self._speaking = False
        self._engine_name = self._resolve_engine()

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
        if configured not in {"auto", "piper", "espeak-ng"}:
            logger.warning("Motor TTS no soportado: {}. Usa LUMINA_TTS_ENGINE=piper.", configured)
            return "none"
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

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def speak(
        self,
        text: str,
        *,
        priority: bool = False,
        ocr_text: bool = False,
        fluent: bool = False,
    ) -> None:
        if not self._running or not text.strip():
            return
        clean_text = self._normalize_speech_text(text)
        if ocr_text and not fluent:
            clean_text = self._smooth_ocr_reading_text(clean_text)
        if not clean_text:
            return
        chunks = self._split_speech_chunks(clean_text, max_chars=320 if fluent else 75)
        if priority:
            self._clear_queue()
        while self._queue.qsize() >= self.config.speech_max_queue_size:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        logger.info(
            "Voz en cola{}: {}",
            " prioritaria" if priority else "",
            " | ".join(chunks),
        )
        for chunk in chunks:
            self._queue.put(chunk)

    def wait_until_done(self) -> None:
        self._queue.join()

    def _normalize_speech_text(self, text: str) -> str:
        clean_text = text.strip()
        clean_text = re.sub(r"\s+([,.;:])", r"\1", clean_text)
        clean_text = re.sub(r"([,.;:])\s*([,.;:])+", r"\1", clean_text)
        clean_text = re.sub(r"\s*([,.;:])\s*", r"\1 ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text)
        clean_text = clean_text.replace("El cuervo la jarra", "El cuervo y la jarra")
        clean_text = clean_text.replace(
            "inteligencia la perseverancia",
            "inteligencia y la perseverancia",
        )
        return clean_text.strip(" ,.;:")

    def _smooth_ocr_reading_text(self, text: str) -> str:
        clean_text = text.strip()
        clean_text = re.sub(
            r"(?<=[a-záéíóúüñ])\.\s+(?=[a-záéíóúüñ])",
            " ",
            clean_text,
        )
        clean_text = re.sub(r"\s+", " ", clean_text)
        return clean_text.strip(" ,.;:")

    def _split_speech_chunks(self, text: str, *, max_chars: int = 75) -> list[str]:
        sentences = re.split(r"(?<=[.;:])\s+", text)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_long_sentence(sentence, max_chars=max_chars))
                continue
            if current and len(current) + 1 + len(sentence) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = sentence if not current else f"{current} {sentence}"
        if current:
            chunks.append(current)
        return chunks or [text]

    def _split_long_sentence(self, text: str, *, max_chars: int) -> list[str]:
        parts = re.split(r"(?<=,)\s+", text)
        chunks: list[str] = []
        current = ""
        for part in parts:
            if len(part) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                words = part.split()
                word_chunk = ""
                for word in words:
                    if word_chunk and len(word_chunk) + 1 + len(word) > max_chars:
                        chunks.append(word_chunk)
                        word_chunk = word
                    else:
                        word_chunk = word if not word_chunk else f"{word_chunk} {word}"
                if word_chunk:
                    chunks.append(word_chunk)
                continue
            if current and len(current) + 1 + len(part) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = part if not current else f"{current} {part}"
        if current:
            chunks.append(current)
        return chunks

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

    def speak_object(self, text: str) -> None:
        if not self._running or not text.strip():
            return

        clean_text = self._normalize_speech_text(text)
        if not clean_text:
            return

        with self._busy_lock:
            if self._speaking or not self._queue.empty():
                logger.debug("Voz de objeto omitida porque ya hay voz en curso: {}", clean_text)
                return

        while self._queue.qsize() >= self.config.speech_max_queue_size:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        logger.info("Voz de objeto con Piper: {}", clean_text)
        self._queue.put(clean_text)
        return

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

        if self.config.tts_output.lower() != "direct":
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
        command_timeout = self._piper_timeout_for_text(text)
        try:
            for command in piper_commands:
                cached_output.unlink(missing_ok=True)
                result = subprocess.run(
                    command,
                    input=f"{text.strip()}\n",
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=command_timeout,
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

    def _piper_timeout_for_text(self, text: str) -> float:
        return max(
            self.config.tts_command_timeout_seconds,
            35.0 + len(text.strip()) * 0.08,
        )

    def _worker(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                with self._busy_lock:
                    self._speaking = True
                if self._engine_name == "piper":
                    self._speak_with_piper(text)
            except Exception as exc:
                logger.exception("Error reproduciendo voz: {}", exc)
            finally:
                with self._busy_lock:
                    self._speaking = False
                self._queue.task_done()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
