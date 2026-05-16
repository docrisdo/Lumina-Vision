from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.config import AppConfig


def _run(command: list[str], *, input_text: str | None = None, timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _play(audio_path: Path, timeout: float) -> bool:
    players: list[list[str]] = []
    if shutil.which("pw-play"):
        players.append(["pw-play"])
    if shutil.which("paplay"):
        players.append(["paplay"])
    if shutil.which("aplay"):
        players.append(["aplay", "-q"])

    for player in players:
        result = _run([*player, str(audio_path)], timeout=timeout)
        if result.returncode == 0:
            print(f"[Lumina] Reproducido con {player[0]}.")
            return True
        print(f"[Lumina] {player[0]} fallo: {result.stderr.strip()}")
    return False


def main() -> int:
    config = AppConfig.load()
    piper_path = shutil.which("piper")
    model_path = config.piper_model_path
    model_config = Path(f"{model_path}.json")
    output_path = Path("/tmp/lumina_piper_direct.wav")

    print(f"[Lumina] piper: {piper_path or 'NO encontrado'}")
    print(f"[Lumina] modelo: {model_path} exists={model_path.exists()}")
    print(f"[Lumina] config modelo: {model_config} exists={model_config.exists()}")
    if not piper_path or not model_path.exists():
        return 1

    try:
        help_result = _run([piper_path, "--help"], timeout=config.tts_command_timeout_seconds)
        help_text = f"{help_result.stdout}\n{help_result.stderr}".strip()
        print("[Lumina] piper --help:")
        print("\n".join(help_text.splitlines()[:25]))
    except subprocess.TimeoutExpired:
        print("[Lumina] piper --help no respondio.")

    commands = [
        [piper_path, "--model", str(model_path), "--output_file", str(output_path)],
        [piper_path, "--model", str(model_path), "--output-file", str(output_path)],
        [piper_path, "-m", str(model_path), "-f", str(output_path)],
    ]

    text = "Prueba de voz de Lumina Vision.\n"
    for command in commands:
        output_path.unlink(missing_ok=True)
        print(f"[Lumina] Probando: {' '.join(command)}")
        try:
            result = _run(command, input_text=text, timeout=config.tts_command_timeout_seconds)
        except subprocess.TimeoutExpired:
            print("[Lumina] Timeout generando audio.")
            continue

        size = output_path.stat().st_size if output_path.exists() else 0
        print(f"[Lumina] exit={result.returncode} wav_size={size}")
        if result.stdout.strip():
            print(f"[Lumina] stdout: {result.stdout.strip()}")
        if result.stderr.strip():
            print(f"[Lumina] stderr: {result.stderr.strip()}")

        if size > 44:
            if shutil.which("pactl"):
                _run(["pactl", "suspend-sink", "@DEFAULT_SINK@", "false"], timeout=2.0)
                _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "false"], timeout=2.0)
            played = _play(output_path, config.tts_command_timeout_seconds)
            return 0 if played else 2

    print("[Lumina] Piper no genero un WAV valido con ninguna variante.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
