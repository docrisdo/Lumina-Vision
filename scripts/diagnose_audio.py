from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path
import math
import struct


ROOT_DIR = Path(__file__).resolve().parent.parent


def _run(command: list[str], *, timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    print(f"\n$ {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    print(f"[exit={result.returncode}]")
    return result


def _make_test_wav(path: Path) -> None:
    sample_rate = 48000
    duration = 0.8
    frequency = 880.0
    frames = int(sample_rate * duration)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(frames):
            value = int(12000 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            wav.writeframes(struct.pack("<h", value))


def main() -> int:
    test_wav = Path("/tmp/lumina_audio_test.wav")
    _make_test_wav(test_wav)

    print("[Lumina] Diagnostico de audio. Debes escuchar un beep corto en alguna prueba.")
    for command in (
        ["pactl", "info"],
        ["pactl", "list", "short", "sinks"],
        ["pactl", "get-default-sink"],
    ):
        if shutil.which(command[0]):
            _run(command)

    if shutil.which("pactl"):
        _run(["pactl", "suspend-sink", "@DEFAULT_SINK@", "false"])
        _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "false"])

    players = []
    if shutil.which("pw-play"):
        players.append(["pw-play", str(test_wav)])
    if shutil.which("paplay"):
        players.append(["paplay", str(test_wav)])
    if shutil.which("aplay"):
        players.append(["aplay", "-q", str(test_wav)])

    if not players:
        print("[Lumina] No encontre pw-play, paplay ni aplay.")
        return 1

    ok = False
    for player in players:
        result = _run(player)
        ok = ok or result.returncode == 0

    if ok:
        print("\n[Lumina] Al menos un reproductor funciono. Si no escuchaste nada, revisa volumen/perfil Bluetooth.")
        return 0

    print("\n[Lumina] Ningun reproductor pudo enviar audio. Revisa el perfil Bluetooth o reconecta la bocina.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
