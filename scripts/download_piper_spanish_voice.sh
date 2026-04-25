#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="$PROJECT_DIR/models/tts"

mkdir -p "$VOICE_DIR"

MODEL_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx"
CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json"

echo "[Lumina] Descargando voz Piper en espanol..."
wget -O "$VOICE_DIR/es_MX-ald-medium.onnx" "$MODEL_URL"
wget -O "$VOICE_DIR/es_MX-ald-medium.onnx.json" "$CONFIG_URL"

echo "[Lumina] Voz descargada en $VOICE_DIR"
echo "[Lumina] En .env usa:"
echo "LUMINA_TTS_ENGINE=piper"
echo "LUMINA_PIPER_MODEL_PATH=models/tts/es_MX-ald-medium.onnx"
