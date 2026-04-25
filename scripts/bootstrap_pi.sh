#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[Lumina] Actualizando paquetes del sistema..."
sudo apt update
sudo apt install -y \
  python3-picamera2 \
  python3-libcamera \
  python3-venv \
  tesseract-ocr \
  tesseract-ocr-spa \
  espeak-ng \
  pulseaudio-utils \
  libespeak1 \
  ffmpeg

echo "[Lumina] Preparando entorno virtual..."
cd "$PROJECT_DIR"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "[Lumina] Archivo .env creado desde .env.example"
fi

echo "[Lumina] Bootstrap completado."
echo "[Lumina] Siguiente paso:"
echo "  1. Coloca tu modelo TFLite en models/efficientdet_lite0.tflite"
echo "  2. Ajusta .env si lo necesitas"
echo "  3. Ejecuta: bash scripts/run_pi.sh"
