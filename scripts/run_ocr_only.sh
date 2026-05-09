#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d .venv ]]; then
  echo "[Lumina] No existe .venv. Ejecuta primero: bash scripts/bootstrap_pi.sh"
  exit 1
fi

source .venv/bin/activate
export LUMINA_ENABLE_OBJECT_DETECTION=false
export LUMINA_ENABLE_OCR=true
export LUMINA_ENABLE_TTS=true
export LUMINA_SHOW_PREVIEW=false
export LUMINA_WEARABLE_MODE=true
export LUMINA_CAMERA_WIDTH="${LUMINA_CAMERA_WIDTH:-768}"
export LUMINA_CAMERA_HEIGHT="${LUMINA_CAMERA_HEIGHT:-432}"
export LUMINA_CAMERA_BUFFER_COUNT="${LUMINA_CAMERA_BUFFER_COUNT:-2}"
export LUMINA_CAMERA_COLOR_MODE="${LUMINA_CAMERA_COLOR_MODE:-none}"
export LUMINA_TTS_WARMUP=false
python scripts/validate_runtime.py
python main.py
