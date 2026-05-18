#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
MODEL_PATH="$MODEL_DIR/frozen_east_text_detection.pb"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$MODEL_DIR"

if [[ -s "$MODEL_PATH" ]]; then
  echo "[Lumina] Modelo EAST ya existe: $MODEL_PATH"
  exit 0
fi

echo "[Lumina] Descargando modelo EAST de OpenCV..."
wget -O "$TMP_DIR/east.tar.gz" "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1"
tar -xzf "$TMP_DIR/east.tar.gz" -C "$TMP_DIR"

FOUND_MODEL="$(find "$TMP_DIR" -name 'frozen_east_text_detection.pb' -type f | head -n 1)"
if [[ -z "$FOUND_MODEL" ]]; then
  echo "[Lumina] No se encontro frozen_east_text_detection.pb dentro del archivo descargado." >&2
  exit 1
fi

cp "$FOUND_MODEL" "$MODEL_PATH"
echo "[Lumina] Modelo EAST instalado en: $MODEL_PATH"
echo "[Lumina] Para activarlo, usa LUMINA_OCR_EAST_ENABLED=true en .env"
