#!/usr/bin/env bash
set -euo pipefail

echo "[Lumina] Procesos de camara activos:"
pgrep -af "python|rpicam|libcamera" || true

echo
echo "[Lumina] Estado de energia/throttling:"
vcgencmd get_throttled || true

echo
echo "[Lumina] Camaras detectadas:"
rpicam-hello --list-cameras || true

echo
echo "[Lumina] Prueba minima sin preview:"
rpicam-still -o /tmp/lumina_camera_test.jpg --width 768 --height 432 --timeout 2000 --nopreview
echo "[Lumina] Captura guardada en /tmp/lumina_camera_test.jpg"
