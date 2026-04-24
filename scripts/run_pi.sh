#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d .venv ]]; then
  echo "[Lumina] No existe .venv. Ejecuta primero: bash scripts/bootstrap_pi.sh"
  exit 1
fi

source .venv/bin/activate
python scripts/validate_runtime.py
python main.py
