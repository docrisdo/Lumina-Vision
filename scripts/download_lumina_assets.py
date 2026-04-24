from __future__ import annotations

import argparse
from pathlib import Path

import requests


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Descarga un modelo y/o labels para Lumina Vision.",
    )
    parser.add_argument("--model-url", help="URL del modelo .tflite")
    parser.add_argument(
        "--model-path",
        default="models/efficientdet_lite0.tflite",
        help="Ruta destino del modelo",
    )
    parser.add_argument("--labels-url", help="URL del archivo de etiquetas")
    parser.add_argument(
        "--labels-path",
        default="models/coco_labels.txt",
        help="Ruta destino del archivo de etiquetas",
    )
    args = parser.parse_args()

    if not args.model_url and not args.labels_url:
        parser.error("Debes indicar al menos --model-url o --labels-url.")

    if args.model_url:
        download_file(args.model_url, Path(args.model_path))
        print(f"Modelo descargado en {args.model_path}")

    if args.labels_url:
        download_file(args.labels_url, Path(args.labels_path))
        print(f"Etiquetas descargadas en {args.labels_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
