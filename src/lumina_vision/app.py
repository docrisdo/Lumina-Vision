from __future__ import annotations

import sys

from loguru import logger

from lumina_vision.config import AppConfig
from lumina_vision.pipeline import LuminaPipeline


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def main() -> int:
    config = AppConfig.load()
    configure_logging(config.log_level)

    logger.info("Iniciando Lumina Vision.")
    logger.info(
        "Configuracion: camara={}x{}, obj={}, ocr={}, tts={}",
        config.camera_width,
        config.camera_height,
        config.enable_object_detection,
        config.enable_ocr,
        config.enable_tts,
    )

    pipeline = LuminaPipeline(config)
    return pipeline.run()
