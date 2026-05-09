from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return int(raw_value) if raw_value is not None else default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    return float(raw_value) if raw_value is not None else default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(slots=True)
class AppConfig:
    wearable_mode: bool
    camera_width: int
    camera_height: int
    camera_hflip: bool
    camera_vflip: bool
    camera_af_mode: str
    camera_af_range: str
    camera_af_speed: str
    camera_color_mode: str
    camera_refocus_before_ocr: bool
    camera_refocus_interval_seconds: float
    camera_focus_settle_seconds: float
    camera_framerate: float
    preview_max_width: int
    enable_object_detection: bool
    enable_ocr: bool
    enable_tts: bool
    show_preview: bool
    save_debug_frames: bool
    detection_model_path: Path
    labels_path: Path
    detection_score_threshold: float
    detection_max_results: int
    detection_run_every_n_frames: int
    school_mode: bool
    ocr_language: str
    ocr_auto_read: bool
    ocr_run_interval_seconds: float
    ocr_min_text_length: int
    ocr_max_width: int
    ocr_min_sharpness: float
    ocr_stable_reads: int
    ocr_prefer_center_crop: bool
    ocr_suppress_objects_seconds: float
    tts_engine: str
    tts_output: str
    tts_startup_test: bool
    piper_model_path: Path
    piper_output_file: Path
    piper_cache_dir: Path
    speech_rate: int
    speech_volume: float
    speech_cooldown_seconds: float
    speech_object_cooldown_seconds: float
    speech_ocr_cooldown_seconds: float
    speech_repeat_same_object_seconds: float
    speech_max_queue_size: int
    speech_enable_objects: bool
    speech_enable_ocr: bool
    log_level: str

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv()
        wearable_mode = _env_bool("LUMINA_WEARABLE_MODE", True)
        return cls(
            wearable_mode=wearable_mode,
            camera_width=_env_int("LUMINA_CAMERA_WIDTH", 1920),
            camera_height=_env_int("LUMINA_CAMERA_HEIGHT", 1080),
            camera_hflip=_env_bool("LUMINA_CAMERA_HFLIP", False),
            camera_vflip=_env_bool("LUMINA_CAMERA_VFLIP", False),
            camera_af_mode=_env_str("LUMINA_CAMERA_AF_MODE", "continuous"),
            camera_af_range=_env_str("LUMINA_CAMERA_AF_RANGE", "full"),
            camera_af_speed=_env_str("LUMINA_CAMERA_AF_SPEED", "fast"),
            camera_color_mode=_env_str("LUMINA_CAMERA_COLOR_MODE", "rgb_to_bgr"),
            camera_refocus_before_ocr=_env_bool("LUMINA_CAMERA_REFOCUS_BEFORE_OCR", True),
            camera_refocus_interval_seconds=max(
                1.0,
                _env_float("LUMINA_CAMERA_REFOCUS_INTERVAL_SECONDS", 3.0),
            ),
            camera_focus_settle_seconds=max(
                0.0,
                _env_float("LUMINA_CAMERA_FOCUS_SETTLE_SECONDS", 0.7),
            ),
            camera_framerate=max(5.0, _env_float("LUMINA_CAMERA_FRAMERATE", 8.0)),
            preview_max_width=max(320, _env_int("LUMINA_PREVIEW_MAX_WIDTH", 960)),
            enable_object_detection=_env_bool("LUMINA_ENABLE_OBJECT_DETECTION", True),
            enable_ocr=_env_bool("LUMINA_ENABLE_OCR", True),
            enable_tts=_env_bool("LUMINA_ENABLE_TTS", True),
            show_preview=_env_bool("LUMINA_SHOW_PREVIEW", not wearable_mode),
            save_debug_frames=_env_bool("LUMINA_SAVE_DEBUG_FRAMES", False),
            detection_model_path=Path(
                _env_str(
                    "LUMINA_DETECTION_MODEL_PATH",
                    "models/efficientdet_lite0.tflite",
                ),
            ),
            labels_path=Path(_env_str("LUMINA_LABELS_PATH", "models/coco_labels.txt")),
            detection_score_threshold=_env_float("LUMINA_DETECTION_SCORE_THRESHOLD", 0.55),
            detection_max_results=_env_int("LUMINA_DETECTION_MAX_RESULTS", 5),
            detection_run_every_n_frames=max(
                1,
                _env_int("LUMINA_DETECTION_RUN_EVERY_N_FRAMES", 2),
            ),
            school_mode=_env_bool("LUMINA_SCHOOL_MODE", True),
            ocr_language=_env_str("LUMINA_OCR_LANGUAGE", "spa+eng"),
            ocr_auto_read=_env_bool("LUMINA_OCR_AUTO_READ", True),
            ocr_run_interval_seconds=max(
                0.2,
                _env_float("LUMINA_OCR_RUN_INTERVAL_SECONDS", 1.6),
            ),
            ocr_min_text_length=max(1, _env_int("LUMINA_OCR_MIN_TEXT_LENGTH", 4)),
            ocr_max_width=max(320, _env_int("LUMINA_OCR_MAX_WIDTH", 1600)),
            ocr_min_sharpness=max(0.0, _env_float("LUMINA_OCR_MIN_SHARPNESS", 20.0)),
            ocr_stable_reads=max(1, _env_int("LUMINA_OCR_STABLE_READS", 1)),
            ocr_prefer_center_crop=_env_bool("LUMINA_OCR_PREFER_CENTER_CROP", True),
            ocr_suppress_objects_seconds=max(
                0.0,
                _env_float("LUMINA_OCR_SUPPRESS_OBJECTS_SECONDS", 4.0),
            ),
            tts_engine=_env_str("LUMINA_TTS_ENGINE", "auto"),
            tts_output=_env_str("LUMINA_TTS_OUTPUT", "direct"),
            tts_startup_test=_env_bool("LUMINA_TTS_STARTUP_TEST", False),
            piper_model_path=Path(
                _env_str(
                    "LUMINA_PIPER_MODEL_PATH",
                    "models/tts/es_MX-ald-medium.onnx",
                ),
            ),
            piper_output_file=Path(
                _env_str("LUMINA_PIPER_OUTPUT_FILE", "/tmp/lumina_voice.wav"),
            ),
            piper_cache_dir=Path(
                _env_str("LUMINA_PIPER_CACHE_DIR", "/tmp/lumina_piper_cache"),
            ),
            speech_rate=_env_int("LUMINA_SPEECH_RATE", 170),
            speech_volume=_env_float("LUMINA_SPEECH_VOLUME", 1.0),
            speech_cooldown_seconds=max(
                0.5,
                _env_float("LUMINA_SPEECH_COOLDOWN_SECONDS", 4.0),
            ),
            speech_object_cooldown_seconds=max(
                1.0,
                _env_float("LUMINA_SPEECH_OBJECT_COOLDOWN_SECONDS", 2.5),
            ),
            speech_ocr_cooldown_seconds=max(
                1.0,
                _env_float("LUMINA_SPEECH_OCR_COOLDOWN_SECONDS", 1.5),
            ),
            speech_repeat_same_object_seconds=max(
                5.0,
                _env_float("LUMINA_SPEECH_REPEAT_SAME_OBJECT_SECONDS", 20.0),
            ),
            speech_max_queue_size=max(1, _env_int("LUMINA_SPEECH_MAX_QUEUE_SIZE", 1)),
            speech_enable_objects=_env_bool("LUMINA_SPEECH_ENABLE_OBJECTS", True),
            speech_enable_ocr=_env_bool("LUMINA_SPEECH_ENABLE_OCR", True),
            log_level=_env_str("LUMINA_LOG_LEVEL", "INFO"),
        )
