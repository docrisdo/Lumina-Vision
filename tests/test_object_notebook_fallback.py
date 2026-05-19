from pathlib import Path

import cv2
import numpy as np

from lumina_vision.config import AppConfig
from lumina_vision.detectors.tflite_detector import Detection, ObjectDetector


def _config() -> AppConfig:
    return AppConfig(
        wearable_mode=False,
        camera_width=640,
        camera_height=480,
        camera_hflip=False,
        camera_vflip=False,
        camera_af_mode="continuous",
        camera_af_range="full",
        camera_af_speed="fast",
        camera_lens_position=-1.0,
        camera_color_mode="none",
        camera_rotation=0,
        camera_refocus_before_ocr=True,
        camera_refocus_interval_seconds=3.0,
        camera_focus_settle_seconds=0.5,
        camera_framerate=8.0,
        camera_buffer_count=2,
        preview_max_width=960,
        enable_object_detection=True,
        enable_ocr=True,
        enable_tts=True,
        enable_ultrasonic=False,
        show_preview=False,
        save_debug_frames=False,
        detection_model_path=Path("models/efficientdet_lite0.tflite"),
        labels_path=Path("models/coco_labels.txt"),
        detection_score_threshold=0.55,
        detection_max_results=5,
        detection_run_every_n_frames=2,
        school_mode=True,
        ocr_language="spa+eng",
        ocr_auto_read=True,
        ocr_page_mode=True,
        ocr_run_interval_seconds=2.0,
        ocr_min_text_length=4,
        ocr_max_width=1536,
        ocr_min_sharpness=18.0,
        ocr_stable_reads=2,
        ocr_prefer_center_crop=True,
        ocr_roi_x1=0.22,
        ocr_roi_y1=0.06,
        ocr_roi_x2=0.78,
        ocr_roi_y2=0.94,
        ocr_fast_mode=True,
        ocr_suppress_objects_seconds=4.0,
        tts_engine="auto",
        tts_output="direct",
        tts_startup_test=False,
        tts_warmup=False,
        tts_command_timeout_seconds=15.0,
        piper_model_path=Path("models/tts/es_MX-ald-medium.onnx"),
        piper_output_file=Path("/tmp/lumina_voice.wav"),
        piper_cache_dir=Path("/tmp/lumina_piper_cache"),
        speech_rate=170,
        speech_volume=1.0,
        speech_cooldown_seconds=4.0,
        speech_object_cooldown_seconds=2.5,
        speech_ocr_cooldown_seconds=1.5,
        speech_repeat_same_object_seconds=20.0,
        speech_max_queue_size=1,
        speech_enable_objects=True,
        speech_enable_ocr=True,
        ultrasonic_trigger_pin=23,
        ultrasonic_echo_pin=24,
        ultrasonic_max_distance_m=3.0,
        ultrasonic_alert_distance_cm=15.0,
        ultrasonic_poll_interval_seconds=0.3,
        ultrasonic_confirm_seconds=0.15,
        ultrasonic_alert_cooldown_seconds=3.0,
        ultrasonic_speech_rate=500,
        log_level="INFO",
    )


def test_notebook_fallback_detects_large_school_notebook_shape():
    detector = ObjectDetector(_config())
    frame = np.full((480, 640, 3), 210, dtype=np.uint8)
    notebook = np.array([[170, 90], [500, 135], [470, 430], [130, 390]], dtype=np.int32)
    cv2.fillConvexPoly(frame, notebook, (85, 135, 205))
    cv2.polylines(frame, [notebook], True, (35, 35, 35), 5)
    cv2.putText(frame, "DUMBO", (210, 230), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (40, 40, 120), 5)

    detections = detector._notebook_fallback_detections(frame, [])

    assert detections
    assert detections[0].label == "libreta"
    assert detections[0].score >= 0.55


def test_notebook_fallback_does_not_duplicate_existing_book_detection():
    detector = ObjectDetector(_config())
    frame = np.full((480, 640, 3), 210, dtype=np.uint8)

    detections = detector._notebook_fallback_detections(
        frame,
        [Detection(label="libro", score=0.80, box=(10, 10, 100, 100))],
    )

    assert detections == []
