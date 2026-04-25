from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cv2
from loguru import logger

from lumina_vision.config import AppConfig

try:
    from picamera2 import Picamera2
    from libcamera import controls
except ImportError:
    Picamera2 = None
    controls = None


@dataclass(slots=True)
class CameraFrame:
    image: Any
    timestamp: float


class CameraManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._backend = "opencv"
        self._camera: Any = None

    def start(self) -> None:
        if Picamera2 is not None:
            self._start_picamera2()
            return

        logger.warning("Picamera2 no esta disponible. Usando fallback con OpenCV.")
        self._start_opencv()

    def _start_picamera2(self) -> None:
        assert Picamera2 is not None

        available_cameras = Picamera2.global_camera_info()
        if not available_cameras:
            raise RuntimeError(
                "Picamera2 no detecto ninguna camara. Cierra otros programas que usen la camara, "
                "revisa el flex CSI y prueba: rpicam-hello --list-cameras",
            )

        self._camera = Picamera2()
        camera_config = self._camera.create_preview_configuration(
            main={"size": (self.config.camera_width, self.config.camera_height), "format": "RGB888"},
            buffer_count=4,
            controls={"FrameRate": self.config.camera_framerate},
        )
        self._camera.configure(camera_config)

        camera_controls = self._build_focus_controls(for_ocr=False)
        if camera_controls:
            self._camera.set_controls(camera_controls)

        self._camera.start()
        self._backend = "picamera2"
        logger.info("Camara iniciada con Picamera2.")
        self.refocus()

    def _build_focus_controls(self, for_ocr: bool) -> dict[str, Any]:
        if controls is None:
            af_mode = self.config.camera_af_mode.lower()
            fallback: dict[str, Any] = {}
            if af_mode == "continuous":
                fallback["AfMode"] = 2
            elif af_mode == "auto":
                fallback["AfMode"] = 1
            return fallback

        af_mode = "auto" if for_ocr else self.config.camera_af_mode.lower()
        focus_controls: dict[str, Any] = {}
        if af_mode == "continuous":
            focus_controls["AfMode"] = controls.AfModeEnum.Continuous
        elif af_mode == "auto":
            focus_controls["AfMode"] = controls.AfModeEnum.Auto
            focus_controls["AfTrigger"] = controls.AfTriggerEnum.Start

        af_range = "macro" if for_ocr else self.config.camera_af_range.lower()
        if af_range == "macro":
            focus_controls["AfRange"] = controls.AfRangeEnum.Macro
        elif af_range == "normal":
            focus_controls["AfRange"] = controls.AfRangeEnum.Normal
        elif af_range == "full":
            focus_controls["AfRange"] = controls.AfRangeEnum.Full

        if self.config.camera_af_speed.lower() == "fast":
            focus_controls["AfSpeed"] = controls.AfSpeedEnum.Fast
        elif self.config.camera_af_speed.lower() == "normal":
            focus_controls["AfSpeed"] = controls.AfSpeedEnum.Normal

        return focus_controls

    def _start_opencv(self) -> None:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        if not capture.isOpened():
            raise RuntimeError("No se pudo abrir la camara con OpenCV.")

        self._camera = capture
        self._backend = "opencv"
        logger.info("Camara iniciada con OpenCV.")

    def read(self) -> Any:
        if self._backend == "picamera2":
            frame = self._camera.capture_array()
            color_mode = self.config.camera_color_mode.lower()
            if color_mode == "rgb_to_bgr":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif color_mode == "bgr_to_rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.config.camera_hflip:
                frame = cv2.flip(frame, 1)
            if self.config.camera_vflip:
                frame = cv2.flip(frame, 0)
            return frame

        ok, frame = self._camera.read()
        if not ok:
            raise RuntimeError("No se pudo leer un frame de la camara.")
        return frame

    def refocus(self) -> None:
        if self._backend != "picamera2" or self._camera is None:
            return
        if not self.config.camera_refocus_before_ocr:
            return

        af_mode = self.config.camera_af_mode.lower()
        if af_mode not in {"auto", "continuous"}:
            return

        try:
            focus_controls = self._build_focus_controls(for_ocr=True)
            if focus_controls:
                self._camera.set_controls(focus_controls)
            if self.config.camera_focus_settle_seconds > 0:
                time.sleep(self.config.camera_focus_settle_seconds)
        except Exception as exc:
            logger.debug("No se pudo reenfocar la camara: {}", exc)

    def stop(self) -> None:
        if self._camera is None:
            return
        if self._backend == "picamera2":
            self._camera.stop()
        else:
            self._camera.release()
        self._camera = None
        logger.info("Camara detenida.")
