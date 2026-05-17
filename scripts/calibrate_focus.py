from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_vision.camera import CameraManager
from lumina_vision.config import AppConfig
from lumina_vision.ocr import OCRService


def _resize_preview(frame, max_width: int = 1100):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _draw_focus_preview(frame, ocr: OCRService, lens_position: float, sharpness: float, best_lens: float | None):
    preview = frame.copy()
    height, width = preview.shape[:2]
    x1, y1, x2, y2 = ocr.roi_box(preview)
    document_box = ocr.document_box(preview)

    if document_box is not None:
        cv2.drawContours(preview, [document_box], -1, (0, 220, 0), 3)
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)

    status = f"LensPosition={lens_position:.1f} | nitidez={sharpness:.1f}"
    if best_lens is not None:
        status += f" | mejor hasta ahora={best_lens:.1f}"
    cv2.putText(
        preview,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "Verde=hoja detectada | Amarillo=zona de lectura | Q=salir",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return _resize_preview(preview)


def main() -> int:
    config = AppConfig.load()
    config.camera_width = 1536
    config.camera_height = 864
    config.camera_framerate = 5.0
    config.camera_buffer_count = 2
    config.camera_lens_position = -1.0
    config.camera_focus_settle_seconds = 0.8

    debug_dir = ROOT_DIR / "debug_focus"
    debug_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraManager(config)
    ocr = OCRService(config)
    camera.start()

    results: list[tuple[float, float]] = []
    try:
        print("[Lumina] Coloca una hoja impresa a la distancia real de lectura.")
        print("[Lumina] No muevas la hoja durante el escaneo de enfoque.")
        print("[Lumina] Se abrira preview. Verde=hoja detectada, amarillo=zona de lectura, Q=salir.")
        time.sleep(1.0)

        best_lens_so_far: float | None = None
        best_score_so_far = -1.0
        for lens_position in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]:
            camera.set_lens_position(lens_position)
            time.sleep(0.8)

            best_frame = None
            best_sharpness = -1.0
            for _ in range(6):
                frame = camera.read()
                sharpness = ocr.sharpness(ocr.focus_region(frame))
                if sharpness > best_sharpness:
                    best_frame = frame
                    best_sharpness = sharpness
                cv2.imshow(
                    "Lumina Focus Calibration",
                    _draw_focus_preview(frame, ocr, lens_position, sharpness, best_lens_so_far),
                )
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("[Lumina] Calibracion cancelada por usuario.")
                    return 1
                time.sleep(0.08)

            results.append((lens_position, best_sharpness))
            if best_sharpness > best_score_so_far:
                best_score_so_far = best_sharpness
                best_lens_so_far = lens_position
            if best_frame is not None:
                cv2.imwrite(str(debug_dir / f"focus_{lens_position:.1f}_{best_sharpness:.1f}.jpg"), best_frame)
            print(f"[Lumina] LensPosition={lens_position:.1f} nitidez={best_sharpness:.1f}")

        best_lens, best_score = max(results, key=lambda item: item[1])
        print("")
        print(f"[Lumina] Mejor enfoque: LUMINA_CAMERA_LENS_POSITION={best_lens:.1f}")
        print(f"[Lumina] Nitidez maxima: {best_score:.1f}")
        print("[Lumina] Agrega ese valor a tu .env para lectura de texto.")
        return 0
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
