from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from lumina_vision.config import AppConfig

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
    except ImportError:
        Interpreter = None


LABEL_TRANSLATIONS = {
    "person": "persona",
    "bicycle": "bicicleta",
    "car": "auto",
    "motorcycle": "motocicleta",
    "airplane": "avion",
    "bus": "autobus",
    "train": "tren",
    "truck": "camion",
    "boat": "bote",
    "traffic light": "semaforo",
    "fire hydrant": "hidrante",
    "street sign": "senal vial",
    "stop sign": "senal de alto",
    "parking meter": "parquimetro",
    "bench": "banca",
    "bird": "pajaro",
    "cat": "gato",
    "dog": "perro",
    "horse": "caballo",
    "sheep": "oveja",
    "cow": "vaca",
    "elephant": "elefante",
    "bear": "oso",
    "zebra": "cebra",
    "giraffe": "jirafa",
    "hat": "sombrero",
    "backpack": "mochila",
    "umbrella": "paraguas",
    "shoe": "zapato",
    "eye glasses": "lentes",
    "handbag": "bolsa",
    "tie": "corbata",
    "suitcase": "maleta",
    "frisbee": "frisbi",
    "skis": "esquis",
    "snowboard": "tabla de snowboard",
    "sports ball": "pelota",
    "kite": "papalote",
    "baseball bat": "bate",
    "baseball glove": "guante",
    "skateboard": "patineta",
    "surfboard": "tabla de surf",
    "tennis racket": "raqueta",
    "bottle": "botella",
    "plate": "plato",
    "wine glass": "copa",
    "cup": "taza",
    "fork": "tenedor",
    "knife": "cuchillo",
    "spoon": "cuchara",
    "bowl": "tazon",
    "banana": "platano",
    "apple": "manzana",
    "sandwich": "sandwich",
    "orange": "naranja",
    "broccoli": "brocoli",
    "carrot": "zanahoria",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "dona",
    "cake": "pastel",
    "chair": "silla",
    "couch": "sofa",
    "potted plant": "planta",
    "bed": "cama",
    "mirror": "espejo",
    "dining table": "mesa",
    "window": "ventana",
    "desk": "escritorio",
    "toilet": "inodoro",
    "door": "puerta",
    "tv": "television",
    "laptop": "laptop",
    "mouse": "raton",
    "remote": "control remoto",
    "keyboard": "teclado",
    "cell phone": "celular",
    "microwave": "microondas",
    "oven": "horno",
    "toaster": "tostador",
    "sink": "lavabo",
    "refrigerator": "refrigerador",
    "blender": "licuadora",
    "book": "libro",
    "clock": "reloj",
    "vase": "florero",
    "scissors": "tijeras",
    "teddy bear": "oso de peluche",
    "hair drier": "secadora",
    "toothbrush": "cepillo de dientes",
    "hair brush": "cepillo",
}


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]


class ObjectDetector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._interpreter: Any = None
        self._input_details: list[dict[str, Any]] = []
        self._output_details: list[dict[str, Any]] = []
        self._input_height = 0
        self._input_width = 0
        self._input_dtype: Any = np.uint8
        self._labels = self._load_labels(config.labels_path)

    def available(self) -> bool:
        return Interpreter is not None and self.config.detection_model_path.exists()

    def load(self) -> None:
        if Interpreter is None:
            raise RuntimeError(
                "No hay runtime de TensorFlow Lite. Instala tflite_runtime o tensorflow.",
            )
        if not self.config.detection_model_path.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo: {self.config.detection_model_path}",
            )

        self._interpreter = Interpreter(model_path=str(self.config.detection_model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        input_shape = self._input_details[0]["shape"]
        self._input_height = int(input_shape[1])
        self._input_width = int(input_shape[2])
        self._input_dtype = self._input_details[0]["dtype"]

        logger.info(
            "Detector TFLite cargado. Entrada esperada: {}x{} dtype={}",
            self._input_width,
            self._input_height,
            self._input_dtype,
        )

    def _load_labels(self, path: Path) -> list[str]:
        if not path.exists():
            logger.warning("No se encontro el archivo de etiquetas: {}", path)
            return []
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [LABEL_TRANSLATIONS.get(label, label) for label in labels]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._interpreter is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_width, self._input_height))
        input_tensor = np.expand_dims(resized, axis=0)

        if self._input_dtype == np.float32:
            input_tensor = (input_tensor.astype(np.float32) / 255.0).astype(np.float32)
        else:
            input_tensor = input_tensor.astype(self._input_dtype)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()

        outputs = [self._interpreter.get_tensor(detail["index"]) for detail in self._output_details]
        boxes, classes, scores, count = self._normalize_outputs(outputs)

        frame_height, frame_width = frame.shape[:2]
        detections: list[Detection] = []

        for idx in range(min(int(count), len(scores))):
            score = float(scores[idx])
            if score < self.config.detection_score_threshold:
                continue

            class_id = int(classes[idx])
            label = self._labels[class_id] if 0 <= class_id < len(self._labels) else f"class_{class_id}"
            ymin, xmin, ymax, xmax = boxes[idx]
            left = max(0, int(xmin * frame_width))
            top = max(0, int(ymin * frame_height))
            right = min(frame_width, int(xmax * frame_width))
            bottom = min(frame_height, int(ymax * frame_height))
            detections.append(
                Detection(
                    label=label,
                    score=score,
                    box=(left, top, right, bottom),
                ),
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[: self.config.detection_max_results]

    def _normalize_outputs(
        self,
        outputs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if len(outputs) < 4:
            raise RuntimeError("El modelo TFLite no devolvio suficientes tensores de salida.")

        flattened = [np.squeeze(output) for output in outputs]

        candidates: dict[str, np.ndarray] = {}
        for output in flattened:
            if output.ndim == 2 and output.shape[-1] == 4:
                candidates["boxes"] = output
            elif output.ndim == 1 and np.issubdtype(output.dtype, np.floating):
                if output.size == 1:
                    candidates["count"] = output
                elif output.max(initial=0.0) <= 1.0:
                    candidates["scores"] = output
                else:
                    candidates["classes"] = output

        if {"boxes", "classes", "scores"} <= set(candidates):
            count_value = int(candidates.get("count", np.array([len(candidates["scores"])]))[0])
            return (
                candidates["boxes"],
                candidates["classes"],
                candidates["scores"],
                count_value,
            )

        return (
            np.squeeze(outputs[0]),
            np.squeeze(outputs[1]),
            np.squeeze(outputs[2]),
            int(np.squeeze(outputs[3])),
        )
