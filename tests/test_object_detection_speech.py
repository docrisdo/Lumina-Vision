from lumina_vision.detectors.tflite_detector import Detection
from scripts.test_object_detection import _detection_signature, _format_detection_message


def test_object_detection_message_says_detected_objects():
    detections = [
        Detection(label="libro", score=0.91, box=(0, 0, 10, 10)),
        Detection(label="botella", score=0.88, box=(20, 20, 40, 40)),
    ]

    message = _format_detection_message(detections)

    assert message == "Veo un libro, una botella"


def test_object_detection_message_counts_repeated_objects():
    detections = [
        Detection(label="silla", score=0.91, box=(0, 0, 10, 10)),
        Detection(label="silla", score=0.86, box=(20, 20, 40, 40)),
    ]

    assert _format_detection_message(detections) == "Veo 2 silla"


def test_object_detection_signature_tracks_label_counts():
    detections = [
        Detection(label="libro", score=0.91, box=(0, 0, 10, 10)),
        Detection(label="libro", score=0.86, box=(20, 20, 40, 40)),
        Detection(label="botella", score=0.82, box=(50, 50, 60, 60)),
    ]

    assert _detection_signature(detections) == "libro:2|botella:1"
