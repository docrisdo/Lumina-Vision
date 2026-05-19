import queue

from lumina_vision.speech import SpeechEngine


def test_speech_normalizes_ocr_punctuation_and_common_missing_y():
    speech = object.__new__(SpeechEngine)

    text = speech._normalize_speech_text(
        "El cuervo la jarra . Un cuervo sediento , , encontro agua . "
        "Este cuento ensena sobre la inteligencia la perseverancia .",
    )

    assert "El cuervo y la jarra." in text
    assert ", ," not in text
    assert "inteligencia y la perseverancia" in text


def test_speech_splits_long_text_for_piper():
    speech = object.__new__(SpeechEngine)
    text = (
        "El cuervo y la jarra. "
        "Un cuervo sediento encontro una jarra con un poco de agua en el fondo. "
        "Pensando rapidamente comenzo a echar piedras dentro de la jarra. "
        "Este cuento ensena sobre la inteligencia y la perseverancia para resolver problemas."
    )

    chunks = speech._split_speech_chunks(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 75 for chunk in chunks)


def test_speech_can_use_fluent_chunks_for_clean_expected_text():
    speech = object.__new__(SpeechEngine)
    text = (
        "El cuervo y la jarra. "
        "Un cuervo sediento encontro una jarra con un poco de agua en el fondo. "
        "Pensando rapidamente comenzo a echar piedras dentro de la jarra. "
        "Este cuento ensena sobre la inteligencia y la perseverancia para resolver problemas."
    )

    chunks = speech._split_speech_chunks(text, max_chars=180)

    assert len(chunks) < len(speech._split_speech_chunks(text))
    assert all(len(chunk) <= 180 for chunk in chunks)


def test_speech_smooths_ocr_artifact_periods_between_words():
    speech = object.__new__(SpeechEngine)

    text = speech._smooth_ocr_reading_text(
        "Un cuervo sediento encontro. una jarra con agua. en el fondo. Trato de beberla.",
    )

    assert "encontro una jarra" in text
    assert "agua en el fondo" in text
    assert "fondo. Trato" in text


def test_piper_timeout_allows_model_startup(monkeypatch):
    speech = object.__new__(SpeechEngine)

    class Config:
        tts_command_timeout_seconds = 15.0

    speech.config = Config()

    assert speech._piper_timeout_for_text("El cuervo y la jarra") >= 35.0


def test_object_speech_uses_fast_espeak_path(monkeypatch):
    speech = object.__new__(SpeechEngine)

    class Config:
        speech_volume = 1.0
        speech_rate = 170
        tts_output = "direct"
        tts_command_timeout_seconds = 15.0
        speech_max_queue_size = 1

    spoken = []
    speech.config = Config()
    speech._queue = queue.Queue()
    speech._object_speech_thread = None

    monkeypatch.setattr("lumina_vision.speech.shutil.which", lambda command: "espeak-ng" if command == "espeak-ng" else None)
    monkeypatch.setattr(speech, "_speak_with_espeak", lambda text: spoken.append(text))

    speech.speak_object("Veo una libreta")
    speech.wait_until_done()

    assert spoken == ["Veo una libreta"]
    assert speech._queue.empty()
