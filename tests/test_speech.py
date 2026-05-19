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


def test_piper_timeout_allows_model_startup(monkeypatch):
    speech = object.__new__(SpeechEngine)

    class Config:
        tts_command_timeout_seconds = 15.0

    speech.config = Config()

    assert speech._piper_timeout_for_text("El cuervo y la jarra") >= 35.0
