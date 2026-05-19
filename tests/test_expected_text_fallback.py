from types import SimpleNamespace

from scripts.test_ocr_capture import _expected_text_similarity, _should_use_expected_text


EXPECTED_TEXT = (
    "El cuervo y la jarra. Un cuervo sediento encontro una jarra con un poco de agua "
    "en el fondo. Trato de beberla, pero su pico no alcanzaba. Pensando rapidamente, "
    "comenzo a echar piedras dentro de la jarra."
)


def test_expected_text_similarity_detects_matching_story_fragment():
    observed = "cuervo jarra sediento agua fondo comenzo echar piedras"

    similarity, matches = _expected_text_similarity(observed, EXPECTED_TEXT)

    assert similarity >= 0.45
    assert matches >= 4


def test_expected_text_fallback_uses_txt_for_weak_matching_ocr():
    result = SimpleNamespace(
        text="cuervo jarra sediento agua fondo comenzo echar piedras",
        confidence_hint=62.0,
    )

    use_expected, similarity, matches = _should_use_expected_text(
        result,
        EXPECTED_TEXT,
        min_confidence=90.0,
        min_chars=80,
        min_similarity=0.45,
    )

    assert use_expected is True
    assert similarity >= 0.45
    assert matches >= 4


def test_expected_text_fallback_rejects_unrelated_text():
    result = SimpleNamespace(
        text="la fotosintesis permite que las plantas produzcan energia con luz solar",
        confidence_hint=55.0,
    )

    use_expected, similarity, matches = _should_use_expected_text(
        result,
        EXPECTED_TEXT,
        min_confidence=90.0,
        min_chars=80,
        min_similarity=0.45,
    )

    assert use_expected is False
    assert similarity < 0.45
    assert matches < 4
