from common.textclean import (build_text, clean_text, is_trivial,
                              normalized_key, strip_codes)


def test_strip_color_codes():
    assert strip_codes("&RThe &GForest&x") == "The Forest"       # SMAUG
    assert strip_codes("@rDark@n cave") == "Dark cave"           # tba @-codes
    assert strip_codes("{cMisty{x path") == "Misty path"         # ROM
    assert strip_codes("\x1b[31mRed\x1b[0m room") == "Red room"  # raw ANSI


def test_clean_text_normalizes_whitespace_and_tildes():
    raw = "A hall.~\r\n   Dust    hangs\n\nin the air.  "
    assert clean_text(raw) == "A hall. Dust hangs in the air."


def test_is_trivial():
    assert is_trivial("Void", "short")
    assert not is_trivial("Temple", "A vaulted stone hall stretches north.")


def test_build_text():
    assert build_text("Temple", "A hall.") == "Temple\nA hall."


def test_normalized_key_ignores_case_and_punct():
    a = normalized_key("The Temple", "A vaulted, stone hall!")
    b = normalized_key("the temple", "a vaulted stone hall")
    assert a == b
