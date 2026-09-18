import pytest
from common.taxonomy import load_taxonomy


def test_load_default():
    t = load_taxonomy()
    assert t.version == "1.0.0"
    assert "spacecraft" in t.bases and "unknown" in t.bases
    assert len(t.bases) == 15 and len(t.modifiers) == 6


def test_validate_label():
    t = load_taxonomy()
    t.validate_label("forest", ["dark", "dense"])
    with pytest.raises(ValueError):
        t.validate_label("jungle", [])
    with pytest.raises(ValueError):
        t.validate_label("forest", ["gloomy"])
