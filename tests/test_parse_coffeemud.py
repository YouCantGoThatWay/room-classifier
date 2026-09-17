from pathlib import Path

from sources.parse_coffeemud import parse_cmare

FIXTURE = Path(__file__).parent / "fixtures" / "sample.cmare"


def test_parses_rooms():
    rooms = parse_cmare(FIXTURE.read_text(errors="replace"))
    assert len(rooms) == 2
    for r in rooms:
        assert r["name"] and r["description"]
        assert r["flags"][0].startswith("cmid:")
        assert isinstance(r["vnum"], int)


def test_sector_hint_from_locale_class():
    rooms = parse_cmare(FIXTURE.read_text(errors="replace"))
    assert all(r["sector_hint"] != "" for r in rooms)
    hints = {r["sector_hint"] for r in rooms}
    assert hints == {"ShipLightGunDeck", "WoodenDeck"}


def test_room_names_and_ids():
    rooms = parse_cmare(FIXTURE.read_text(errors="replace"))
    names = {r["name"] for r in rooms}
    assert names == {"Carroballista", "The bench"}
    ids = {r["flags"][0] for r in rooms}
    assert ids == {
        "cmid:the reinforced carroballista [NEWNAME]#1",
        "cmid:the reinforced carroballista [NEWNAME]#3",
    }
