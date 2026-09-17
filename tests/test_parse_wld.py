from pathlib import Path
from sources.parse_wld import SECTOR_NAMES, parse_wld

FIXTURE = Path(__file__).parent / "fixtures" / "sample.wld"


def test_parses_rooms():
    rooms = parse_wld(FIXTURE.read_text())
    assert [r["vnum"] for r in rooms] == [3001, 3054]
    t = rooms[0]
    assert t["name"] == "The Temple of Midgaard"
    assert t["description"].startswith("You are in the southern end")
    assert "pillars rise" in t["description"]
    assert t["sector_hint"] == "inside"
    assert t["flags"] == ["abd"]


def test_unknown_sector_maps_to_empty():
    rooms = parse_wld(FIXTURE.read_text())
    assert rooms[1]["sector_hint"] == SECTOR_NAMES.get(11, "")
