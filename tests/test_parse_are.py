from pathlib import Path
from sources.parse_are import parse_are_rooms

FIXTURE = Path(__file__).parent / "fixtures" / "sample.are"


def test_parses_rooms_section_only():
    rooms = parse_are_rooms(FIXTURE.read_text())
    assert [r["vnum"] for r in rooms] == [9201, 9202]
    assert rooms[0]["sector_hint"] == "desert"
    assert rooms[0]["description"].startswith("Endless dunes")


def test_sector_is_third_token_even_with_extras():
    rooms = parse_are_rooms(FIXTURE.read_text())
    assert rooms[1]["sector_hint"] == "city"


def test_truncated_rooms_section_does_not_crash():
    assert parse_are_rooms("#ROOMS\n#9201") == []
