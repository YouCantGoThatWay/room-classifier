from pathlib import Path

from sources.parse_awake import parse_awake

FIXTURE = Path(__file__).parent / "fixtures" / "sample_awake.wld"


def test_parses_rooms():
    rooms = parse_awake(FIXTURE.read_text())
    assert [r["vnum"] for r in rooms] == [0, 10500]
    t = rooms[0]
    assert t["name"] == "A Bright Light"
    assert t["description"].startswith("   What... oh no")
    assert "grab at" in t["description"]
    assert t["sector_hint"] == "inside"
    assert t["flags"] == []


def test_word_sectype_maps_to_hint_vocabulary():
    rooms = parse_awake(FIXTURE.read_text())
    assert rooms[1]["sector_hint"] == "city"
    assert rooms[1]["name"] == "Below A Corporate Tower"
    assert rooms[1]["description"].startswith("   Towering above East Tacoma")


def test_unknown_sectype_maps_to_empty():
    text = (
        "#5\n"
        "Name:\tMystery Node\n"
        "Desc:$\n"
        "   A shifting place.\n"
        "~\n"
        "Flags:\t0\n"
        "SecType:\tOrange Node\n"
        "BREAK\n"
    )
    rooms = parse_awake(text)
    assert rooms[0]["sector_hint"] == ""


def test_truncated_header_does_not_crash():
    assert parse_awake("#3001") == []
    assert parse_awake("#3001\n") == []


def test_truncated_after_name_does_not_crash():
    rooms = parse_awake("#3001\nName:\tHall\n")
    assert rooms == [
        {"vnum": 3001, "name": "Hall", "description": "",
         "sector_hint": "", "flags": []}
    ]


def test_truncated_description_does_not_crash():
    text = "#3001\nName:\tHall\nDesc:$\n   An unfinished room.\n"
    rooms = parse_awake(text)
    assert rooms[0]["vnum"] == 3001
    assert rooms[0]["description"] == "   An unfinished room."
    assert rooms[0]["sector_hint"] == ""


def test_old_style_circle_room_is_skipped_not_crashed():
    # A minority of AwakeMUD .wld files (e.g. 182.wld, 195.wld) are plain
    # CircleMUD-dialect rooms, not the AwakeMUD key-value dialect.
    text = (
        "#18200\n"
        "Office Entrance~\n"
        "   The tiles on the floor are spotless.\n"
        "~\n"
        "0 8 0 0\n"
        "S\n"
        "$~\n"
    )
    assert parse_awake(text) == []
