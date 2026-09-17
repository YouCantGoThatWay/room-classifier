from pathlib import Path

from sources.parse_fuss import parse_fuss_rooms

FIXTURE = Path(__file__).parent / "fixtures" / "sample_fuss.are"


def test_parses_room_blocks():
    rooms = parse_fuss_rooms(FIXTURE.read_text())
    assert [r["vnum"] for r in rooms] == [18001, 18002]
    t = rooms[0]
    assert t["name"] == "Salis D'aar Spaceport"
    assert t["description"].startswith("The spaceport sits on a natural bedrock")
    assert "risk" in t["description"]
    assert t["sector_hint"] == "city"
    assert t["flags"] == ["nomob"]


def test_multi_word_flags_split_into_list():
    rooms = parse_fuss_rooms(FIXTURE.read_text())
    assert rooms[1]["flags"] == ["can_land", "can_fly"]
    assert rooms[1]["name"] == "Landing Pad 2"


def test_unknown_sector_maps_to_empty():
    text = (
        "#ROOM\n"
        "Vnum     9999\n"
        "Name     Odd Place~\n"
        "Sector   ocean floor~\n"
        "Flags    nomob~\n"
        "Desc     A strange place.\n"
        "~\n"
        "#ENDROOM\n"
    )
    rooms = parse_fuss_rooms(text)
    assert rooms[0]["sector_hint"] == ""


def test_room_without_desc_field_is_empty_description():
    text = (
        "#ROOM\n"
        "Vnum     1\n"
        "Name     1~\n"
        "Sector   city~\n"
        "Flags    nomob~\n"
        "Stats    0 0 840\n"
        "#ENDROOM\n"
    )
    rooms = parse_fuss_rooms(text)
    assert rooms[0]["description"] == ""


def test_truncated_room_does_not_crash():
    assert parse_fuss_rooms("#ROOM\nVnum     18001\n") == []
    assert parse_fuss_rooms("#ROOM\n") == []
    assert parse_fuss_rooms("") == []


def test_falls_back_to_old_style_rooms_section():
    # A minority of FUSS-lineage .are files (e.g. SWFOTE's space.are) use
    # the plain ROM/Merc #ROOMS section dialect instead of #ROOM blocks.
    text = (
        "#ROOMS\n"
        "#9201\n"
        "Endless Dunes~\n"
        "You see sand everywhere.\n"
        "~\n"
        "0 0 11 0\n"
        "S\n"
        "#0\n"
    )
    rooms = parse_fuss_rooms(text)
    assert [r["vnum"] for r in rooms] == [9201]
    assert rooms[0]["sector_hint"] == "desert"
