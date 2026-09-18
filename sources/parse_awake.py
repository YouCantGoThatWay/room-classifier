"""Parser for AwakeMUD .wld files (key-value dialect, not CircleMUD-style).

Rooms start with a bare ``#<vnum>`` line, followed by ``Key:\tvalue`` lines
(``Name:``, ``Desc:$`` + free-text description terminated by a lone ``~``,
``SecType:``, plus assorted other fields), then ``[POINTS]`` / ``[EXIT
<direction>]`` sub-sections, and end with a bare ``BREAK`` line.

A small minority of AwakeMUD .wld files (e.g. 182.wld, 195.wld) are plain
CircleMUD-dialect rooms instead -- those rooms don't match the ``Name:``
key-value shape here and are silently skipped rather than crashing.
"""
SECTYPE_ALIASES = {
    "inside": "inside",
    "city": "city",
    "field": "field",
    "forest": "forest",
    "hills": "hills",
    "mountain": "mountain",
    "mountains": "mountain",
    "water (swim)": "water_swim",
    "water (no swim)": "water_noswim",
    "flying": "flying",
    "underwater": "underwater",
    "desert": "desert",
    "air": "air",
}


def _map_sectype(value: str) -> str:
    return SECTYPE_ALIASES.get(value.strip().lower(), "")


def parse_awake(text: str) -> list[dict]:
    lines = text.splitlines()
    rooms, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("#") or line.startswith("$"):
            i += 1
            continue
        try:
            vnum = int(line[1:])
        except ValueError:
            i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        key, _, val = lines[i].partition(":")
        if key.strip() != "Name":
            # Not the AwakeMUD dialect (e.g. plain CircleMUD room) -- skip.
            i += 1
            continue
        name = val.strip()
        i += 1

        desc = ""
        if i < len(lines):
            key, _, val = lines[i].partition(":")
            if key.strip() == "Desc":
                i += 1
                desc_lines = []
                while i < len(lines) and lines[i].strip() != "~":
                    desc_lines.append(lines[i])
                    i += 1
                i += 1  # past the lone ~
                desc = "\n".join(desc_lines)

        sector_hint = ""
        while i < len(lines) and lines[i].strip() != "BREAK":
            key, _, val = lines[i].partition(":")
            if key.strip() == "SecType":
                sector_hint = _map_sectype(val)
            i += 1
        i += 1  # past BREAK (or past EOF if truncated)

        rooms.append({"vnum": vnum, "name": name, "description": desc,
                      "sector_hint": sector_hint, "flags": []})
    return rooms
