"""Parser for FUSS (Star Wars: FOTE / SMAUG-lineage) .are files.

FUSS areas hold repeated ``#ROOM`` ... ``#ENDROOM`` blocks (no ``#ROOMS``
wrapper section) containing space-separated ``Key value`` lines, e.g.::

    #ROOM
    Vnum     18001
    Name     Salis D'aar Spaceport~
    Sector   city~
    Flags    nomob~
    Stats    0 0 15288
    Desc     The spaceport sits on ...
    ...
    ~
    #EXIT
    Direction north~
    ToRoom    18017
    #ENDEXIT
    #ENDROOM

``Sector`` is a word (not a number) here, unlike the ROM/Merc dialect.
``Desc`` is optional -- plenty of real rooms carry no description at all.

A minority of files in the FUSS-lineage corpus (e.g. SWFOTE's space.are)
are actually the older ROM/Merc ``#ROOMS`` section dialect; when no
``#ROOM`` blocks are found at all, fall back to that parser.

Assumes a single dialect per file: if a file contained both ``#ROOM``
blocks and a ``#ROOMS`` section, the native ``#ROOM``-block parse (run
first) would win outright and the ``#ROOMS`` rooms would be dropped, since
the fallback only runs when the native parse yields zero rooms. No such
mixed-dialect file exists in the current corpus (verified: fallback fires
only for space.are, which has no ``#ROOM`` blocks at all).
"""
from sources.parse_are import parse_are_rooms

SECTOR_ALIASES = {
    "inside": "inside",
    "city": "city",
    "field": "field",
    "forest": "forest",
    "hills": "hills",
    "mountain": "mountain",
    "mountains": "mountain",
    "water swim": "water_swim",
    "water noswim": "water_noswim",
    "water (swim)": "water_swim",
    "water (no swim)": "water_noswim",
    "flying": "flying",
    "underwater": "underwater",
    "desert": "desert",
    "air": "air",
}


def _map_sector(value: str) -> str:
    return SECTOR_ALIASES.get(value.strip().lower(), "")


def _parse_fuss_blocks(text: str) -> list[dict]:
    lines = text.splitlines()
    rooms, i = [], 0
    while i < len(lines):
        if lines[i].strip() != "#ROOM":
            i += 1
            continue
        i += 1
        vnum = None
        name, sector_hint, flags, desc = "", "", [], ""
        while i < len(lines) and lines[i].strip() != "#ENDROOM":
            line = lines[i]
            key, _, rest = line.partition(" ")
            key, rest = key.strip(), rest.strip()
            if key == "Vnum":
                try:
                    vnum = int(rest)
                except ValueError:
                    vnum = None
                i += 1
            elif key == "Name":
                name = rest.rstrip("~").strip()
                i += 1
            elif key == "Sector":
                sector_hint = _map_sector(rest.rstrip("~"))
                i += 1
            elif key == "Flags":
                flags = rest.rstrip("~").split()
                i += 1
            elif key == "Desc":
                i += 1
                desc_lines = [rest] if rest else []
                while i < len(lines) and lines[i].strip() != "~":
                    desc_lines.append(lines[i])
                    i += 1
                i += 1  # past the lone ~
                desc = "\n".join(desc_lines)
            else:
                i += 1
        i += 1  # past #ENDROOM (or past EOF if truncated)
        if vnum is not None and name:
            rooms.append({"vnum": vnum, "name": name, "description": desc,
                          "sector_hint": sector_hint, "flags": flags})
    return rooms


def parse_fuss_rooms(text: str) -> list[dict]:
    rooms = _parse_fuss_blocks(text)
    if rooms:
        return rooms
    if any(line.strip() == "#ROOMS" for line in text.splitlines()):
        return parse_are_rooms(text)
    return []
