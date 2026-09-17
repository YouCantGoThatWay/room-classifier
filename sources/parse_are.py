"""Parser for Merc / ROM / SMAUG .are files (#ROOMS section only)."""
from sources.parse_wld import SECTOR_NAMES as _BASE

SECTOR_NAMES = {**_BASE, 10: "air", 11: "desert"}


def parse_are_rooms(text: str) -> list[dict]:
    lines = text.splitlines()
    rooms, i, in_rooms = [], 0, False
    while i < len(lines):
        line = lines[i].strip()
        if line == "#ROOMS":
            in_rooms = True
            i += 1
            continue
        if not in_rooms:
            i += 1
            continue
        if line == "#0":
            break
        if not line.startswith("#"):
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
        name = lines[i].rstrip().rstrip("~")
        i += 1
        desc = []
        while i < len(lines) and lines[i].strip() != "~":
            desc.append(lines[i])
            i += 1
        i += 1
        sector_hint, flags = "", []
        if i < len(lines):
            toks = lines[i].split()
            if len(toks) >= 3:
                try:
                    sector_hint = SECTOR_NAMES.get(int(toks[2]), "")
                except ValueError:
                    pass
                flags = [toks[1]]
            i += 1
        while i < len(lines) and lines[i].strip() != "S":
            i += 1
        i += 1
        rooms.append({"vnum": vnum, "name": name,
                      "description": "\n".join(desc),
                      "sector_hint": sector_hint, "flags": flags})
    return rooms
