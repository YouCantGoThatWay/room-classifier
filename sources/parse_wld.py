"""Parser for CircleMUD / tbaMUD .wld room files (title, desc, sector)."""
SECTOR_NAMES = {
    0: "inside", 1: "city", 2: "field", 3: "forest", 4: "hills",
    5: "mountain", 6: "water_swim", 7: "water_noswim", 8: "flying",
    9: "underwater",
}


def parse_wld(text: str) -> list[dict]:
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
        name = lines[i].rstrip().rstrip("~")
        i += 1
        desc = []
        while i < len(lines) and lines[i].strip() != "~":
            desc.append(lines[i])
            i += 1
        i += 1  # past the lone ~
        sector_hint, flags = "", []
        if i < len(lines):
            toks = lines[i].split()
            if toks:
                try:
                    sector_hint = SECTOR_NAMES.get(int(toks[-1]), "")
                except ValueError:
                    pass
                flags = toks[1:-1]
            i += 1
        while i < len(lines) and lines[i].strip() != "S":
            i += 1
        i += 1
        rooms.append({"vnum": vnum, "name": name,
                      "description": "\n".join(desc),
                      "sector_hint": sector_hint, "flags": flags})
    return rooms
