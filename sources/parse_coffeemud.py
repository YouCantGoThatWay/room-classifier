"""Parser for CoffeeMud .cmare area exports.

Real .cmare files (verified against the cloned CoffeeMud repo, Task 8 step 1)
do not store room XML at the top level. Rooms are embedded, HTML-entity
escaped, inside an outer <ITEM>/<ITEXT> (or similar) container -- e.g. a
boardable ship or caravan item carries its own sub-area under
<SSAREA><AREA><ADATA>...<AROOMS><AROOM>...</AROOM></AROOMS></AREA></SSAREA>.
Once unescaped, each <AROOM> block carries:
  <ROOMID>   -- the original (string) room id, e.g. "shipname#3"
  <RCLAS>    -- the Java locale class name, e.g. ShipDeck, WoodenDeck
  <RDISP>    -- the room's display title
  <RDESC>    -- the room's long description (often self-closing/empty)
None of these match the brief's assumed top-level tag names
(AROOM/ROOMID/DISPLAY/TITLE/DESCRIPTION/RCLAS|CLASS) exactly except ROOMID
and RCLAS -- RDISP/RDESC replace DISPLAY|TITLE/DESCRIPTION.

We unescape the whole document once up front (the escaping is a single,
uniform level throughout the file) so a plain regex scan over <AROOM>...
</AROOM> blocks works regardless of how deeply the block was nested in the
original escaped text.
"""
import html
import re

_AROOM = re.compile(r"<AROOM>(.*?)</AROOM>", re.S | re.I)


def _tag(block: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", block, re.S | re.I)
    return html.unescape(m.group(1)).strip() if m else ""


def parse_cmare(text: str) -> list[dict]:
    text = html.unescape(text)
    rooms = []
    for idx, m in enumerate(_AROOM.finditer(text)):
        block = m.group(1)
        rid = _tag(block, "ROOMID")
        name = _tag(block, "RDISP")
        desc = _tag(block, "RDESC")
        locale = _tag(block, "RCLAS")
        if not name or not desc:
            continue
        rooms.append({"vnum": idx, "name": name, "description": desc,
                      "sector_hint": locale,
                      "flags": [f"cmid:{rid or idx}"]})
    return rooms
