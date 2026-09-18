"""Parse cloned repos into normalized RoomRecord jsonl per source."""
from pathlib import Path

from common.schema import RoomRecord, write_jsonl
from common.textclean import clean_text, is_trivial
from sources.download import SOURCES, SourceSpec
from sources.parse_are import parse_are_rooms
from sources.parse_awake import parse_awake
from sources.parse_coffeemud import parse_cmare
from sources.parse_fuss import parse_fuss_rooms
from sources.parse_wld import parse_wld

_PARSERS = {"wld": parse_wld, "are": parse_are_rooms,
            "coffeemud": parse_cmare, "awake": parse_awake,
            "fuss": parse_fuss_rooms}


def build_source(spec: SourceSpec, repo_dir: Path) -> list[RoomRecord]:
    parse = _PARSERS[spec.parser]
    records: list[RoomRecord] = []
    for f in sorted(repo_dir.glob(spec.world_glob)):
        if f.name.startswith("."):
            continue  # exFAT AppleDouble ._* sidecars match the globs
        area = f.stem
        for room in parse(f.read_text(errors="replace")):
            name = clean_text(room["name"])
            desc = clean_text(room["description"])
            if not name or is_trivial(name, desc):
                continue
            rec = RoomRecord(
                id=f"{spec.name}:{area}:{room['vnum']}",
                source=spec.name, tier=spec.tier, world=spec.name,
                area=area, name=name, description=desc,
                sector_hint=room["sector_hint"], flags=room["flags"],
                license=spec.license_note)
            rec.validate()
            records.append(rec)
    return records


def main(raw_dir: Path = Path("data/raw"),
         out_dir: Path = Path("data/normalized")) -> None:
    for name, spec in SOURCES.items():
        repo = raw_dir / name
        if not repo.exists():
            print(f"{name}: SKIP (not cloned)")
            continue
        recs = build_source(spec, repo)
        n = write_jsonl(out_dir / f"{name}.jsonl", recs)
        print(f"{name}: {n} rooms")


if __name__ == "__main__":
    main()
