import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent / "taxonomy.json"


@dataclass
class Taxonomy:
    version: str
    bases: list[str]
    modifiers: list[str]
    tie_breaks: list[str]

    def validate_label(self, environment: str, modifiers: list[str]) -> None:
        if environment not in self.bases:
            raise ValueError(f"unknown environment {environment!r}")
        bad = [m for m in modifiers if m not in self.modifiers]
        if bad:
            raise ValueError(f"unknown modifiers {bad}")


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    data = json.loads((path or _DEFAULT).read_text(encoding="utf-8"))
    return Taxonomy(version=data["version"], bases=data["bases"],
                    modifiers=data["modifiers"],
                    tie_breaks=data.get("tie_breaks", []))
