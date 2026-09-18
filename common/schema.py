"""Single JSONL record schema shared by every pipeline stage."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TIERS = ("train", "eval")


@dataclass
class RoomRecord:
    id: str
    source: str
    tier: str
    world: str
    area: str
    name: str
    description: str
    sector_hint: str = ""
    flags: list[str] = field(default_factory=list)
    license: str = ""
    synthetic: bool = False
    parent_id: str | None = None
    environment: str | None = None
    modifiers: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("id must be non-empty")
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if not self.name or not self.description:
            raise ValueError(f"{self.id}: name and description must be non-empty")


def write_jsonl(path: Path, records: Iterable[RoomRecord]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[RoomRecord]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(RoomRecord(**json.loads(line)))
    return out
