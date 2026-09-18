"""World/area-grouped split. THE tier-enforcement point for training."""
import hashlib
from pathlib import Path

from common.schema import RoomRecord, read_jsonl, write_jsonl


def _group_key(rec: RoomRecord) -> str:
    if rec.synthetic:
        if rec.parent_id:
            parts = rec.parent_id.split(":")
            return f"{parts[0]}:{parts[1]}"
        return "__synthetic_train__"
    return f"{rec.source}:{rec.area}"


def assign_split(rec: RoomRecord) -> str:
    key = _group_key(rec)
    if key == "__synthetic_train__":
        return "train"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10
    return {0: "test", 1: "val"}.get(h, "train")


def split_records(records: list[RoomRecord]) -> dict[str, list[RoomRecord]]:
    parts: dict[str, list[RoomRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        if r.tier == "eval":
            raise ValueError(f"{r.id}: eval-tier record in training path")
        if not r.environment:
            raise ValueError(f"{r.id}: missing environment label")
        part = assign_split(r)
        if r.synthetic and part != "train":
            raise AssertionError(f"{r.id}: synthetic record in {part}")
        parts[part].append(r)
    return parts


def main() -> None:
    records = read_jsonl(Path("data/labeled/train.jsonl"))
    parts = split_records(records)
    for name, recs in parts.items():
        write_jsonl(Path(f"data/splits/{name}.jsonl"), recs)
        print(f"{name}: {len(recs)}")


if __name__ == "__main__":
    main()
