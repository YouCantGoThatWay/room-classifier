"""Prepare deterministic labeling batches for in-session Claude labeling."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common.schema import RoomRecord, read_jsonl


def make_batches(records: list[RoomRecord], out_dir: Path, size: int = 40,
                 cap: int | None = None, seed: int = 17) -> int:
    rng = random.Random(seed)
    if cap is not None and cap < len(records):
        groups: dict[tuple, list[RoomRecord]] = defaultdict(list)
        for r in records:
            groups[(r.source, r.area)].append(r)
        frac = cap / len(records)
        sample: list[RoomRecord] = []
        for g in sorted(groups, key=str):
            members = sorted(groups[g], key=lambda r: r.id)
            k = max(1, round(len(members) * frac))
            sample.extend(rng.sample(members, min(k, len(members))))
        records = sample[:cap]
    records = sorted(records, key=lambda r: r.id)
    rng.shuffle(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(0, len(records), size):
        batch = records[i:i + size]
        payload = {"rooms": [{"id": r.id, "name": r.name,
                              "description": r.description,
                              "sector_hint": r.sector_hint}
                             for r in batch]}
        (out_dir / f"batch_{n:03d}.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False))
        n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--size", type=int, default=40)
    a = ap.parse_args()
    recs = read_jsonl(Path(f"data/dedup/{a.split}.jsonl"))
    n = make_batches(recs, Path(f"data/labeling/{a.split}"),
                     size=a.size, cap=a.cap)
    print(f"{n} batches for {a.split}")
