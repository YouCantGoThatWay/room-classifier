"""Validate + merge Claude label files onto records; build review queue."""
import argparse
import json
from collections import Counter
from pathlib import Path

from common.schema import RoomRecord, read_jsonl, write_jsonl
from common.taxonomy import load_taxonomy

SECTOR_COMPAT: dict[str, set[str]] = {
    "forest": {"forest"}, "field": {"grassland"},
    "hills": {"grassland", "mountain"}, "mountain": {"mountain", "snow"},
    "water_swim": {"water"}, "water_noswim": {"water"},
    "underwater": {"underwater"}, "desert": {"desert"}, "swamp": {"swamp"},
    "city": {"settlement", "urban", "road", "indoor"},
    "inside": {"indoor", "spacecraft", "cave"},
}


def merge_labels(records: list[RoomRecord], labels: list[dict]
                 ) -> tuple[list[RoomRecord], list[dict]]:
    tax = load_taxonomy()
    by_id = {r.id: r for r in records}
    labeled, queue = [], []
    for lab in labels:
        rec = by_id.get(lab["id"])
        if rec is None:
            raise ValueError(f"label for unknown id {lab['id']}")
        tax.validate_label(lab["environment"], lab.get("modifiers", []))
        rec.environment = lab["environment"]
        rec.modifiers = lab.get("modifiers", [])
        labeled.append(rec)
        compat = SECTOR_COMPAT.get(rec.sector_hint)
        conflict = compat is not None and lab["environment"] not in compat
        if lab.get("ambiguous") or lab.get("confidence") == "low" or conflict:
            queue.append({**lab, "sector_conflict": bool(conflict),
                          "sector_hint": rec.sector_hint})
    return labeled, queue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    a = ap.parse_args()
    records = read_jsonl(Path(f"data/dedup/{a.split}.jsonl"))
    labels: list[dict] = []
    for f in sorted(Path(f"data/labeling/{a.split}").glob("labels_*.jsonl")):
        labels.extend(json.loads(x) for x in f.read_text().splitlines() if x)
    labeled, queue = merge_labels(records, labels)
    write_jsonl(Path(f"data/labeled/{a.split}.jsonl"), labeled)
    with open(f"data/labeled/{a.split}_review.jsonl", "w") as f:
        for q in queue:
            f.write(json.dumps(q) + "\n")
    print(f"labeled={len(labeled)} review={len(queue)}")
    print(Counter(r.environment for r in labeled).most_common())


if __name__ == "__main__":
    main()
