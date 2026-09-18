"""Validation/merge for Claude-generated synthetic room descriptions."""
import json
from collections import Counter
from pathlib import Path

from common.schema import RoomRecord, read_jsonl, write_jsonl
from common.taxonomy import load_taxonomy
from common.textclean import is_trivial, normalized_key


def class_deficits(labeled: list[RoomRecord], target: int) -> dict[str, int]:
    counts = Counter(r.environment for r in labeled)
    tax = load_taxonomy()
    return {b: max(0, target - counts.get(b, 0))
            for b in tax.bases if b != "unknown"}


def validate_synthetic(items: list[dict]) -> list[RoomRecord]:
    tax = load_taxonomy()
    seen: set[str] = set()
    counters: Counter = Counter()
    out: list[RoomRecord] = []
    for it in items:
        env = it["environment"]
        tax.validate_label(env, it.get("modifiers", []))
        if is_trivial(it["name"], it["description"]):
            raise ValueError(f"trivial synthetic description: {it['name']!r}")
        key = normalized_key(it["name"], it["description"])
        if key in seen:
            raise ValueError(f"duplicate synthetic room: {it['name']!r}")
        seen.add(key)
        rec = RoomRecord(
            id=f"synthetic:{env}:{counters[env]}", source="synthetic",
            tier="train", world="synthetic", area=f"synthetic:{env}",
            name=it["name"], description=it["description"],
            license="original, generated", synthetic=True,
            parent_id=it.get("parent_id"), environment=env,
            modifiers=it.get("modifiers", []))
        rec.validate()
        counters[env] += 1
        out.append(rec)
    return out


def main() -> None:
    raw = Path("data/labeling/synthetic_raw.jsonl")
    items = [json.loads(x) for x in raw.read_text().splitlines() if x.strip()]
    recs = validate_synthetic(items)
    labeled_path = Path("data/labeled/train.jsonl")
    existing = read_jsonl(labeled_path)
    write_jsonl(labeled_path, existing + recs)
    print(f"appended {len(recs)} synthetic records "
          f"({Counter(r.environment for r in recs).most_common()})")


if __name__ == "__main__":
    main()
