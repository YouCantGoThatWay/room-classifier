"""Exact + near-duplicate collapse. Eval-tier wins ties (conservative)."""
import hashlib
from collections import defaultdict
from pathlib import Path

from common.schema import RoomRecord, read_jsonl, write_jsonl
from common.textclean import normalized_key

N_BANDS = 4
JACCARD_THRESHOLD = 0.5


def shingles(text: str, k: int = 5) -> set[str]:
    toks = text.split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _band_keys(sh: set[str]) -> list[str]:
    keys = []
    for salt in range(N_BANDS):
        keys.append(min(hashlib.md5(f"{salt}:{s}".encode()).hexdigest()
                        for s in sh) if sh else str(salt))
    return keys


def _merge(group: list[RoomRecord]) -> RoomRecord:
    group.sort(key=lambda r: (r.tier != "eval", r.id))  # eval first
    survivor = group[0]

    # Collect all sources: from group members + from any existing dup_sources flags
    sources_set = {r.source for r in group}
    for r in group:
        for flag in r.flags:
            if flag.startswith("dup_sources:"):
                # Parse "dup_sources:src1,src2,..." into a set
                sources_str = flag[len("dup_sources:"):]
                sources_set.update(sources_str.split(","))

    # Strip all existing dup_sources flags from survivor
    survivor.flags = [f for f in survivor.flags if not f.startswith("dup_sources:")]

    # Append exactly one consolidated flag when >1 source
    if len(sources_set) > 1:
        survivor.flags = survivor.flags + [f"dup_sources:{','.join(sorted(sources_set))}"]

    return survivor


def dedupe(records: list[RoomRecord]) -> list[RoomRecord]:
    # 1. exact
    exact: dict[str, list[RoomRecord]] = defaultdict(list)
    for r in records:
        exact[normalized_key(r.name, r.description)].append(r)
    survivors = [_merge(g) for g in exact.values()]

    # 2. near-dup via banded min-hash candidates
    sh = {r.id: shingles(normalized_key(r.name, r.description))
          for r in survivors}
    buckets: dict[str, list[RoomRecord]] = defaultdict(list)
    for r in survivors:
        for key in _band_keys(sh[r.id]):
            buckets[key].append(r)
    parent: dict[str, str] = {r.id: r.id for r in survivors}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in buckets.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if find(a.id) != find(b.id) and \
                        jaccard(sh[a.id], sh[b.id]) >= JACCARD_THRESHOLD:
                    parent[find(b.id)] = find(a.id)

    clusters: dict[str, list[RoomRecord]] = defaultdict(list)
    for r in survivors:
        clusters[find(r.id)].append(r)
    return sorted((_merge(g) for g in clusters.values()), key=lambda r: r.id)


def main() -> None:
    records: list[RoomRecord] = []
    for f in sorted(Path("data/normalized").glob("*.jsonl")):
        if f.name.startswith("."):
            continue
        records.extend(read_jsonl(f))
    out = dedupe(records)
    train = [r for r in out if r.tier == "train"]
    ev = [r for r in out if r.tier == "eval"]
    write_jsonl(Path("data/dedup/train.jsonl"), train)
    write_jsonl(Path("data/dedup/eval.jsonl"), ev)
    print(f"in={len(records)} out={len(out)} train={len(train)} eval={len(ev)}")


if __name__ == "__main__":
    main()
