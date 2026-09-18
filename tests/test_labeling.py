import json
from pathlib import Path

from common.schema import RoomRecord
from labeling.batches import make_batches
from labeling.merge import merge_labels


def rec(i, sector=""):
    return RoomRecord(id=f"s:a:{i}", source="s", tier="train", world="s",
                      area="a", name=f"Room {i}",
                      description="A long enough description of this room.",
                      sector_hint=sector)


def test_make_batches_deterministic_and_capped(tmp_path: Path):
    recs = [rec(i) for i in range(100)]
    n = make_batches(recs, tmp_path, size=40, cap=80)
    assert n == 2
    b0 = json.loads((tmp_path / "batch_000.json").read_text())
    assert len(b0["rooms"]) == 40
    assert set(b0["rooms"][0]) == {"id", "name", "description", "sector_hint"}
    # same seed → same first id
    tmp2 = tmp_path / "again"
    make_batches(recs, tmp2, size=40, cap=80)
    assert json.loads((tmp2 / "batch_000.json").read_text()) == b0


def label(i, env="forest", conf="high", amb=False):
    return {"id": f"s:a:{i}", "environment": env, "modifiers": [],
            "confidence": conf, "ambiguous": amb, "reason": ""}


def test_merge_applies_labels_and_queues_flags():
    recs = [rec(0), rec(1), rec(2, sector="water_swim")]
    labels = [label(0), label(1, conf="low"), label(2, env="forest")]
    labeled, queue = merge_labels(recs, labels)
    assert labeled[0].environment == "forest"
    queued_ids = {q["id"] for q in queue}
    assert "s:a:1" in queued_ids          # low confidence
    assert "s:a:2" in queued_ids          # sector conflict water vs forest
    assert "s:a:0" not in queued_ids


def test_merge_rejects_bad_taxonomy():
    import pytest
    with pytest.raises(ValueError):
        merge_labels([rec(0)], [label(0, env="jungle")])
