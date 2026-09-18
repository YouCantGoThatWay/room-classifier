import pytest
from common.schema import RoomRecord
from training.split import assign_split, split_records


def rec(id, source="s", area="a", tier="train", synthetic=False,
        parent_id=None, environment="forest"):
    return RoomRecord(id=id, source=source, tier=tier, world=source,
                      area=area, name="n", description="d" * 40,
                      synthetic=synthetic, parent_id=parent_id,
                      environment=environment)


def test_split_is_by_area_and_deterministic():
    a = [assign_split(rec(f"s:a:{i}", area="a")) for i in range(50)]
    assert len(set(a)) == 1  # whole area goes to one partition
    assert assign_split(rec("x", area="a")) == a[0]


def test_synthetic_follows_parent():
    parent = rec("s:zone9:5", area="zone9")
    child = rec("synthetic:forest:0", source="synthetic",
                area="synthetic:forest", synthetic=True,
                parent_id="s:zone9:5")
    assert assign_split(child) == assign_split(parent)


def test_parentless_synthetic_always_train():
    child = rec("synthetic:forest:1", source="synthetic",
                area="synthetic:forest", synthetic=True)
    assert assign_split(child) == "train"


def test_split_records_rejects_eval_tier_and_unlabeled():
    with pytest.raises(ValueError, match="eval"):
        split_records([rec("e:1", tier="eval")])
    with pytest.raises(ValueError, match="environment"):
        split_records([rec("s:a:1", environment=None)])


def test_split_records_partitions_everything():
    recs = [rec(f"s:z{i}:{j}", area=f"z{i}")
            for i in range(30) for j in range(3)]
    parts = split_records(recs)
    assert sum(len(v) for v in parts.values()) == len(recs)
    assert parts["train"] and parts["val"] and parts["test"]
