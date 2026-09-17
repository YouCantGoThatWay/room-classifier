from pathlib import Path
import pytest
from common.schema import RoomRecord, read_jsonl, write_jsonl


def make_record(**over):
    base = dict(id="tba:30:3001", source="tbamud", tier="train",
                world="tbamud", area="30", name="The Temple",
                description="A vaulted stone hall.")
    base.update(over)
    return RoomRecord(**base)


def test_roundtrip(tmp_path: Path):
    recs = [make_record(), make_record(id="tba:30:3002", synthetic=True,
                                       parent_id="tba:30:3001")]
    p = tmp_path / "out.jsonl"
    assert write_jsonl(p, recs) == 2
    back = read_jsonl(p)
    assert back == recs


def test_validate_rejects_bad_tier():
    with pytest.raises(ValueError, match="tier"):
        make_record(tier="banana").validate()


def test_validate_rejects_empty_id_or_description():
    with pytest.raises(ValueError):
        make_record(id="").validate()
    with pytest.raises(ValueError):
        make_record(description="").validate()


def test_defaults():
    r = make_record()
    assert r.flags == [] and r.modifiers == [] and r.environment is None
    r.validate()
