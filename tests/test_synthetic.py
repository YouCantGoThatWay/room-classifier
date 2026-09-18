import pytest
from common.schema import RoomRecord
from labeling.synthetic import assert_no_synthetic, class_deficits, validate_synthetic


def item(env="spacecraft", name="Observation Deck",
         desc="Stars wheel past the curved viewport of the silent deck."):
    return {"name": name, "description": desc, "environment": env,
            "modifiers": [], "parent_id": None}


def test_validate_builds_records():
    recs = validate_synthetic([item(), item(name="Cargo Bay",
                                        desc="Crates line the echoing bay "
                                             "beneath flickering strips.")])
    assert [r.id for r in recs] == ["synthetic:spacecraft:0",
                                    "synthetic:spacecraft:1"]
    assert all(r.synthetic and r.tier == "train" for r in recs)


def test_validate_rejects_bad_env_and_dups():
    with pytest.raises(ValueError):
        validate_synthetic([item(env="hyperspace")])
    with pytest.raises(ValueError):
        validate_synthetic([item(), item()])


def test_class_deficits():
    labeled = [RoomRecord(id=f"s:a:{i}", source="s", tier="train",
                          world="s", area="a", name="n",
                          description="d" * 40, environment="forest")
               for i in range(3)]
    d = class_deficits(labeled, target=5)
    assert d["forest"] == 2 and d["spacecraft"] == 5
    assert "unknown" not in d


def test_assert_no_synthetic_passes_on_clean_set():
    labeled = [RoomRecord(id=f"s:a:{i}", source="s", tier="train",
                          world="s", area="a", name="n",
                          description="d" * 40, environment="forest")
               for i in range(3)]
    assert_no_synthetic(labeled)  # Should not raise


def test_assert_no_synthetic_raises_on_synthetic_records():
    labeled = [RoomRecord(id="synthetic:forest:0", source="synthetic",
                          tier="train", world="synthetic", area="synthetic:forest",
                          name="n", description="d" * 40, environment="forest",
                          synthetic=True)]
    with pytest.raises(SystemExit):
        assert_no_synthetic(labeled)
