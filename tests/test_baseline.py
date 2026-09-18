import pytest

from common.schema import RoomRecord
from training.baseline import predict_with_proba, train_baseline

FOREST = ["Tall pines crowd the narrow trail beneath a green canopy.",
          "Oaks and birches surround a mossy clearing full of ferns.",
          "The forest floor is thick with needles and fallen branches."]
DESERT = ["Endless dunes roll away beneath a burning merciless sun.",
          "Cracked sand stretches to the horizon; nothing grows here.",
          "Wind-carved dunes of red sand shimmer in the desert heat."]


def rec(i, desc, env):
    return RoomRecord(id=f"s:a:{i}", source="s", tier="train", world="s",
                      area="a", name="Room", description=desc,
                      environment=env)


def _corpus():
    recs = [rec(i, d, "forest") for i, d in enumerate(FOREST)]
    recs += [rec(10 + i, d, "desert") for i, d in enumerate(DESERT)]
    return recs


def test_baseline_learns_toy_corpus():
    model = train_baseline(_corpus())
    preds = predict_with_proba(model, [
        rec(99, "Pines and oaks form a dense green canopy overhead.",
            "forest"),
        rec(98, "Scorching sand dunes stretch toward the empty horizon.",
            "desert")])
    assert preds[0][0] == "forest" and preds[1][0] == "desert"
    assert all(0.0 < p <= 1.0 for _, p in preds)


def test_train_baseline_rejects_eval_tier_records():
    recs = _corpus()
    recs[0] = RoomRecord(id="s:a:eval0", source="s", tier="eval", world="s",
                         area="a", name="Room", description=recs[0].description,
                         environment=recs[0].environment)
    with pytest.raises(ValueError, match="eval-tier record"):
        train_baseline(recs)
