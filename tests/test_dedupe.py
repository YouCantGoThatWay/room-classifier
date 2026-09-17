from common.schema import RoomRecord
from sources.dedupe import dedupe, jaccard, shingles


def rec(id, tier, name, desc, source="s"):
    return RoomRecord(id=id, source=source, tier=tier, world=source,
                      area="a", name=name, description=desc)


LONG = ("The bridge crosses the wide river from east to west and "
        "traders hurry over its worn planks toward the market square.")


def test_shingles_and_jaccard():
    a = shingles("one two three four five six")
    assert "one two three four five" in a
    assert jaccard(a, a) == 1.0


def test_exact_dup_eval_wins():
    out = dedupe([rec("a:1", "train", "Bridge", LONG, "tba"),
                  rec("b:1", "eval", "Bridge", LONG, "rom")])
    assert len(out) == 1
    assert out[0].tier == "eval"
    assert any(f.startswith("dup_sources:") for f in out[0].flags)


def test_near_dup_collapses():
    tweaked = LONG.replace("worn planks", "old worn planks")
    out = dedupe([rec("a:1", "train", "Bridge", LONG),
                  rec("a:2", "train", "Bridge", tweaked)])
    assert len(out) == 1


def test_distinct_rooms_survive():
    out = dedupe([rec("a:1", "train", "Bridge", LONG),
                  rec("a:2", "train", "Desert",
                      "Endless dunes roll away beneath a burning sun "
                      "toward mountains on the far horizon.")])
    assert len(out) == 2
