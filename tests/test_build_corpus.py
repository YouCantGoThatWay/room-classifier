from pathlib import Path
from sources.build_corpus import build_source
from sources.download import SourceSpec

FIXDIR = Path(__file__).parent / "fixtures"


def _spec(parser, glob):
    return SourceSpec(name="testsrc", git_url="", tier="eval",
                      license_note="test", world_glob=glob, parser=parser)


def test_build_from_wld(tmp_path: Path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "rivenspire.wld").write_text(
        (FIXDIR / "sample.wld").read_text())
    recs = build_source(_spec("wld", "w/*.wld"), tmp_path)
    assert {r.id for r in recs} == {"testsrc:rivenspire:3001",
                                    "testsrc:rivenspire:3054"}
    r = recs[0]
    assert r.tier == "eval" and r.area == "rivenspire"
    assert "\n" not in r.description  # cleaned
    r.validate()


def test_trivial_rooms_dropped(tmp_path: Path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "t.wld").write_text(
        "#1\nVoid~\nx.\n~\n0 0 0\nS\n$~\n")
    assert build_source(_spec("wld", "w/*.wld"), tmp_path) == []
