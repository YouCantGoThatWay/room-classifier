import json
import subprocess
from pathlib import Path

from sources.download import SOURCES, clone_all


def test_registry_tiers_and_parsers():
    assert set(SOURCES) == {"tbamud", "coffeemud", "smaug", "rom",
                            "awakemud", "swfote"}
    assert SOURCES["tbamud"].tier == "train"
    assert SOURCES["coffeemud"].tier == "train"
    for grey in ("smaug", "rom", "awakemud", "swfote"):
        assert SOURCES[grey].tier == "eval"
    assert SOURCES["awakemud"].parser == "wld"
    assert SOURCES["swfote"].parser == "are"


def _make_local_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-qm", "init"], check=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def test_clone_all_writes_pins(tmp_path: Path, monkeypatch):
    sha = _make_local_repo(tmp_path / "fakeremote")
    import sources.download as dl
    monkeypatch.setitem(dl.SOURCES, "tbamud",
                        dl.SOURCES["tbamud"].__class__(
                            name="tbamud", git_url=str(tmp_path / "fakeremote"),
                            tier="train", license_note="test",
                            world_glob="*.txt", parser="wld"))
    dest = tmp_path / "raw"
    pins = clone_all(dest, only=["tbamud"])
    assert pins == {"tbamud": sha}
    assert json.loads((dest / "PINS.json").read_text())["tbamud"] == sha
    # idempotent second run, still no other sources touched
    assert clone_all(dest, only=["tbamud"]) == {"tbamud": sha}
