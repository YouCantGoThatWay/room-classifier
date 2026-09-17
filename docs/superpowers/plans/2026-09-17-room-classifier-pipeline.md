# Room-Environment Classifier Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end pipeline that downloads and parses MUD world files, labels rooms (Claude in-session), trains a TF-IDF baseline and a SetFit/MiniLM classifier locally, evaluates on unseen-genre worlds, and exports an ONNX model package with C# parity fixtures.

**Architecture:** A `uv`-managed Python 3.12 mono-repo of small CLI stages sharing one JSONL record schema (`common/schema.py`). Data flows `data/raw` → `data/normalized` → `data/dedup` → `data/labeled` → `data/splits` → `data/models` → `export/out`. License tiers (`train`/`eval`) travel on every record and are enforced in code: the split/training loaders raise on any `tier="eval"` record.

**Tech Stack:** Python 3.12, uv, pytest, scikit-learn, SetFit + sentence-transformers (`all-MiniLM-L6-v2`), torch (MPS/CPU), onnx, onnxruntime, HF `tokenizers`.

**Spec:** `docs/superpowers/specs/2026-09-17-room-classifier-pipeline-design.md`

## Global Constraints

- Python 3.12; dependencies managed with `uv`; run everything as `uv run ...`.
- Eval-tier text must never enter training: any loader feeding a trainer raises `ValueError` on `tier == "eval"`.
- `taxonomy.json` is the single source of truth: 15 base classes (`indoor, settlement, road, grassland, forest, desert, mountain, cave, water, underwater, swamp, snow, spacecraft, urban, unknown`), 6 modifiers (`dark, bright, dense, ruined, magical, underground`), version `1.0.0`.
- Synthetic records: `synthetic: true`, never in any eval/test set; parentless synthetic always goes to the train split.
- `data/` is gitignored except `data/README.md`; test fixtures live in `tests/fixtures/` and are committed.
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 256 word-piece limit). Preprocessing text is always `build_text(name, description)` from `common/textclean.py` — every consumer (baseline, SetFit, parity fixtures) uses that one function.
- Labeling and synthetic generation are Claude in-session activities (Tasks 12, 13); code tasks only prepare batches and validate/merge results.
- Commit after every task; messages `feat:`/`test:`/`chore:` style.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.python-version`, `data/README.md`, `common/__init__.py`, `sources/__init__.py`, `labeling/__init__.py`, `training/__init__.py`, `eval/__init__.py`, `export/__init__.py`, `tests/__init__.py`, `tests/fixtures/.gitkeep`

**Interfaces:**
- Produces: the package layout every later task imports (`common`, `sources`, `labeling`, `training`, `eval`, `export`) and a working `uv run pytest`.

- [ ] **Step 1: Initialize uv project**

Run:
```bash
cd "/Volumes/Extreme SSD/workspace/experimental_model"
uv init --python 3.12 --no-readme --name room-classifier .
uv add scikit-learn numpy joblib
uv add setfit torch sentence-transformers
uv add onnx onnxruntime tokenizers
uv add --dev pytest
```
(If `uv init` creates `main.py` or `hello.py`, delete it.)

- [ ] **Step 2: Write pyproject additions**

Append to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: downloads models / long-running; deselect with -m 'not slow'"]

[tool.setuptools]
packages = ["common", "sources", "labeling", "training", "eval", "export"]
```

- [ ] **Step 3: Write `.gitignore` and `data/README.md`**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
data/*
!data/README.md
export/out/
.DS_Store
```

`data/README.md`:
```markdown
Pipeline data directory (gitignored). Stages:
raw/ (cloned repos + PINS.json) → normalized/ → dedup/ → labeling/ →
labeled/ → splits/ → models/ → eval/
```

- [ ] **Step 4: Create empty packages**

Create each listed `__init__.py` as an empty file, plus `tests/fixtures/.gitkeep`.

- [ ] **Step 5: Verify pytest runs**

Run: `uv run pytest`
Expected: `no tests ran` exit cleanly (exit code 5 is fine).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: scaffold uv project and package layout"
```

---

### Task 2: Record schema (`common/schema.py`)

**Files:**
- Create: `common/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `@dataclass RoomRecord(id: str, source: str, tier: str, world: str, area: str, name: str, description: str, sector_hint: str = "", flags: list[str] = [], license: str = "", synthetic: bool = False, parent_id: str | None = None, environment: str | None = None, modifiers: list[str] = [])`
  - `RoomRecord.validate(self) -> None` (raises `ValueError`)
  - `write_jsonl(path: Path, records: Iterable[RoomRecord]) -> int`
  - `read_jsonl(path: Path) -> list[RoomRecord]`

- [ ] **Step 1: Write the failing test**

`tests/test_schema.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError` / import error.

- [ ] **Step 3: Write implementation**

`common/schema.py`:
```python
"""Single JSONL record schema shared by every pipeline stage."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TIERS = ("train", "eval")


@dataclass
class RoomRecord:
    id: str
    source: str
    tier: str
    world: str
    area: str
    name: str
    description: str
    sector_hint: str = ""
    flags: list[str] = field(default_factory=list)
    license: str = ""
    synthetic: bool = False
    parent_id: str | None = None
    environment: str | None = None
    modifiers: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("id must be non-empty")
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if not self.name or not self.description:
            raise ValueError(f"{self.id}: name and description must be non-empty")


def write_jsonl(path: Path, records: Iterable[RoomRecord]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[RoomRecord]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(RoomRecord(**json.loads(line)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schema.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add common/schema.py tests/test_schema.py
git commit -m "feat: RoomRecord schema with jsonl io and validation"
```

---

### Task 3: Text cleaning (`common/textclean.py`)

**Files:**
- Create: `common/textclean.py`
- Test: `tests/test_textclean.py`

**Interfaces:**
- Produces: `strip_codes(text: str) -> str`, `clean_text(text: str) -> str` (strip codes + normalize whitespace), `is_trivial(name: str, description: str) -> bool`, `build_text(name: str, description: str) -> str`, `normalized_key(name: str, description: str) -> str` (lowercased, punctuation/whitespace-collapsed dedupe key).

- [ ] **Step 1: Write the failing test**

`tests/test_textclean.py`:
```python
from common.textclean import (build_text, clean_text, is_trivial,
                              normalized_key, strip_codes)


def test_strip_color_codes():
    assert strip_codes("&RThe &GForest&x") == "The Forest"       # SMAUG
    assert strip_codes("@rDark@n cave") == "Dark cave"           # tba @-codes
    assert strip_codes("{cMisty{x path") == "Misty path"         # ROM
    assert strip_codes("\x1b[31mRed\x1b[0m room") == "Red room"  # raw ANSI


def test_clean_text_normalizes_whitespace_and_tildes():
    raw = "A hall.~\r\n   Dust    hangs\n\nin the air.  "
    assert clean_text(raw) == "A hall. Dust hangs in the air."


def test_is_trivial():
    assert is_trivial("Void", "short")
    assert not is_trivial("Temple", "A vaulted stone hall stretches north.")


def test_build_text():
    assert build_text("Temple", "A hall.") == "Temple\nA hall."


def test_normalized_key_ignores_case_and_punct():
    a = normalized_key("The Temple", "A vaulted, stone hall!")
    b = normalized_key("the temple", "a vaulted stone hall")
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_textclean.py -v` — expected FAIL (import error).

- [ ] **Step 3: Write implementation**

`common/textclean.py`:
```python
"""Cleaning of MUD room text: color codes, tildes, whitespace."""
import re

_CODE_RES = [
    re.compile(r"\x1b\[[0-9;]*m"),   # raw ANSI
    re.compile(r"&[a-zA-Z0-9]"),     # SMAUG &-codes
    re.compile(r"@[a-zA-Z0-9]"),     # tbaMUD @-codes
    re.compile(r"\{[a-zA-Z]"),       # ROM {-codes
]
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")

MIN_DESC_CHARS = 30


def strip_codes(text: str) -> str:
    for rx in _CODE_RES:
        text = rx.sub("", text)
    return text


def clean_text(text: str) -> str:
    text = strip_codes(text).replace("~", "")
    return _WS.sub(" ", text).strip()


def is_trivial(name: str, description: str) -> bool:
    return len(clean_text(description)) < MIN_DESC_CHARS


def build_text(name: str, description: str) -> str:
    """The one canonical model input format. C# must reproduce this."""
    return f"{clean_text(name)}\n{clean_text(description)}"


def normalized_key(name: str, description: str) -> str:
    joined = f"{name} {description}".lower()
    joined = strip_codes(joined).replace("~", "")
    return _WS.sub(" ", _PUNCT.sub("", joined)).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_textclean.py -v` — expected 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add common/textclean.py tests/test_textclean.py
git commit -m "feat: text cleaning, canonical build_text, dedupe key"
```

---

### Task 4: Taxonomy file and loader (`taxonomy.json`, `common/taxonomy.py`)

**Files:**
- Create: `taxonomy.json`, `common/taxonomy.py`
- Test: `tests/test_taxonomy.py`

**Interfaces:**
- Produces: `load_taxonomy(path: Path | None = None) -> Taxonomy` (default path = repo-root `taxonomy.json`); `Taxonomy.bases: list[str]`, `.modifiers: list[str]`, `.version: str`, `.validate_label(environment: str, modifiers: list[str]) -> None` (raises `ValueError`).

- [ ] **Step 1: Write `taxonomy.json`**

```json
{
  "version": "1.0.0",
  "bases": ["indoor", "settlement", "road", "grassland", "forest",
            "desert", "mountain", "cave", "water", "underwater",
            "swamp", "snow", "spacecraft", "urban", "unknown"],
  "modifiers": ["dark", "bright", "dense", "ruined", "magical", "underground"],
  "tie_breaks": [
    "Classify the room the player stands in; scenes through windows, exits, or lore do not count.",
    "Enclosed built space beats terrain: a tavern in a forest is indoor.",
    "spacecraft beats indoor: a station greenhouse is spacecraft.",
    "A bridge is the terrain it crosses (bridge feature tags deferred).",
    "urban = modern/sci-fi city exterior; settlement = pre-modern village/town exterior.",
    "cave containing a lake is cave unless the player is in the water; then water/underwater.",
    "snow beats mountain when snow/ice dominates the description.",
    "unknown only when evidence is genuinely insufficient."
  ]
}
```

- [ ] **Step 2: Write the failing test**

`tests/test_taxonomy.py`:
```python
import pytest
from common.taxonomy import load_taxonomy


def test_load_default():
    t = load_taxonomy()
    assert t.version == "1.0.0"
    assert "spacecraft" in t.bases and "unknown" in t.bases
    assert len(t.bases) == 15 and len(t.modifiers) == 6


def test_validate_label():
    t = load_taxonomy()
    t.validate_label("forest", ["dark", "dense"])
    with pytest.raises(ValueError):
        t.validate_label("jungle", [])
    with pytest.raises(ValueError):
        t.validate_label("forest", ["gloomy"])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_taxonomy.py -v` — expected FAIL.

- [ ] **Step 4: Write implementation**

`common/taxonomy.py`:
```python
import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent / "taxonomy.json"


@dataclass
class Taxonomy:
    version: str
    bases: list[str]
    modifiers: list[str]
    tie_breaks: list[str]

    def validate_label(self, environment: str, modifiers: list[str]) -> None:
        if environment not in self.bases:
            raise ValueError(f"unknown environment {environment!r}")
        bad = [m for m in modifiers if m not in self.modifiers]
        if bad:
            raise ValueError(f"unknown modifiers {bad}")


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    data = json.loads((path or _DEFAULT).read_text(encoding="utf-8"))
    return Taxonomy(version=data["version"], bases=data["bases"],
                    modifiers=data["modifiers"],
                    tie_breaks=data.get("tie_breaks", []))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_taxonomy.py -v` — expected 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add taxonomy.json common/taxonomy.py tests/test_taxonomy.py
git commit -m "feat: taxonomy v1.0.0 with loader and label validation"
```

---

### Task 5: CircleMUD/tbaMUD `.wld` parser (`sources/parse_wld.py`)

**Files:**
- Create: `sources/parse_wld.py`, `tests/fixtures/sample.wld`
- Test: `tests/test_parse_wld.py`

**Interfaces:**
- Produces: `parse_wld(text: str) -> list[dict]`, each dict `{"vnum": int, "name": str, "description": str, "sector_hint": str, "flags": list[str]}`; `SECTOR_NAMES: dict[int, str]`.
- Format notes for the implementer: a room starts at `#<vnum>`; next line is the name ending `~`; description lines follow until a line that is exactly `~`; the next line is `zone flags... sector` — **sector is the final integer token** (works for stock Circle and tbaMUD 128-bit variants); skip everything until `S` on its own line ends the room. `$~` or EOF ends the file.

- [ ] **Step 1: Write fixture**

`tests/fixtures/sample.wld`:
```
#3001
The Temple of Midgaard~
You are in the southern end of the temple hall in the temple of Midgaard.
Huge marble pillars rise up to the ceiling far above your head.
~
30 abd 0
D0
~
~
0 -1 3054
S
#3054
On the Bridge~
The bridge crosses the river from east to west.
~
30 0 11
S
$~
```

- [ ] **Step 2: Write the failing test**

`tests/test_parse_wld.py`:
```python
from pathlib import Path
from sources.parse_wld import SECTOR_NAMES, parse_wld

FIXTURE = Path(__file__).parent / "fixtures" / "sample.wld"


def test_parses_rooms():
    rooms = parse_wld(FIXTURE.read_text())
    assert [r["vnum"] for r in rooms] == [3001, 3054]
    t = rooms[0]
    assert t["name"] == "The Temple of Midgaard"
    assert t["description"].startswith("You are in the southern end")
    assert "pillars rise" in t["description"]
    assert t["sector_hint"] == "inside"
    assert t["flags"] == ["abd"]


def test_unknown_sector_maps_to_empty():
    rooms = parse_wld(FIXTURE.read_text())
    assert rooms[1]["sector_hint"] == SECTOR_NAMES.get(11, "")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_parse_wld.py -v` — expected FAIL.

- [ ] **Step 4: Write implementation**

`sources/parse_wld.py`:
```python
"""Parser for CircleMUD / tbaMUD .wld room files (title, desc, sector)."""
SECTOR_NAMES = {
    0: "inside", 1: "city", 2: "field", 3: "forest", 4: "hills",
    5: "mountain", 6: "water_swim", 7: "water_noswim", 8: "flying",
    9: "underwater",
}


def parse_wld(text: str) -> list[dict]:
    lines = text.splitlines()
    rooms, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("#") or line.startswith("$"):
            i += 1
            continue
        try:
            vnum = int(line[1:])
        except ValueError:
            i += 1
            continue
        i += 1
        name = lines[i].rstrip().rstrip("~")
        i += 1
        desc = []
        while i < len(lines) and lines[i].strip() != "~":
            desc.append(lines[i])
            i += 1
        i += 1  # past the lone ~
        sector_hint, flags = "", []
        if i < len(lines):
            toks = lines[i].split()
            if toks:
                try:
                    sector_hint = SECTOR_NAMES.get(int(toks[-1]), "")
                except ValueError:
                    pass
                flags = toks[1:-1]
            i += 1
        while i < len(lines) and lines[i].strip() != "S":
            i += 1
        i += 1
        rooms.append({"vnum": vnum, "name": name,
                      "description": "\n".join(desc),
                      "sector_hint": sector_hint, "flags": flags})
    return rooms
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_parse_wld.py -v` — expected 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add sources/parse_wld.py tests/test_parse_wld.py tests/fixtures/sample.wld
git commit -m "feat: CircleMUD/tbaMUD .wld room parser"
```

---

### Task 6: Merc/SMAUG `.are` parser (`sources/parse_are.py`)

**Files:**
- Create: `sources/parse_are.py`, `tests/fixtures/sample.are`
- Test: `tests/test_parse_are.py`

**Interfaces:**
- Produces: `parse_are_rooms(text: str) -> list[dict]` with the same dict shape as `parse_wld`.
- Format notes: rooms live between a `#ROOMS` line and a `#0` line. Each room: `#<vnum>`, name ending `~`, description until lone `~`, then `<area> <flags> <sector> [extras...]` — **sector is token index 2** (SMAUG may append extras after it). Skip to `S`. Sector numbering matches Merc: reuse `SECTOR_NAMES` from `parse_wld` (Merc adds 10="air", 11="desert" — extend the map here via a merged dict).

- [ ] **Step 1: Write fixture**

`tests/fixtures/sample.are`:
```
#AREA	{ 5 35} Someone  Sample Area~

#ROOMS
#9201
In a Sandy Desert~
Endless dunes roll away in every direction beneath a burning sun.
~
92 0 11
S
#9202
Temple Square~
The great square before the temple bustles with traders.
~
92 8 1 0 0 0
S
#0

#$
```
(The `#AREA` header line and trailing `#$` are intentionally present: the parser must ignore everything outside `#ROOMS`...`#0`.)

- [ ] **Step 2: Write the failing test**

`tests/test_parse_are.py`:
```python
from pathlib import Path
from sources.parse_are import parse_are_rooms

FIXTURE = Path(__file__).parent / "fixtures" / "sample.are"


def test_parses_rooms_section_only():
    rooms = parse_are_rooms(FIXTURE.read_text())
    assert [r["vnum"] for r in rooms] == [9201, 9202]
    assert rooms[0]["sector_hint"] == "desert"
    assert rooms[0]["description"].startswith("Endless dunes")


def test_sector_is_third_token_even_with_extras():
    rooms = parse_are_rooms(FIXTURE.read_text())
    assert rooms[1]["sector_hint"] == "city"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_parse_are.py -v` — expected FAIL.

- [ ] **Step 4: Write implementation**

`sources/parse_are.py`:
```python
"""Parser for Merc / ROM / SMAUG .are files (#ROOMS section only)."""
from sources.parse_wld import SECTOR_NAMES as _BASE

SECTOR_NAMES = {**_BASE, 10: "air", 11: "desert"}


def parse_are_rooms(text: str) -> list[dict]:
    lines = text.splitlines()
    rooms, i, in_rooms = [], 0, False
    while i < len(lines):
        line = lines[i].strip()
        if line == "#ROOMS":
            in_rooms = True
            i += 1
            continue
        if not in_rooms:
            i += 1
            continue
        if line == "#0":
            break
        if not line.startswith("#"):
            i += 1
            continue
        try:
            vnum = int(line[1:])
        except ValueError:
            i += 1
            continue
        i += 1
        name = lines[i].rstrip().rstrip("~")
        i += 1
        desc = []
        while i < len(lines) and lines[i].strip() != "~":
            desc.append(lines[i])
            i += 1
        i += 1
        sector_hint, flags = "", []
        if i < len(lines):
            toks = lines[i].split()
            if len(toks) >= 3:
                try:
                    sector_hint = SECTOR_NAMES.get(int(toks[2]), "")
                except ValueError:
                    pass
                flags = [toks[1]]
            i += 1
        while i < len(lines) and lines[i].strip() != "S":
            i += 1
        i += 1
        rooms.append({"vnum": vnum, "name": name,
                      "description": "\n".join(desc),
                      "sector_hint": sector_hint, "flags": flags})
    return rooms
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_parse_are.py -v` — expected 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add sources/parse_are.py tests/test_parse_are.py tests/fixtures/sample.are
git commit -m "feat: Merc/SMAUG .are rooms parser"
```

---

### Task 7: Source registry and downloader (`sources/download.py`)

**Files:**
- Create: `sources/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Produces: `SOURCES: dict[str, SourceSpec]` where `@dataclass SourceSpec(name: str, git_url: str, tier: str, license_note: str, world_glob: str, parser: str)` (`parser` in `{"wld","are","coffeemud"}`); `clone_all(dest: Path, only: list[str] | None = None) -> dict[str, str]` — clones each repo shallowly into `dest/<name>`, returns `{name: head_sha}` and writes `dest/PINS.json`. Re-running with an existing clone reuses it (records its current SHA, no network).

- [ ] **Step 1: Write the failing test** (no network — tests registry + pin file writing with a fake repo)

`tests/test_download.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v` — expected FAIL.

- [ ] **Step 3: Write implementation**

`sources/download.py`:
```python
"""Source registry and repo cloner. Pins = SHA recorded at acquisition."""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    name: str
    git_url: str
    tier: str          # "train" | "eval"
    license_note: str
    world_glob: str    # glob under the repo root for world files
    parser: str        # "wld" | "are" | "coffeemud"


SOURCES: dict[str, SourceSpec] = {
    "tbamud": SourceSpec("tbamud", "https://github.com/tbamud/tbamud.git",
                         "train", "CircleMUD/DikuMUD, LGPL since 2020",
                         "lib/world/wld/*.wld", "wld"),
    "coffeemud": SourceSpec("coffeemud",
                            "https://github.com/bozimmerman/CoffeeMud.git",
                            "train", "Apache-2.0",
                            "resources/**/*.cmare", "coffeemud"),
    "smaug": SourceSpec("smaug", "https://github.com/smaugmuds/_smaug_.git",
                        "eval", "Diku/Merc/SMAUG non-commercial chain",
                        "**/area/*.are", "are"),
    "rom": SourceSpec("rom", "https://github.com/avinson/rom24-quickmud.git",
                      "eval", "Diku/Merc/ROM non-commercial chain",
                      "area/*.are", "are"),
    "awakemud": SourceSpec("awakemud",
                           "https://github.com/luciensadi/AwakeMUD.git",
                           "eval", "Circle lineage + Shadowrun fan IP",
                           "world/**/*.wld", "wld"),
    "swfote": SourceSpec("swfote", "https://github.com/Arthmoor/SWFotE.git",
                         "eval", "SMAUG lineage + Star Wars fan IP",
                         "area/*.are", "are"),
}


def clone_all(dest: Path, only: list[str] | None = None) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    pins: dict[str, str] = {}
    if (dest / "PINS.json").exists():
        pins = json.loads((dest / "PINS.json").read_text())
    for name, spec in SOURCES.items():
        if only and name not in only:
            continue
        repo = dest / name
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", spec.git_url,
                            str(repo)], check=True)
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             check=True).stdout.strip()
        pins[name] = sha
    (dest / "PINS.json").write_text(json.dumps(pins, indent=2))
    return {k: v for k, v in pins.items() if not only or k in only}


if __name__ == "__main__":
    import sys
    result = clone_all(Path("data/raw"),
                       only=sys.argv[1:] or None)
    print(json.dumps(result, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v` — expected 2 PASS.

- [ ] **Step 5: Clone the real repos (network)**

Run: `uv run python -m sources.download`
Expected: six repos under `data/raw/`, `data/raw/PINS.json` with six SHAs. If a URL 404s (repos do move), find the current canonical repo for that codebase on GitHub, update `SOURCES`, and note the change in the commit message.

- [ ] **Step 6: Verify world files exist where the globs point**

Run:
```bash
ls data/raw/tbamud/lib/world/wld | head
find data/raw/swfote -name "*.are" | head
find data/raw/coffeemud -name "*.cmare" | head
```
Expected: non-empty listings. If a glob is wrong for the cloned layout, fix `world_glob` in `SOURCES` to match reality and re-run the unit tests.

- [ ] **Step 7: Commit**

```bash
git add sources/download.py tests/test_download.py
git commit -m "feat: source registry with tier metadata and pinned cloning"
```

---

### Task 8: CoffeeMud area parser (`sources/parse_coffeemud.py`)

CoffeeMud `.cmare` files are XML-ish archives whose exact tag names must be
confirmed against the real files cloned in Task 7 — do Step 1 first and adapt.

**Files:**
- Create: `sources/parse_coffeemud.py`, `tests/fixtures/sample.cmare`
- Test: `tests/test_parse_coffeemud.py`

**Interfaces:**
- Produces: `parse_cmare(text: str) -> list[dict]` with the same dict shape as `parse_wld` (`vnum` may be a synthetic int index if CoffeeMud uses string room IDs; put the original ID string into `flags[0]` as `"cmid:<id>"`).

- [ ] **Step 1: Inspect a real file and extract a fixture**

Run:
```bash
FILE=$(find data/raw/coffeemud -name "*.cmare" | head -1); echo "$FILE"
head -c 4000 "$FILE"
```
Read the structure. Expected shape: XML with room elements carrying a
display name and description text (CoffeeMud serializes rooms with tags
like `<AROOM>`/`<ROOM>` containing `<ROOMID>`, `<TITLE>`/`<DISPLAY>`,
`<DESCRIPTION>`, and a locale/class name such as `WoodRoom`,
`CaveRoom`, `CityStreet` — the locale class is the `sector_hint`).
Copy **two rooms' worth** of real structure into
`tests/fixtures/sample.cmare` (trim inventory/mob payloads), keeping the
exact tag names observed. If the observed format differs from the tags
assumed in Step 2's test, adjust test + implementation to the real tags
— the contract (function name, output dict shape) must not change.

- [ ] **Step 2: Write the failing test**

`tests/test_parse_coffeemud.py` (adapt expected literals to the fixture you extracted):
```python
from pathlib import Path
from sources.parse_coffeemud import parse_cmare

FIXTURE = Path(__file__).parent / "fixtures" / "sample.cmare"


def test_parses_rooms():
    rooms = parse_cmare(FIXTURE.read_text(errors="replace"))
    assert len(rooms) == 2
    for r in rooms:
        assert r["name"] and r["description"]
        assert r["flags"][0].startswith("cmid:")
        assert isinstance(r["vnum"], int)


def test_sector_hint_from_locale_class():
    rooms = parse_cmare(FIXTURE.read_text(errors="replace"))
    assert all(r["sector_hint"] != "" for r in rooms)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_parse_coffeemud.py -v` — expected FAIL.

- [ ] **Step 4: Write implementation** (regex-based, tolerant of the container format; adapt tag names to the fixture)

`sources/parse_coffeemud.py`:
```python
"""Parser for CoffeeMud .cmare area exports.

.cmare files embed rooms as XML-ish blocks. We extract room id, title,
description, and the Java locale class (e.g. WoodRoom) as sector_hint.
Tag names verified against the cloned CoffeeMud repo (Task 8 step 1).
"""
import html
import re

_ROOM = re.compile(r"<AROOM>(.*?)</AROOM>", re.S | re.I)


def _tag(block: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", block, re.S | re.I)
    return html.unescape(m.group(1)).strip() if m else ""


def parse_cmare(text: str) -> list[dict]:
    rooms = []
    for idx, m in enumerate(_ROOM.finditer(text)):
        block = m.group(1)
        rid = _tag(block, "ROOMID")
        name = _tag(block, "DISPLAY") or _tag(block, "TITLE")
        desc = _tag(block, "DESCRIPTION")
        locale = _tag(block, "RCLAS") or _tag(block, "CLASS")
        if not name or not desc:
            continue
        rooms.append({"vnum": idx, "name": name, "description": desc,
                      "sector_hint": locale,
                      "flags": [f"cmid:{rid or idx}"]})
    return rooms
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_parse_coffeemud.py -v` — expected 2 PASS.

- [ ] **Step 6: Smoke-run against the real repo**

Run:
```bash
uv run python -c "
from pathlib import Path
from sources.parse_coffeemud import parse_cmare
files = list(Path('data/raw/coffeemud').rglob('*.cmare'))[:5]
for f in files:
    rooms = parse_cmare(f.read_text(errors='replace'))
    print(f.name, len(rooms))
"
```
Expected: several files with a plausible (non-zero) room count. Zero
across all files means wrong tag names — go back to Step 1.

- [ ] **Step 7: Commit**

```bash
git add sources/parse_coffeemud.py tests/test_parse_coffeemud.py tests/fixtures/sample.cmare
git commit -m "feat: CoffeeMud .cmare room parser"
```

---

### Task 9: Corpus builder (`sources/build_corpus.py`)

**Files:**
- Create: `sources/build_corpus.py`
- Test: `tests/test_build_corpus.py`

**Interfaces:**
- Consumes: `SOURCES`, the three parsers, `RoomRecord`, `clean_text`, `is_trivial`.
- Produces: `build_source(spec: SourceSpec, repo_dir: Path) -> list[RoomRecord]`; CLI `python -m sources.build_corpus` reads `data/raw/`, writes `data/normalized/<name>.jsonl` per source and prints per-source counts. Record id format: `"{source}:{area}:{vnum}"`, `area` = world file stem, `world` = source name. Descriptions cleaned with `clean_text`; trivial rooms dropped; every record `.validate()`d.

- [ ] **Step 1: Write the failing test**

`tests/test_build_corpus.py`:
```python
from pathlib import Path
from sources.build_corpus import build_source
from sources.download import SourceSpec

FIXDIR = Path(__file__).parent / "fixtures"


def _spec(parser, glob):
    return SourceSpec(name="testsrc", git_url="", tier="eval",
                      license_note="test", world_glob=glob, parser=parser)


def test_build_from_wld(tmp_path: Path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "midgaard.wld").write_text(
        (FIXDIR / "sample.wld").read_text())
    recs = build_source(_spec("wld", "w/*.wld"), tmp_path)
    assert {r.id for r in recs} == {"testsrc:midgaard:3001",
                                    "testsrc:midgaard:3054"}
    r = recs[0]
    assert r.tier == "eval" and r.area == "midgaard"
    assert "\n" not in r.description  # cleaned
    r.validate()


def test_trivial_rooms_dropped(tmp_path: Path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "t.wld").write_text(
        "#1\nVoid~\nx.\n~\n0 0 0\nS\n$~\n")
    assert build_source(_spec("wld", "w/*.wld"), tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_build_corpus.py -v` — expected FAIL.

- [ ] **Step 3: Write implementation**

`sources/build_corpus.py`:
```python
"""Parse cloned repos into normalized RoomRecord jsonl per source."""
from pathlib import Path

from common.schema import RoomRecord, write_jsonl
from common.textclean import clean_text, is_trivial
from sources.download import SOURCES, SourceSpec
from sources.parse_are import parse_are_rooms
from sources.parse_coffeemud import parse_cmare
from sources.parse_wld import parse_wld

_PARSERS = {"wld": parse_wld, "are": parse_are_rooms,
            "coffeemud": parse_cmare}


def build_source(spec: SourceSpec, repo_dir: Path) -> list[RoomRecord]:
    parse = _PARSERS[spec.parser]
    records: list[RoomRecord] = []
    for f in sorted(repo_dir.glob(spec.world_glob)):
        area = f.stem
        for room in parse(f.read_text(errors="replace")):
            name = clean_text(room["name"])
            desc = clean_text(room["description"])
            if not name or is_trivial(name, desc):
                continue
            rec = RoomRecord(
                id=f"{spec.name}:{area}:{room['vnum']}",
                source=spec.name, tier=spec.tier, world=spec.name,
                area=area, name=name, description=desc,
                sector_hint=room["sector_hint"], flags=room["flags"],
                license=spec.license_note)
            rec.validate()
            records.append(rec)
    return records


def main(raw_dir: Path = Path("data/raw"),
         out_dir: Path = Path("data/normalized")) -> None:
    for name, spec in SOURCES.items():
        repo = raw_dir / name
        if not repo.exists():
            print(f"{name}: SKIP (not cloned)")
            continue
        recs = build_source(spec, repo)
        n = write_jsonl(out_dir / f"{name}.jsonl", recs)
        print(f"{name}: {n} rooms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_build_corpus.py -v` — expected 2 PASS.

- [ ] **Step 5: Run for real and sanity-check counts**

Run: `uv run python -m sources.build_corpus`
Expected: thousands of rooms for tbamud/coffeemud; hundreds-to-thousands
per grey source; no source at 0. A 0 means the `world_glob` doesn't
match the cloned layout — fix the glob in `SOURCES` and re-run.

- [ ] **Step 6: Commit**

```bash
git add sources/build_corpus.py tests/test_build_corpus.py
git commit -m "feat: corpus builder producing normalized per-source jsonl"
```

---

### Task 10: Dedupe (`sources/dedupe.py`)

**Files:**
- Create: `sources/dedupe.py`
- Test: `tests/test_dedupe.py`

**Interfaces:**
- Consumes: `RoomRecord`, `normalized_key` from `common.textclean`.
- Produces: `dedupe(records: list[RoomRecord]) -> list[RoomRecord]` — exact dedupe on `normalized_key(name, description)`, then near-dup detection (5-token shingles, 4 salted min-hash bucket keys, Jaccard ≥ 0.8 within buckets). Duplicate groups keep one canonical record; **if any member is eval-tier the survivor is eval-tier** (eval wins); survivor's `flags` gains `"dup_sources:<comma-list>"` when merged from multiple sources. CLI writes `data/dedup/train.jsonl` and `data/dedup/eval.jsonl`.

- [ ] **Step 1: Write the failing test**

`tests/test_dedupe.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dedupe.py -v` — expected FAIL.

- [ ] **Step 3: Write implementation**

`sources/dedupe.py`:
```python
"""Exact + near-duplicate collapse. Eval-tier wins ties (conservative)."""
import hashlib
from collections import defaultdict
from pathlib import Path

from common.schema import RoomRecord, read_jsonl, write_jsonl
from common.textclean import normalized_key

N_BANDS = 4
JACCARD_THRESHOLD = 0.8


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
    sources = sorted({r.source for r in group})
    if len(sources) > 1:
        survivor.flags = survivor.flags + [f"dup_sources:{','.join(sources)}"]
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
        records.extend(read_jsonl(f))
    out = dedupe(records)
    train = [r for r in out if r.tier == "train"]
    ev = [r for r in out if r.tier == "eval"]
    write_jsonl(Path("data/dedup/train.jsonl"), train)
    write_jsonl(Path("data/dedup/eval.jsonl"), ev)
    print(f"in={len(records)} out={len(out)} train={len(train)} eval={len(ev)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dedupe.py -v` — expected 4 PASS.

- [ ] **Step 5: Run for real**

Run: `uv run python -m sources.dedupe`
Expected: `out` noticeably below `in` (stock areas are widely copied);
both output files non-empty.

- [ ] **Step 6: Commit**

```bash
git add sources/dedupe.py tests/test_dedupe.py
git commit -m "feat: exact and near-dup collapse with eval-wins tiering"
```

---

### Task 11: Labeling batch prep and merge (`labeling/batches.py`, `labeling/merge.py`)

**Files:**
- Create: `labeling/batches.py`, `labeling/merge.py`
- Test: `tests/test_labeling.py`

**Interfaces:**
- Consumes: `RoomRecord`, `load_taxonomy`.
- Produces:
  - `make_batches(records: list[RoomRecord], out_dir: Path, size: int = 40, cap: int | None = None, seed: int = 17) -> int` — deterministic shuffle; if `cap`, stratified sample proportional to `(source, area)` groups; writes `batch_000.json`... each `{"rooms": [{"id","name","description","sector_hint"}]}`; returns batch count.
  - `merge_labels(records, labels: list[dict]) -> tuple[list[RoomRecord], list[dict]]` — label dicts are `{"id","environment","modifiers","confidence","ambiguous","reason"}` (`confidence` in `{"high","low"}`); validates against taxonomy; sets `environment`/`modifiers` on the matching record; returns `(labeled_records, review_queue)` where the review queue contains label dicts that are `ambiguous`, `low` confidence, or **sector-conflicted** per `SECTOR_COMPAT`.
  - `SECTOR_COMPAT: dict[str, set[str]]` — e.g. `{"forest": {"forest"}, "field": {"grassland"}, "hills": {"grassland","mountain"}, "mountain": {"mountain","snow"}, "water_swim": {"water"}, "water_noswim": {"water"}, "underwater": {"underwater"}, "city": {"settlement","urban","road","indoor"}, "inside": {"indoor","spacecraft","cave"}, "desert": {"desert"}, "swamp": {"swamp"}}` — an empty/unlisted hint conflicts with nothing.
  - CLI `python -m labeling.batches --split train|eval --cap N` and `python -m labeling.merge --split train|eval` (merge reads all `data/labeling/<split>/labels_*.jsonl`, writes `data/labeled/<split>.jsonl` + `data/labeled/<split>_review.jsonl`, prints class counts).

- [ ] **Step 1: Write the failing test**

`tests/test_labeling.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_labeling.py -v` — expected FAIL.

- [ ] **Step 3: Write `labeling/batches.py`**

```python
"""Prepare deterministic labeling batches for in-session Claude labeling."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common.schema import RoomRecord, read_jsonl


def make_batches(records: list[RoomRecord], out_dir: Path, size: int = 40,
                 cap: int | None = None, seed: int = 17) -> int:
    rng = random.Random(seed)
    if cap is not None and cap < len(records):
        groups: dict[tuple, list[RoomRecord]] = defaultdict(list)
        for r in records:
            groups[(r.source, r.area)].append(r)
        frac = cap / len(records)
        sample: list[RoomRecord] = []
        for g in sorted(groups, key=str):
            members = sorted(groups[g], key=lambda r: r.id)
            k = max(1, round(len(members) * frac))
            sample.extend(rng.sample(members, min(k, len(members))))
        records = sample[:cap]
    records = sorted(records, key=lambda r: r.id)
    rng.shuffle(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(0, len(records), size):
        batch = records[i:i + size]
        payload = {"rooms": [{"id": r.id, "name": r.name,
                              "description": r.description,
                              "sector_hint": r.sector_hint}
                             for r in batch]}
        (out_dir / f"batch_{n:03d}.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False))
        n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--size", type=int, default=40)
    a = ap.parse_args()
    recs = read_jsonl(Path(f"data/dedup/{a.split}.jsonl"))
    n = make_batches(recs, Path(f"data/labeling/{a.split}"),
                     size=a.size, cap=a.cap)
    print(f"{n} batches for {a.split}")
```

- [ ] **Step 4: Write `labeling/merge.py`**

```python
"""Validate + merge Claude label files onto records; build review queue."""
import argparse
import json
from collections import Counter
from pathlib import Path

from common.schema import RoomRecord, read_jsonl, write_jsonl
from common.taxonomy import load_taxonomy

SECTOR_COMPAT: dict[str, set[str]] = {
    "forest": {"forest"}, "field": {"grassland"},
    "hills": {"grassland", "mountain"}, "mountain": {"mountain", "snow"},
    "water_swim": {"water"}, "water_noswim": {"water"},
    "underwater": {"underwater"}, "desert": {"desert"}, "swamp": {"swamp"},
    "city": {"settlement", "urban", "road", "indoor"},
    "inside": {"indoor", "spacecraft", "cave"},
}


def merge_labels(records: list[RoomRecord], labels: list[dict]
                 ) -> tuple[list[RoomRecord], list[dict]]:
    tax = load_taxonomy()
    by_id = {r.id: r for r in records}
    labeled, queue = [], []
    for lab in labels:
        rec = by_id.get(lab["id"])
        if rec is None:
            raise ValueError(f"label for unknown id {lab['id']}")
        tax.validate_label(lab["environment"], lab.get("modifiers", []))
        rec.environment = lab["environment"]
        rec.modifiers = lab.get("modifiers", [])
        labeled.append(rec)
        compat = SECTOR_COMPAT.get(rec.sector_hint)
        conflict = compat is not None and lab["environment"] not in compat
        if lab.get("ambiguous") or lab.get("confidence") == "low" or conflict:
            queue.append({**lab, "sector_conflict": bool(conflict),
                          "sector_hint": rec.sector_hint})
    return labeled, queue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    a = ap.parse_args()
    records = read_jsonl(Path(f"data/dedup/{a.split}.jsonl"))
    labels: list[dict] = []
    for f in sorted(Path(f"data/labeling/{a.split}").glob("labels_*.jsonl")):
        labels.extend(json.loads(x) for x in f.read_text().splitlines() if x)
    labeled, queue = merge_labels(records, labels)
    write_jsonl(Path(f"data/labeled/{a.split}.jsonl"), labeled)
    with open(f"data/labeled/{a.split}_review.jsonl", "w") as f:
        for q in queue:
            f.write(json.dumps(q) + "\n")
    print(f"labeled={len(labeled)} review={len(queue)}")
    print(Counter(r.environment for r in labeled).most_common())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_labeling.py -v` — expected 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add labeling/ tests/test_labeling.py
git commit -m "feat: labeling batch prep and label merge with review queue"
```

---

### Task 12: ACTIVITY — Claude in-session labeling

Not a coding task; performed by the orchestrating Claude session (labels
must NOT be produced by a fresh context-free subagent unless the batch
file plus the instructions below are its entire prompt).

- [ ] **Step 1: Generate batches**

```bash
uv run python -m labeling.batches --split train --cap 8000
uv run python -m labeling.batches --split eval --cap 1500
```

- [ ] **Step 2: Label every batch**

For each `data/labeling/<split>/batch_NNN.json`, read the rooms and write
`data/labeling/<split>/labels_NNN.jsonl` — one JSON object per room:
`{"id", "environment", "modifiers", "confidence", "ambiguous", "reason"}`.
Labeling instructions (apply exactly):
- Use only `taxonomy.json` bases/modifiers; apply its `tie_breaks` verbatim.
- Classify the room the player stands in; ignore views through windows,
  exits, or lore mentions.
- `sector_hint` is weak evidence — text wins when they disagree.
- `confidence: "low"` whenever a second reader could defensibly pick a
  different base class; `ambiguous: true` + one-line `reason` when the
  text itself underdetermines the class.
- `environment: "unknown"` only for genuinely insufficient evidence.

- [ ] **Step 3: Merge and validate**

```bash
uv run python -m labeling.merge --split train
uv run python -m labeling.merge --split eval
```
Expected: class-count table printed; merge fails loudly on any label
outside the taxonomy (fix the label file, re-run).

- [ ] **Step 4: Hand the review queue to the user**

Report review-queue sizes and the class-count table to the user; they
review `data/labeled/*_review.jsonl` at their leisure. Do not block the
pipeline on it — corrections re-run the merge.

- [ ] **Step 5: Commit** (scripts/config only — data/ stays gitignored)

```bash
git add -A && git commit -m "chore: record labeling run configuration" --allow-empty
```

---

### Task 13: Synthetic augmentation (`labeling/synthetic.py` + ACTIVITY)

**Files:**
- Create: `labeling/synthetic.py`
- Test: `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `RoomRecord`, `load_taxonomy`, `is_trivial`.
- Produces: `validate_synthetic(items: list[dict]) -> list[RoomRecord]` — input dicts `{"name","description","environment","modifiers","parent_id"}`; builds records with `id="synthetic:<env>:<n>"`, `source="synthetic"`, `tier="train"`, `world="synthetic"`, `area="synthetic:<env>"`, `synthetic=True`, `license="original, generated"`; rejects taxonomy violations, trivial descriptions, and duplicate names+descriptions within the input. CLI `python -m labeling.synthetic` reads `data/labeling/synthetic_raw.jsonl`, appends validated records to `data/labeled/train.jsonl`, prints counts. `class_deficits(labeled: list[RoomRecord], target: int) -> dict[str, int]` reports how many examples each base class is short of `target`.

- [ ] **Step 1: Write the failing test**

`tests/test_synthetic.py`:
```python
import pytest
from common.schema import RoomRecord
from labeling.synthetic import class_deficits, validate_synthetic


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_synthetic.py -v` — expected FAIL.

- [ ] **Step 3: Write implementation**

`labeling/synthetic.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_synthetic.py -v` — expected 3 PASS.

- [ ] **Step 5: ACTIVITY — generate synthetic rooms**

After Task 12's class counts: compute deficits
(`uv run python -c "from common.schema import read_jsonl; from labeling.synthetic import class_deficits; from pathlib import Path; print(class_deficits(read_jsonl(Path('data/labeled/train.jsonl')), 150))"`).
Claude writes `data/labeling/synthetic_raw.jsonl` covering the deficits:
original descriptions (never copied or lightly paraphrased from any
source), varied genre/length/wording/explicitness, including hard
negatives (indoor room with a painted forest, bridge over water labeled
by the crossed terrain, cave with a lake, spacecraft greenhouse). Then:

```bash
uv run python -m labeling.synthetic
```

- [ ] **Step 6: Commit**

```bash
git add labeling/synthetic.py tests/test_synthetic.py
git commit -m "feat: synthetic augmentation validation and merge"
```

---

### Task 14: Train/val/test split (`training/split.py`)

**Files:**
- Create: `training/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: `RoomRecord`.
- Produces: `assign_split(rec: RoomRecord) -> str` — group key `f"{rec.source}:{rec.area}"` hashed (`md5 % 10`): 0→`"test"`, 1→`"val"`, else `"train"`. Synthetic with `parent_id` inherits the parent's group key (`parent_id` format `source:area:vnum` → key from its first two segments); parentless synthetic → always `"train"`. `split_records(records: list[RoomRecord]) -> dict[str, list[RoomRecord]]` — **raises `ValueError` on any `tier == "eval"` record** (the enforcement point) and on any record without `environment`; asserts no synthetic record lands in val/test. CLI reads `data/labeled/train.jsonl`, writes `data/splits/{train,val,test}.jsonl`.

- [ ] **Step 1: Write the failing test**

`tests/test_split.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split.py -v` — expected FAIL.

- [ ] **Step 3: Write implementation**

`training/split.py`:
```python
"""World/area-grouped split. THE tier-enforcement point for training."""
import hashlib
from pathlib import Path

from common.schema import RoomRecord, read_jsonl, write_jsonl


def _group_key(rec: RoomRecord) -> str:
    if rec.synthetic:
        if rec.parent_id:
            parts = rec.parent_id.split(":")
            return f"{parts[0]}:{parts[1]}"
        return "__synthetic_train__"
    return f"{rec.source}:{rec.area}"


def assign_split(rec: RoomRecord) -> str:
    key = _group_key(rec)
    if key == "__synthetic_train__":
        return "train"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10
    return {0: "test", 1: "val"}.get(h, "train")


def split_records(records: list[RoomRecord]) -> dict[str, list[RoomRecord]]:
    parts: dict[str, list[RoomRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        if r.tier == "eval":
            raise ValueError(f"{r.id}: eval-tier record in training path")
        if not r.environment:
            raise ValueError(f"{r.id}: missing environment label")
        part = assign_split(r)
        if r.synthetic and part != "train":
            raise AssertionError(f"{r.id}: synthetic record in {part}")
        parts[part].append(r)
    return parts


def main() -> None:
    records = read_jsonl(Path("data/labeled/train.jsonl"))
    parts = split_records(records)
    for name, recs in parts.items():
        write_jsonl(Path(f"data/splits/{name}.jsonl"), recs)
        print(f"{name}: {len(recs)}")


if __name__ == "__main__":
    main()
```

Note: a synthetic record whose parent's area hashes to val/test would
trip the `AssertionError` by design — the fix at data time is to drop or
re-parent it, never to weaken the assertion. During the ACTIVITY tasks,
generate parented synthetics only from train-partition parents (check
with `assign_split` first).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split.py -v` — expected 5 PASS.

- [ ] **Step 5: Run for real**

Run: `uv run python -m training.split`
Expected: roughly 80/10/10 by area; all three non-empty.

- [ ] **Step 6: Commit**

```bash
git add training/split.py tests/test_split.py
git commit -m "feat: area-grouped split with eval-tier enforcement"
```

---

### Task 15: TF-IDF baseline (`training/baseline.py`)

**Files:**
- Create: `training/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `RoomRecord`, `build_text`, splits from Task 14.
- Produces: `train_baseline(train: list[RoomRecord]) -> sklearn.pipeline.Pipeline` (TfidfVectorizer(1-2 grams, min_df=2) + LogisticRegression(max_iter=1000, class_weight="balanced")); `predict_with_proba(model, records) -> list[tuple[str, float]]` ((label, max-probability) per record, usable identically for baseline and SetFit via duck-typed `predict_proba`/`classes_`). CLI trains on `data/splits/train.jsonl`, saves `data/models/baseline.joblib`, prints val macro-F1.

- [ ] **Step 1: Write the failing test**

`tests/test_baseline.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baseline.py -v` — expected FAIL.

- [ ] **Step 3: Write implementation**

`training/baseline.py`:
```python
"""TF-IDF + logistic regression baseline. The number to beat."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from common.schema import RoomRecord, read_jsonl
from common.textclean import build_text


def _texts(records: list[RoomRecord]) -> list[str]:
    return [build_text(r.name, r.description) for r in records]


def train_baseline(train: list[RoomRecord]) -> Pipeline:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(_texts(train), [r.environment for r in train])
    return model


def predict_with_proba(model, records: list[RoomRecord]
                       ) -> list[tuple[str, float]]:
    proba = model.predict_proba(_texts(records))
    classes = list(model.classes_)
    return [(classes[row.argmax()], float(row.max())) for row in proba]


def main() -> None:
    train = read_jsonl(Path("data/splits/train.jsonl"))
    val = read_jsonl(Path("data/splits/val.jsonl"))
    model = train_baseline(train)
    preds = [p for p, _ in predict_with_proba(model, val)]
    macro = f1_score([r.environment for r in val], preds, average="macro")
    Path("data/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "data/models/baseline.joblib")
    print(f"val macro-F1: {macro:.3f}  (n_train={len(train)}, n_val={len(val)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_baseline.py -v` — expected 1 PASS.

- [ ] **Step 5: Train for real**

Run: `uv run python -m training.baseline`
Expected: a val macro-F1 printed; record the number — it is the bar
SetFit must clear.

- [ ] **Step 6: Commit**

```bash
git add training/baseline.py tests/test_baseline.py
git commit -m "feat: tf-idf logistic baseline with shared predict interface"
```

---

### Task 16: SetFit training (`training/train_setfit.py`)

**Files:**
- Create: `training/train_setfit.py`
- Test: `tests/test_train_setfit.py` (marked `slow` — downloads MiniLM)

**Interfaces:**
- Consumes: splits, `build_text`, `predict_with_proba` (SetFitModel exposes `predict_proba`; wrap so the eval code is model-agnostic).
- Produces: `train_setfit(train: list[RoomRecord], epochs: int = 1) -> SetFitModel`; `SetFitWrapper` with `.predict_proba(texts)` and `.classes_` so `predict_with_proba` works unchanged; `load_setfit(path) -> SetFitWrapper`. CLI trains, saves to `data/models/setfit/`, prints val macro-F1.

- [ ] **Step 1: Write the failing test**

`tests/test_train_setfit.py`:
```python
import pytest
from tests.test_baseline import _corpus, rec
from training.baseline import predict_with_proba


@pytest.mark.slow
def test_setfit_smoke(tmp_path):
    from training.train_setfit import SetFitWrapper, load_setfit, train_setfit
    model = train_setfit(_corpus(), epochs=1)
    wrapper = SetFitWrapper(model)
    preds = predict_with_proba(wrapper, [
        rec(99, "Pines and oaks form a dense green canopy overhead.",
            "forest")])
    assert preds[0][0] in {"forest", "desert"}
    model.save_pretrained(str(tmp_path / "m"))
    again = load_setfit(tmp_path / "m")
    assert list(again.classes_) == list(wrapper.classes_)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_setfit.py -v -m slow` — expected FAIL (import error).

- [ ] **Step 3: Write implementation**

`training/train_setfit.py`:
```python
"""SetFit fine-tune of all-MiniLM-L6-v2 with a logistic-regression head."""
from pathlib import Path

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score

from common.schema import RoomRecord, read_jsonl
from common.textclean import build_text
from training.baseline import predict_with_proba

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SetFitWrapper:
    """Duck-types sklearn's predict_proba/classes_ for shared eval code."""

    def __init__(self, model: SetFitModel):
        self.model = model
        self.classes_ = list(model.labels) if model.labels else \
            list(model.model_head.classes_)

    def predict_proba(self, texts: list[str]):
        import numpy as np
        return np.asarray(self.model.predict_proba(texts))


def _dataset(records: list[RoomRecord]) -> Dataset:
    return Dataset.from_dict({
        "text": [build_text(r.name, r.description) for r in records],
        "label": [r.environment for r in records]})


def train_setfit(train: list[RoomRecord], epochs: int = 1) -> SetFitModel:
    labels = sorted({r.environment for r in train})
    model = SetFitModel.from_pretrained(BASE_MODEL, labels=labels)
    args = TrainingArguments(batch_size=16, num_epochs=epochs,
                             sampling_strategy="oversampling")
    Trainer(model=model, args=args, train_dataset=_dataset(train)).train()
    return model


def load_setfit(path: Path) -> SetFitWrapper:
    return SetFitWrapper(SetFitModel.from_pretrained(str(path)))


def main() -> None:
    train = read_jsonl(Path("data/splits/train.jsonl"))
    val = read_jsonl(Path("data/splits/val.jsonl"))
    model = train_setfit(train)
    model.save_pretrained("data/models/setfit")
    wrapper = SetFitWrapper(model)
    preds = [p for p, _ in predict_with_proba(wrapper, val)]
    macro = f1_score([r.environment for r in val], preds, average="macro")
    print(f"val macro-F1: {macro:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test**

Run: `uv run pytest tests/test_train_setfit.py -v -m slow`
Expected: PASS (first run downloads ~90MB model). If the SetFit API
surface differs in the installed version (it has shifted between majors),
fix against `uv run python -c "import setfit; print(setfit.__version__)"`
docs — the wrapper contract must not change.

- [ ] **Step 5: Train for real**

Run: `uv run python -m training.train_setfit`
Expected: minutes on M3 Pro; val macro-F1 printed. Compare with the
baseline number from Task 15 and record both in the commit message.

- [ ] **Step 6: Commit**

```bash
git add training/train_setfit.py tests/test_train_setfit.py
git commit -m "feat: setfit training (val F1 <X> vs baseline <Y>)"
```

---

### Task 17: Evaluation (`eval/evaluate.py`)

**Files:**
- Create: `eval/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: models from Tasks 15/16 via `predict_with_proba`; labeled splits + `data/labeled/eval.jsonl` (the grey-tier unseen-genre set — legitimate here: evaluation only).
- Produces: `metrics(y_true: list[str], preds: list[tuple[str, float]], labels: list[str], threshold: float = 0.0) -> dict` with keys `macro_f1`, `per_class` (`{label: {"precision","recall","f1","support"}}`), `confusion` (nested dict), `coverage` (fraction ≥ threshold), `accuracy_covered` (accuracy among covered); `threshold_sweep(y_true, preds, labels) -> list[dict]` over thresholds `0.0–0.9 step 0.05`. CLI `python -m eval.evaluate --model baseline|setfit` writes `data/eval/report_<model>.md` (tables for test split, eval-tier set, sweep, title-only comparison) and `data/eval/report_<model>.json`.

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate.py`:
```python
from eval.evaluate import metrics, threshold_sweep

Y = ["forest", "forest", "desert", "cave"]
P = [("forest", 0.9), ("desert", 0.4), ("desert", 0.8), ("cave", 0.6)]
LABELS = ["cave", "desert", "forest"]


def test_metrics_basic():
    m = metrics(Y, P, LABELS)
    assert 0 < m["macro_f1"] < 1
    assert m["per_class"]["forest"]["support"] == 2
    assert m["confusion"]["forest"]["desert"] == 1
    assert m["coverage"] == 1.0


def test_threshold_filters_low_confidence():
    m = metrics(Y, P, LABELS, threshold=0.5)
    assert m["coverage"] == 0.75          # the 0.4 prediction abstains
    assert m["accuracy_covered"] == 1.0   # remaining three are correct


def test_sweep_monotone_coverage():
    rows = threshold_sweep(Y, P, LABELS)
    covs = [r["coverage"] for r in rows]
    assert covs == sorted(covs, reverse=True)
    assert rows[0]["threshold"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v` — expected FAIL.

- [ ] **Step 3: Write implementation**

`eval/evaluate.py`:
```python
"""Metrics, confusion, abstention threshold sweep, report generation."""
import argparse
import json
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

from common.schema import RoomRecord, read_jsonl
from common.textclean import build_text


def metrics(y_true, preds, labels, threshold: float = 0.0) -> dict:
    covered = [(t, p) for t, (p, c) in zip(y_true, preds) if c >= threshold]
    coverage = len(covered) / len(y_true) if y_true else 0.0
    acc = (sum(t == p for t, p in covered) / len(covered)) if covered else 0.0
    y_pred = [p for p, _ in preds]
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class = {lab: {"precision": float(pr[i]), "recall": float(rc[i]),
                       "f1": float(f1[i]), "support": int(sup[i])}
                 for i, lab in enumerate(labels)}
    present = [l for l in labels if per_class[l]["support"] > 0]
    macro = (sum(per_class[l]["f1"] for l in present) / len(present)
             if present else 0.0)
    confusion: dict[str, dict[str, int]] = {}
    for t, p in zip(y_true, y_pred):
        confusion.setdefault(t, {})[p] = confusion.get(t, {}).get(p, 0) + 1
    return {"macro_f1": macro, "per_class": per_class,
            "confusion": confusion, "coverage": coverage,
            "accuracy_covered": acc}


def threshold_sweep(y_true, preds, labels) -> list[dict]:
    rows = []
    for i in range(19):
        th = round(i * 0.05, 2)
        m = metrics(y_true, preds, labels, threshold=th)
        rows.append({"threshold": th, "coverage": m["coverage"],
                     "accuracy_covered": m["accuracy_covered"]})
    return rows


def _load_model(which: str):
    if which == "baseline":
        import joblib
        return joblib.load("data/models/baseline.joblib")
    from training.train_setfit import load_setfit
    return load_setfit(Path("data/models/setfit"))


def _title_only(records: list[RoomRecord]) -> list[RoomRecord]:
    import copy
    out = []
    for r in records:
        c = copy.copy(r)
        c.description = c.name  # build_text still valid; body = title
        out.append(c)
    return out


def _md_table(m: dict) -> str:
    lines = ["| class | P | R | F1 | n |", "|---|---|---|---|---|"]
    for lab, s in m["per_class"].items():
        lines.append(f"| {lab} | {s['precision']:.2f} | {s['recall']:.2f} "
                     f"| {s['f1']:.2f} | {s['support']} |")
    return "\n".join(lines)


def main() -> None:
    from training.baseline import predict_with_proba
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "setfit"], required=True)
    a = ap.parse_args()
    model = _load_model(a.model)
    labels = sorted(model.classes_)
    report = {"model": a.model}
    md = [f"# Evaluation report — {a.model}\n"]
    for name, path in [("test (clean, held-out areas)", "data/splits/test.jsonl"),
                       ("unseen-genre (grey eval tier)", "data/labeled/eval.jsonl")]:
        recs = read_jsonl(Path(path))
        y = [r.environment for r in recs]
        preds = predict_with_proba(model, recs)
        m = metrics(y, preds, labels)
        report[name] = m
        md += [f"## {name}\n", f"macro-F1: **{m['macro_f1']:.3f}**\n",
               _md_table(m), ""]
        if name.startswith("test"):
            sweep = threshold_sweep(y, preds, labels)
            report["sweep"] = sweep
            md += ["## Abstention sweep\n",
                   "| threshold | coverage | acc@covered |", "|---|---|---|"]
            md += [f"| {r['threshold']:.2f} | {r['coverage']:.2f} "
                   f"| {r['accuracy_covered']:.2f} |" for r in sweep]
            md.append("")
            t_preds = predict_with_proba(model, _title_only(recs))
            tm = metrics(y, t_preds, labels)
            report["title_only_macro_f1"] = tm["macro_f1"]
            md.append(f"Title-only macro-F1: {tm['macro_f1']:.3f} "
                      f"(vs full {m['macro_f1']:.3f})\n")
    out = Path("data/eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report_{a.model}.json").write_text(json.dumps(report, indent=1))
    (out / f"report_{a.model}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -v` — expected 3 PASS.

- [ ] **Step 5: Evaluate both models for real**

```bash
uv run python -m eval.evaluate --model baseline
uv run python -m eval.evaluate --model setfit
```
Expected: two reports in `data/eval/`. Decision rule from the spec: keep
the baseline if SetFit's measured benefit is negligible (< ~2 macro-F1
points on both test and unseen-genre sets); otherwise proceed with
SetFit. Record the decision + chosen abstention threshold (pick from the
sweep: highest coverage with acc@covered ≥ 0.9, adjustable by user) in
the commit message and in `data/eval/DECISION.md`.

- [ ] **Step 6: Commit**

```bash
git add eval/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation with confusion, sweep, unseen-genre sets"
```

---

### Task 18: ONNX export (`export/export_onnx.py`)

**Files:**
- Create: `export/export_onnx.py`
- Test: `tests/test_export_onnx.py` (marked `slow`)

**Interfaces:**
- Consumes: `data/models/setfit/` (SetFit body = SentenceTransformer, head = sklearn LogisticRegression).
- Produces: `export_model(model_dir: Path, out_dir: Path, threshold: float) -> None` writing to `out_dir`:
  - `encoder.onnx` — the transformer body, inputs `input_ids`,`attention_mask` (int64, dynamic batch/sequence), output `last_hidden_state`.
  - `tokenizer.json` — HF fast-tokenizer file copied from the model dir.
  - `head.json` — `{"classes": [...], "coef": [[...]], "intercept": [...]}` from the sklearn head.
  - `preprocessing_spec.json` — the exact C# contract: `{"build_text": "clean(name)+\"\\n\"+clean(description)", "strip_patterns": [...], "max_word_pieces": 256, "long_text_policy": "split description into chunks of <=200 words, embed build_text(name, chunk) per chunk, mean the embeddings", "pooling": "mean over attention_mask", "normalize": "l2", "head": "logits = emb @ coef.T + intercept; softmax", "threshold": <chosen>, "taxonomy_version": "1.0.0", "model": "all-MiniLM-L6-v2 (fine-tuned)"}`.
  - `embed(texts: list[str], onnx_path, tokenizer_path) -> np.ndarray` — the reference implementation of that spec using onnxruntime + `tokenizers` (used again by Task 19 parity).

- [ ] **Step 1: Write the failing test**

`tests/test_export_onnx.py`:
```python
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
def test_export_and_reference_embed(tmp_path: Path):
    from export.export_onnx import embed, export_model
    model_dir = Path("data/models/setfit")
    if not model_dir.exists():
        pytest.skip("train setfit first (Task 16)")
    out = tmp_path / "out"
    export_model(model_dir, out, threshold=0.5)
    for f in ["encoder.onnx", "tokenizer.json", "head.json",
              "preprocessing_spec.json"]:
        assert (out / f).exists()
    head = json.loads((out / "head.json").read_text())
    emb = embed(["Temple\nA vaulted stone hall."],
                out / "encoder.onnx", out / "tokenizer.json")
    assert emb.shape == (1, 384)
    assert np.isclose(np.linalg.norm(emb[0]), 1.0, atol=1e-4)  # l2-normed
    assert len(head["coef"][0]) == 384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_export_onnx.py -v -m slow` — expected FAIL (import error) or SKIP if Task 16 not yet run for real (do not proceed on SKIP).

- [ ] **Step 3: Write implementation**

`export/export_onnx.py`:
```python
"""Export fine-tuned SetFit body to ONNX + head/tokenizer/preprocessing."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from common.textclean import _CODE_RES  # documented strip patterns


def export_model(model_dir: Path, out_dir: Path, threshold: float) -> None:
    from setfit import SetFitModel
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SetFitModel.from_pretrained(str(model_dir))
    st = model.model_body
    auto = st[0].auto_model.eval()

    ids = torch.ones(1, 8, dtype=torch.int64)
    mask = torch.ones(1, 8, dtype=torch.int64)
    torch.onnx.export(
        auto, (ids, mask), str(out_dir / "encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "last_hidden_state": {0: "batch", 1: "seq"}},
        opset_version=17)

    shutil.copy(model_dir / "tokenizer.json", out_dir / "tokenizer.json")

    head = model.model_head
    (out_dir / "head.json").write_text(json.dumps({
        "classes": [str(c) for c in head.classes_],
        "coef": head.coef_.tolist(),
        "intercept": head.intercept_.tolist()}))

    (out_dir / "preprocessing_spec.json").write_text(json.dumps({
        "build_text": 'clean(name) + "\\n" + clean(description)',
        "strip_patterns": [rx.pattern for rx in _CODE_RES] + ["~"],
        "whitespace": "collapse runs to single space, trim",
        "max_word_pieces": 256,
        "long_text_policy": ("split description into chunks of <=200 words; "
                             "embed build_text(name, chunk) per chunk; "
                             "mean the chunk embeddings"),
        "pooling": "mean over attention_mask", "normalize": "l2",
        "head": "logits = emb @ coef.T + intercept; softmax",
        "threshold": threshold, "taxonomy_version": "1.0.0",
        "model": "sentence-transformers/all-MiniLM-L6-v2 (fine-tuned)",
    }, indent=1))


def embed(texts: list[str], onnx_path: Path, tokenizer_path: Path
          ) -> np.ndarray:
    """Reference implementation of preprocessing_spec.json (C# mirrors this)."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.enable_truncation(max_length=256)
    sess = ort.InferenceSession(str(onnx_path))
    out = []
    for t in texts:
        enc = tok.encode(t)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        hidden = sess.run(["last_hidden_state"],
                          {"input_ids": ids, "attention_mask": mask})[0]
        m = mask[..., None].astype(np.float32)
        emb = (hidden * m).sum(axis=1) / m.sum(axis=1)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        out.append(emb[0])
    return np.array(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True)
    a = ap.parse_args()
    export_model(Path("data/models/setfit"), Path("export/out"), a.threshold)
    print("exported to export/out/")
```

(Note: chunking in `embed` is deliberately absent — truncation at 256
covers v1 inference; the chunk policy is specified for C# long-text
handling and exercised in parity fixtures only via texts under the
limit. If the chosen model is the **baseline**, stop: Task 18–20 apply
only to the SetFit path; a baseline-only release ships the joblib +
a documented TF-IDF spec instead, which is a plan change to raise to
the user.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_export_onnx.py -v -m slow` — expected PASS.

- [ ] **Step 5: Export for real** (threshold from Task 17's `DECISION.md`)

Run: `uv run python -m export.export_onnx --threshold <chosen>`

- [ ] **Step 6: Commit**

```bash
git add export/export_onnx.py tests/test_export_onnx.py
git commit -m "feat: onnx export with head.json and preprocessing spec"
```

---

### Task 19: Parity fixtures (`export/parity.py`)

**Files:**
- Create: `export/parity.py`
- Test: `tests/test_parity.py` (marked `slow`)

**Interfaces:**
- Consumes: `embed` from Task 18, `head.json`, splits.
- Produces: `make_fixtures(out_dir: Path, records: list[RoomRecord], n: int = 20) -> Path` — writes `out_dir/parity_fixtures.json`: per fixture `{"id","text","token_ids","embedding" (384 floats), "probs","predicted"}`; `verify(out_dir: Path, atol_emb: float = 1e-3, atol_prob: float = 1e-3) -> bool` — recomputes everything from `encoder.onnx` + `tokenizer.json` + `head.json` and compares. The C# implementation later verifies against the same file with the same tolerances.

- [ ] **Step 1: Write the failing test**

`tests/test_parity.py`:
```python
import json
from pathlib import Path

import pytest


@pytest.mark.slow
def test_fixture_roundtrip(tmp_path: Path):
    from export.export_onnx import export_model
    from export.parity import make_fixtures, verify
    from common.schema import read_jsonl
    model_dir = Path("data/models/setfit")
    if not model_dir.exists():
        pytest.skip("train setfit first (Task 16)")
    out = tmp_path / "out"
    export_model(model_dir, out, threshold=0.5)
    recs = read_jsonl(Path("data/splits/test.jsonl"))[:20]
    fpath = make_fixtures(out, recs)
    data = json.loads(fpath.read_text())
    assert len(data["fixtures"]) == len(recs)
    f0 = data["fixtures"][0]
    assert len(f0["embedding"]) == 384 and f0["predicted"] in \
        json.loads((out / "head.json").read_text())["classes"]
    assert verify(out) is True


@pytest.mark.slow
def test_verify_detects_corruption(tmp_path: Path):
    from export.export_onnx import export_model
    from export.parity import make_fixtures, verify
    from common.schema import read_jsonl
    model_dir = Path("data/models/setfit")
    if not model_dir.exists():
        pytest.skip("train setfit first")
    out = tmp_path / "out"
    export_model(model_dir, out, threshold=0.5)
    make_fixtures(out, read_jsonl(Path("data/splits/test.jsonl"))[:5])
    p = out / "parity_fixtures.json"
    data = json.loads(p.read_text())
    data["fixtures"][0]["embedding"][0] += 0.5
    p.write_text(json.dumps(data))
    assert verify(out) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parity.py -v -m slow` — expected FAIL (import error).

- [ ] **Step 3: Write implementation**

`export/parity.py`:
```python
"""Golden fixtures proving tokenizer+encoder+head parity across runtimes."""
import json
from pathlib import Path

import numpy as np

from common.schema import RoomRecord
from common.textclean import build_text
from export.export_onnx import embed


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _predict(emb: np.ndarray, head: dict) -> tuple[np.ndarray, list[str]]:
    coef = np.array(head["coef"])
    logits = emb @ coef.T + np.array(head["intercept"])
    if logits.ndim == 2 and logits.shape[1] == 1:  # binary sklearn shape
        logits = np.hstack([-logits, logits])
    probs = _softmax(logits)
    preds = [head["classes"][i] for i in probs.argmax(axis=1)]
    return probs, preds


def _token_ids(text: str, tokenizer_path: Path) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.enable_truncation(max_length=256)
    return tok.encode(text).ids


def make_fixtures(out_dir: Path, records: list[RoomRecord],
                  n: int = 20) -> Path:
    head = json.loads((out_dir / "head.json").read_text())
    records = records[:n]
    texts = [build_text(r.name, r.description) for r in records]
    embs = embed(texts, out_dir / "encoder.onnx", out_dir / "tokenizer.json")
    probs, preds = _predict(embs, head)
    fixtures = []
    for r, t, e, p, pred in zip(records, texts, embs, probs, preds):
        fixtures.append({"id": r.id, "text": t,
                         "token_ids": _token_ids(t, out_dir / "tokenizer.json"),
                         "embedding": [float(x) for x in e],
                         "probs": [float(x) for x in p],
                         "predicted": pred})
    path = out_dir / "parity_fixtures.json"
    path.write_text(json.dumps({"atol_embedding": 1e-3, "atol_probs": 1e-3,
                                "fixtures": fixtures}))
    return path


def verify(out_dir: Path, atol_emb: float = 1e-3,
           atol_prob: float = 1e-3) -> bool:
    data = json.loads((out_dir / "parity_fixtures.json").read_text())
    head = json.loads((out_dir / "head.json").read_text())
    texts = [f["text"] for f in data["fixtures"]]
    embs = embed(texts, out_dir / "encoder.onnx", out_dir / "tokenizer.json")
    probs, preds = _predict(embs, head)
    for f, e, p, pred in zip(data["fixtures"], embs, probs, preds):
        if f["token_ids"] != _token_ids(f["text"],
                                        out_dir / "tokenizer.json"):
            return False
        if not np.allclose(f["embedding"], e, atol=atol_emb):
            return False
        if not np.allclose(f["probs"], p, atol=atol_prob):
            return False
        if f["predicted"] != pred:
            return False
    return True


if __name__ == "__main__":
    from common.schema import read_jsonl
    out = Path("export/out")
    make_fixtures(out, read_jsonl(Path("data/splits/test.jsonl")))
    print("parity:", "OK" if verify(out) else "FAILED")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parity.py -v -m slow` — expected 2 PASS.

- [ ] **Step 5: Generate real fixtures**

Run: `uv run python -m export.parity` — expected `parity: OK`.

- [ ] **Step 6: Commit**

```bash
git add export/parity.py tests/test_parity.py
git commit -m "feat: cross-runtime parity fixtures and verifier"
```

---

### Task 20: Model package assembly (`export/package.py`)

**Files:**
- Create: `export/package.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `export/out/` contents (Tasks 18–19), `taxonomy.json`, `data/eval/report_setfit.md`, `data/raw/PINS.json`.
- Produces: `assemble(out_dir: Path, package_dir: Path, extra: dict[str, Path]) -> Path` — copies `encoder.onnx, tokenizer.json, head.json, preprocessing_spec.json, parity_fixtures.json` plus `taxonomy.json`, `eval_report.md`, `LICENSES.md`, writes `manifest.json` (`{"version": "0.1.0", "created": iso-date, "files": {name: sha256}, "source_pins": {...}}`), zips to `<package_dir>/wundur-room-classifier-0.1.0.zip`, returns zip path. `LICENSES.md` content is generated from `SOURCES` train-tier entries + a line noting synthetic data + the MiniLM Apache-2.0 model license.

- [ ] **Step 1: Write the failing test**

`tests/test_package.py`:
```python
import hashlib
import json
import zipfile
from pathlib import Path

from export.package import assemble


def test_assemble(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    for name in ["encoder.onnx", "tokenizer.json", "head.json",
                 "preprocessing_spec.json", "parity_fixtures.json"]:
        (out / name).write_text(f"fake {name}")
    tax = tmp_path / "taxonomy.json"
    tax.write_text('{"version": "1.0.0"}')
    report = tmp_path / "report.md"
    report.write_text("# eval")
    pins = tmp_path / "PINS.json"
    pins.write_text('{"tbamud": "abc"}')
    zip_path = assemble(out, tmp_path / "pkg",
                        {"taxonomy.json": tax, "eval_report.md": report,
                         "PINS.json": pins})
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        assert {"encoder.onnx", "taxonomy.json", "eval_report.md",
                "manifest.json", "LICENSES.md"} <= names
        manifest = json.loads(z.read("manifest.json"))
    expected = hashlib.sha256(b"fake encoder.onnx").hexdigest()
    assert manifest["files"]["encoder.onnx"] == expected
    assert manifest["source_pins"] == {"tbamud": "abc"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_package.py -v` — expected FAIL.

- [ ] **Step 3: Write implementation**

`export/package.py`:
```python
"""Assemble the versioned model package the Wundur client consumes."""
import datetime
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from sources.download import SOURCES

PACKAGE_VERSION = "0.1.0"
CORE_FILES = ["encoder.onnx", "tokenizer.json", "head.json",
              "preprocessing_spec.json", "parity_fixtures.json"]


def _licenses_md() -> str:
    lines = ["# Licenses\n",
             "Model: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), "
             "fine-tuned.\n", "Training data:\n"]
    for spec in SOURCES.values():
        if spec.tier == "train":
            lines.append(f"- {spec.name}: {spec.license_note}")
    lines.append("- synthetic: original generated text, no external license")
    lines.append("\nEval-only sources never entered model weights.")
    return "\n".join(lines) + "\n"


def assemble(out_dir: Path, package_dir: Path,
             extra: dict[str, Path]) -> Path:
    stage = package_dir / f"wundur-room-classifier-{PACKAGE_VERSION}"
    stage.mkdir(parents=True, exist_ok=True)
    for name in CORE_FILES:
        shutil.copy(out_dir / name, stage / name)
    pins = {}
    for name, src in extra.items():
        if name == "PINS.json":
            pins = json.loads(src.read_text())
            continue
        shutil.copy(src, stage / name)
    (stage / "LICENSES.md").write_text(_licenses_md())
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(stage.iterdir())}
    (stage / "manifest.json").write_text(json.dumps({
        "version": PACKAGE_VERSION,
        "created": datetime.date.today().isoformat(),
        "files": files, "source_pins": pins}, indent=1))
    zip_path = package_dir / f"{stage.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.iterdir()):
            z.write(p, p.name)
    return zip_path


if __name__ == "__main__":
    zp = assemble(Path("export/out"), Path("export"),
                  {"taxonomy.json": Path("taxonomy.json"),
                   "eval_report.md": Path("data/eval/report_setfit.md"),
                   "PINS.json": Path("data/raw/PINS.json")})
    print(zp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_package.py -v` — expected 1 PASS.

- [ ] **Step 5: Assemble the real package and run the whole suite**

```bash
uv run python -m export.package
uv run pytest -m "not slow"
```
Expected: zip printed; all fast tests pass. Report the final package
path, model choice, eval numbers, and chosen threshold to the user.

- [ ] **Step 6: Commit**

```bash
git add export/package.py tests/test_package.py
git commit -m "feat: model package assembly with manifest and licenses"
```

---

## Self-review notes

- Spec coverage: acquisition (T7,9), parsing (T5,6,8), cleaning/dedupe (T3,10), taxonomy + tie-breaks (T4), labeling with review queue and sector cross-check (T11,12), synthetic with hard negatives (T13), area split + tier enforcement (T14), baseline-first (T15), SetFit (T16), evaluation incl. unseen-genre + sweep + title-only comparison (T17), ONNX + preprocessing spec (T18), parity fixtures (T19), package with hashes/licenses/pins (T20). Quantization: intentionally deferred (spec marks it optional) — revisit after v1 numbers.
- Known adaptation points (called out in-task): CoffeeMud tag names (T8 step 1), repo URLs/globs vs. cloned reality (T7 steps 5–6), SetFit API surface per installed version (T16 step 4).
