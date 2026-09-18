# room-classifier

A training pipeline for a small, fully local text classifier that assigns an
**environment class** (forest, cave, spacecraft, urban, …) to MUD rooms from
their title and description. The intended consumer is a MUD client's
auto-mapper: color each room box by predicted environment so the map reads at
a glance — green ribbons of forest, grey city blocks, blue water — with
uncertain rooms left neutral rather than colored wrong.

Classification is presentation-only by design: it never establishes room
identity, changes exits, or decides whether a route is safe.

## Results (v0.1.1)

| Model | Held-out test (macro-F1) | Unseen-genre worlds (macro-F1) |
|---|---|---|
| TF-IDF + logistic regression | 0.534 | 0.441 |
| **SetFit / all-MiniLM-L6-v2 (shipped)** | **0.633** | **0.529** |

At the shipping abstention threshold (0.80), the model colors ~42% of rooms
from entirely unfamiliar games at ~91% accuracy and abstains on the rest.
Caveats: the clean test split has zero support for `snow`, `spacecraft`, and
`urban` (so test macro-F1 is a 12-class figure), and `spacecraft`/`urban`
remain weak on unseen genres — see `data/eval/` reports after running the
pipeline.

## Taxonomy

15 base classes (single-label): `indoor, settlement, road, grassland, forest,
desert, mountain, cave, water, underwater, swamp, snow, spacecraft, urban,
unknown` — plus 6 multi-label modifiers (`dark, bright, dense, ruined,
magical, underground`, labeled but not trained in v1). Tie-break rules live
in `taxonomy.json`, the single source of truth.

## Pipeline

```
sources/    download (pinned commits) + parse 4 world-file dialects
            (CircleMUD/tbaMUD .wld, Merc/SMAUG .are, FUSS .are, AwakeMUD .wld,
            CoffeeMud .cmare) → normalize → dedupe (exact + minhash near-dup)
labeling/   LLM-labeled corpus tooling: batch prep, taxonomy-validated merge,
            review queue, synthetic augmentation for starved classes
training/   area-grouped train/val/test split (license-tier enforced),
            TF-IDF baseline, SetFit fine-tune (M-series laptop friendly)
eval/       macro-F1, per-class P/R, confusion, abstention threshold sweep,
            unseen-genre evaluation
export/     ONNX export, preprocessing contract, cross-runtime parity
            fixtures, versioned model package zip
```

Run stages in order:

```bash
uv sync
uv run python -m sources.download
uv run python -m sources.build_corpus
uv run python -m sources.dedupe
# labeling: see labeling/batches.py + labeling/merge.py (labels come from an LLM or humans)
uv run python -m training.split
uv run python -m training.baseline
uv run python -m training.train_setfit
uv run python -m eval.evaluate --model setfit
uv run python -m export.export_onnx --threshold 0.80
uv run python -m export.parity
uv run python -m export.package
```

Tests: `uv run pytest -m "not slow"` (fast) or `uv run pytest` (includes
model download / ONNX export).

## Model package

`export/package.py` assembles a versioned zip containing: `encoder.onnx`
(fine-tuned MiniLM body, 384-dim), `tokenizer.json`, `head.json` (logistic
head as a plain 15×384 matrix + intercepts), `preprocessing_spec.json` (the
exact contract a non-Python consumer must reproduce — cleaning regexes,
256-word-piece truncation, mean-pool + L2 normalize, softmax, batch=1
constraint), `parity_fixtures.json` (23 golden examples, including raw-text,
color-code, and truncation-boundary cases, for verifying a port numerically),
plus the taxonomy, evaluation report, license notices, and a sha256 manifest
with pinned source commits.

## Data and licensing posture

Training uses only license-clean sources — tbaMUD (LGPL since the 2020
CircleMUD/DikuMUD relicense), CoffeeMud (Apache-2.0), and original synthetic
text. Grey-licensed and franchise-fan content (Merc/ROM/SMAUG lineage,
Shadowrun- and Star Wars-themed worlds) is **evaluation-only**: its text
never enters model weights, enforced in code (the split/trainers reject
eval-tier records). No game world content is committed to this repository;
world files are downloaded at pinned commits into the gitignored `data/`
directory. Test fixtures use original prose, except one structural CoffeeMud
excerpt retained under Apache-2.0 (see `tests/fixtures/NOTICE.md`).
