# Wundur room-environment classifier — training pipeline design

Date: 2026-09-17
Status: approved design; implementation plan to follow.

## Goal

Produce a small, locally trained text classifier that assigns an
environment class (and later modifier tags) to MUD rooms from their
title + description, packaged as an ONNX model the Wundur C# client
can consume. Classification is presentation-only: it never establishes
room identity, changes exits, or determines route safety. Room color is
a user-configurable palette mapping keyed on the predicted class (with
`base.modifier` override and fallback to `base`), never a model output.

Everything runs locally: acquisition, labeling (Claude, in-session),
training on an Apple Silicon MacBook (M3 Pro — minutes of training,
well within capability), and ONNX export.

## Decisions already made

- **Label target:** environment enum + modifier tags. Not colors.
  Colors are user configuration layered on top.
- **Taxonomy shape:** two-level. Single-label base environment plus
  multi-label modifiers. No flat `dark_forest`-style enum explosion.
- **Data posture:** *clean train, grey eval.* Only license-clean
  sources and synthetic text enter training. Grey-licensed and
  franchise-IP worlds are evaluation-only; their text never enters
  model weights.
- **Labeler:** Claude directly, in-session, batched. Azure GPT-5
  second-opinion pass is deferred (documented option for later
  disagreement mining, not v1).
- **v1 scope:** train the base-environment head only. Modifiers are
  labeled from day one but not trained in v1. Feature tags
  (shop/inn/stairs/...) deferred entirely.
- **Client integration out of scope:** deliverable is the model
  package; the C# `IRoomEnvironmentClassifier` work happens in the
  Wundur repo later.

## Data sources

| Source | Tier | Genre | Format |
|---|---|---|---|
| tbaMUD stock world | train | fantasy | CircleMUD `.wld` |
| CoffeeMud stock areas | train | fantasy | CoffeeMud XML (Apache-2.0) |
| Synthetic (Claude-generated) | train | all, esp. sci-fi/modern | JSONL |
| SMAUG FUSS stock | eval-only | fantasy | Merc/SMAUG `.are` |
| ROM stock areas | eval-only | fantasy | Merc `.are` |
| AwakeMUD CE world | eval-only | cyberpunk/modern | CircleMUD-style `.wld` |
| SWFotE FUSS world | eval-only | sci-fi (Star Wars) | SMAUG `.are` |

Licensing rationale: CircleMUD/tbaMUD relicensed LGPL in 2020
(verified at circlemud.org/license.html); CoffeeMud is Apache-2.0.
Merc/SMAUG-lineage content retains the old non-commercial Diku/Merc
license chain (the 2020 relicense covered only the original Diku
authors' work), and AwakeMUD/SWFotE prose is fan-written within
Shadowrun/Star Wars IP — hence eval-only. Every record carries
`tier`, source, and license metadata; the training code refuses
eval-tier records by construction, not convention.

All repos are cloned at pinned commits recorded in the acquisition
scripts.

## Pipeline stages

### 1. Acquire and parse (`sources/`)

Per-source download scripts (git clone at pinned commit) and three
parsers: CircleMUD `.wld`, Merc/SMAUG `.are` (`#ROOMS` sections),
CoffeeMud XML. All normalize to one JSONL schema:

```json
{"id": "", "source": "", "tier": "train|eval", "world": "", "area": "",
 "name": "", "description": "", "sector_hint": "", "flags": [],
 "license": "", "synthetic": false, "parent_id": null}
```

`sector_hint`/`flags` preserve the source's terrain metadata as weak
supervision for cross-checking, never as ground truth.

### 2. Clean and dedupe

Strip color codes (Circle `@x`, SMAUG `&x`, raw ANSI), tildes,
normalize whitespace; drop rooms with empty/trivial descriptions.
Exact dedupe by normalized-text hash, near-dedupe by shingle overlap
(stock areas are widely copied between codebases). The canonical
record retains the list of all sources it appeared in; a duplicate
seen in both tiers is assigned to eval (conservative direction).

### 3. Taxonomy (`taxonomy.json`)

Single source of truth, versioned. Base classes (single-label):

`indoor, settlement, road, grassland, forest, desert, mountain, cave,
water, underwater, swamp, snow, spacecraft, urban, unknown`

(`urban` = modern/sci-fi city exteriors, so cyberpunk streets do not
share a class with medieval villages.)

Modifiers (multi-label): `dark, bright, dense, ruined, magical,
underground`.

Tie-break rules are written in the file and applied consistently by
labeler and reviewers, e.g.: enclosed built space beats terrain
(tavern → `indoor`); `spacecraft` beats `indoor` (station greenhouse →
`spacecraft`); a bridge is the terrain it crosses at base level (bridge
feature tags are deferred); scenes visible through windows/exits/lore
do not count — classify the room the player is standing in.

### 4. Label (Claude, in-session)

Batched labeling of the deduped corpus. Output per room: base class,
modifiers, confidence (high/low), and an `ambiguous` flag with a short
reason. Disagreements between the label and `sector_hint` are flagged.
Human review is only of the flagged pile.

Caps: ~6–8k training-tier rooms via stratified sampling if the corpus
is larger; ~1.5k eval-tier rooms across the grey worlds.

### 5. Synthetic augmentation

After the first labeling pass reveals class counts, generate original
descriptions for starved classes (expected: spacecraft, underwater,
snow, urban), varying genre, length, wording, and explicitness.
Records marked `synthetic: true` with `parent_id` when derived from a
real room, and never placed in any eval set. Include hard negatives
(painted forest indoors, cave lake, spacecraft greenhouse) and
feature-negative examples.

### 6. Train (`training/`)

Split by world/area before augmentation; all descendants of a source
room stay in one partition. First a TF-IDF + logistic-regression
baseline, then SetFit on `sentence-transformers/all-MiniLM-L6-v2`
(384-dim, English, 256-word-piece truncation — long descriptions
chunked and mean-pooled; the exact policy is written down for later
C# parity). Runs on M3 Pro via MPS or CPU. Keep the simpler model if
the encoder's measured benefit is negligible.

### 7. Evaluate (`eval/`)

Macro-F1, per-class precision/recall, confusion matrix on the held-out
clean split; the grey/IP worlds as fully-unseen-genre test sets;
confidence-threshold sweep to select the abstention cutoff (below it
the client renders neutral). Compare title-only vs title+description.
Scores are not calibrated probabilities; the chosen threshold is an
empirical operating point.

### 8. Export (`export/`)

ONNX export of the encoder + head, tokenizer vocab/assets,
`taxonomy.json`, preprocessing spec, license notices, evaluation
report, file hashes — the model package the client consumes. A parity
fixture file (inputs with expected tokenizations, embeddings, and
predictions) lets the future C# implementation verify byte-level
agreement. Quantization (int8) optional, evaluated for accuracy before
adoption.

## Repo layout

```
sources/        download + parse scripts per MUD
data/           raw → normalized → labeled JSONL (gitignored; small fixtures kept)
training/       baseline + SetFit
eval/           metrics, confusion, threshold sweep
export/         ONNX + package assembly + parity fixtures
taxonomy.json   classes, modifiers, tie-breaks, version
docs/           this spec and the implementation plan
```

Python 3.12, `uv`-managed. Training-tier/eval-tier separation enforced
in code (loader rejects eval-tier records in training paths).

## Risks and mitigations

- **Taxonomy disagreement ceiling** — tie-break rules written before
  labeling; ambiguous-flag review measures where humans/models can't
  agree.
- **Stock-area duplication across tiers** — near-dedupe with
  eval-wins assignment.
- **C# parity** — tokenizer + preprocessing named as explicit package
  deliverables with a verification fixture; parity failures are the
  known risk, not ONNX export itself.
- **Class starvation** — synthetic augmentation driven by measured
  counts, not guesses.
