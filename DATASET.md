# Training dataset card

Distributed as `train.jsonl` on this repo's releases (it is not tracked in
git). 8,540 records: 8,000 LLM-labeled rooms from license-clean MUD worlds
plus 540 original synthetic rooms.

## Schema (one JSON object per line)

| field | meaning |
|---|---|
| `id` | `{source}:{area}:{vnum}` or `synthetic:{class}:{n}` |
| `source` | `tbamud`, `coffeemud`, or `synthetic` |
| `tier` | always `train` in this file |
| `world`, `area` | source world / area-file stem |
| `name`, `description` | the room text (cleaned: color codes stripped, whitespace collapsed) |
| `sector_hint` | the source engine's terrain sector, kept as weak evidence only |
| `flags` | source room flags; may include a `dup_sources:` provenance note |
| `license` | license note for the room text |
| `synthetic`, `parent_id` | generated-record marker (all synthetic records here are parentless) |
| `environment` | the label: one of the 15 base classes in `taxonomy.json` |
| `modifiers` | multi-label modifiers (labeled, not used by the v1 model) |

## Provenance and licensing

- Room text from **tbaMUD** (7,961 records): LGPL (CircleMUD/DikuMUD were
  relicensed LGPL in 2020). Redistributed here with attribution.
- Room text from **CoffeeMud** (39 records): Apache-2.0, © Bo Zimmerman.
- **Synthetic** rooms (540 records): original text written for this dataset,
  covering underrepresented classes (underwater, snow, spacecraft, urban,
  swamp, desert) plus deliberate hard negatives.
- **Labels** (`environment`, `modifiers`): produced by an LLM following the
  tie-break rules in `taxonomy.json`, with low-confidence and
  sector-conflicting labels flagged for human review during construction.
  The labels and dataset assembly are licensed **CC-BY-4.0** — attribute
  "the room-classifier project".

Deliberately excluded: all content from Merc/ROM/SMAUG-lineage worlds and
franchise fan worlds (Shadowrun-, Star Wars-themed). Those were used for
*evaluation only* and are not redistributed; the pipeline in this repo
regenerates that evaluation set locally from pinned upstream commits.

## Known limitations

- Labels are single-annotator LLM output; ~30-40% carry a low-confidence or
  ambiguous flag in the source pipeline (the flags are not included in this
  file). Expect label noise on genuinely ambiguous rooms.
- Class balance is skewed toward `indoor`/`settlement`/`road`; the rarest
  natural classes are supplemented synthetically.
- `sector_hint` disagrees with the label on a minority of rooms by design —
  the text wins over the engine's movement-cost sector.
