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
             for p in sorted(stage.iterdir()) if not p.name.startswith(".")}
    (stage / "manifest.json").write_text(json.dumps({
        "version": PACKAGE_VERSION,
        "created": datetime.date.today().isoformat(),
        "files": files, "source_pins": pins}, indent=1))
    zip_path = package_dir / f"{stage.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((p for p in stage.iterdir() if not p.name.startswith("."))):
            z.write(p, p.name)
    shutil.rmtree(stage, ignore_errors=True)
    return zip_path


if __name__ == "__main__":
    zp = assemble(Path("export/out"), Path("export"),
                  {"taxonomy.json": Path("taxonomy.json"),
                   "eval_report.md": Path("data/eval/report_setfit.md"),
                   "PINS.json": Path("data/raw/PINS.json")})
    print(zp)
