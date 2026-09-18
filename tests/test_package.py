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
