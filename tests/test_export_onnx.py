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
    text = "Temple\nA vaulted stone hall."
    emb = embed([text], out / "encoder.onnx", out / "tokenizer.json")
    assert emb.shape == (1, 384)
    assert np.isclose(np.linalg.norm(emb[0]), 1.0, atol=1e-4)  # l2-normed
    assert len(head["coef"][0]) == 384

    # Parity: embed() must match SetFit's model_body.encode for short texts.
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(str(model_dir))
    expected = model.model_body.encode([text], convert_to_numpy=True)
    assert np.allclose(emb[0], expected[0], atol=1e-3)
