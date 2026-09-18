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
    # n sampled records + 3 crafted edge-case fixtures.
    assert len(data["fixtures"]) == len(recs) + 3
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


@pytest.mark.slow
def test_verify_detects_corrupted_raw_text(tmp_path: Path):
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
    # Mutate a fixture's raw description so it no longer matches the
    # stored (cleaned) text -- verify() must catch this.
    data["fixtures"][0]["description"] += " extra corrupting text"
    p.write_text(json.dumps(data))
    assert verify(out) is False


def test_predict_matches_sklearn_binary():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from export.parity import _predict
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 8)); y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    head = {"classes": [str(c) for c in clf.classes_],
            "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
    probs, preds = _predict(X[:5], head)
    assert np.allclose(probs, clf.predict_proba(X[:5]), atol=1e-9)
