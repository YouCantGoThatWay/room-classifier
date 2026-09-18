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
