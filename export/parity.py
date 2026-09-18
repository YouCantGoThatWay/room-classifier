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
        p = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        probs = np.column_stack([1.0 - p, p])
    else:
        probs = _softmax(logits)
    preds = [head["classes"][i] for i in probs.argmax(axis=1)]
    return probs, preds


def _token_ids(text: str, tokenizer_path: Path) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.enable_truncation(max_length=256)
    return tok.encode(text).ids


# Crafted fixtures covering edge cases the sampled tbamud records don't
# exercise: color/tilde codes in the raw inputs, descriptions long enough
# to be truncated at 256 word pieces, and a terse minimal description.
_CRAFTED_FIXTURES = [
    {
        "id": "crafted:codes:0",
        "name": "&RThe @gPainted~ Hall",
        "description": ("A  &Rgrand hall@g   stretches before you~, its "
                        "walls lined with {Gfaded murals~.   Dust   motes "
                        "drift   through &Yshafts~ of {Wlight~."),
    },
    {
        "id": "crafted:long:0",
        "name": "Long Hall",
        "description": ("The stone wall stretches upward into the gloom, "
                        "carved with worn runes that no living scholar "
                        "can read. " * 25).strip(),
    },
    {
        "id": "crafted:terse:0",
        "name": "Nook",
        "description": "A small dusty nook.",
    },
]


def make_fixtures(out_dir: Path, records: list[RoomRecord],
                  n: int = 20) -> Path:
    head = json.loads((out_dir / "head.json").read_text())
    records = records[:n]
    items = [{"id": r.id, "name": r.name, "description": r.description}
             for r in records] + _CRAFTED_FIXTURES
    texts = [build_text(it["name"], it["description"]) for it in items]
    embs = embed(texts, out_dir / "encoder.onnx", out_dir / "tokenizer.json")
    probs, preds = _predict(embs, head)
    fixtures = []
    for it, t, e, p, pred in zip(items, texts, embs, probs, preds):
        fixtures.append({"id": it["id"], "name": it["name"],
                         "description": it["description"], "text": t,
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
        # Recompute text from the stored raw name/description to validate
        # the cleaning chain (this is what the C# port must reproduce).
        if build_text(f["name"], f["description"]) != f["text"]:
            return False
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
