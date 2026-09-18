"""Export fine-tuned SetFit body to ONNX + head/tokenizer/preprocessing."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from common.textclean import _CODE_RES  # documented strip patterns


class _EncoderWrapper(torch.nn.Module):
    """Narrows BertModel.forward to (input_ids, attention_mask) -> hidden.

    Needed for tracing: recent `transformers` versions decorate
    BertModel.forward with output-capturing machinery that expands the
    full keyword signature (token_type_ids, use_cache, ...) during JIT
    tracing and raises `TypeError: got multiple values for argument
    'use_cache'`. Exporting a thin wrapper with a two-argument forward
    avoids that expansion.
    """

    def __init__(self, auto_model):
        super().__init__()
        self.auto_model = auto_model

    def forward(self, input_ids, attention_mask):
        return self.auto_model(
            input_ids=input_ids, attention_mask=attention_mask,
            return_dict=False)[0]


def export_model(model_dir: Path, out_dir: Path, threshold: float) -> None:
    from setfit import SetFitModel
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SetFitModel.from_pretrained(str(model_dir))
    st = model.model_body
    # torch.onnx.export runs on CPU; move off MPS/CUDA if the model loaded
    # there (SetFit.from_pretrained picks up an available accelerator).
    st = st.to("cpu")
    auto = st[0].auto_model.eval()
    wrapper = _EncoderWrapper(auto).eval()

    ids = torch.ones(1, 8, dtype=torch.int64)
    mask = torch.ones(1, 8, dtype=torch.int64)
    # Adaptation: torch 2.14's torch.onnx.export defaults to dynamo=True,
    # which requires the optional `onnxscript` package (not installed here
    # and not in project deps). Pass dynamo=False to use the legacy
    # TorchScript-based exporter, which supports dynamic_axes/opset_version
    # exactly as briefed with no extra dependency.
    torch.onnx.export(
        wrapper, (ids, mask), str(out_dir / "encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "last_hidden_state": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False)

    tokenizer_src = model_dir / "tokenizer.json"
    if not tokenizer_src.exists():
        # Fall back to the base model's tokenizer from the HF cache if
        # SetFit's save_pretrained did not write one (not expected to
        # trigger here -- data/models/setfit/tokenizer.json is present --
        # but documented per the task brief's instruction).
        from huggingface_hub import hf_hub_download
        tokenizer_src = Path(hf_hub_download(
            "sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json"))
    shutil.copy(tokenizer_src, out_dir / "tokenizer.json")

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
