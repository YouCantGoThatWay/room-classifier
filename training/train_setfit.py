"""SetFit fine-tune of all-MiniLM-L6-v2 with a logistic-regression head.

Contrastive-phase cost control (sanctioned adaptation)
-------------------------------------------------------
SetFit's contrastive pair generation is ~O(n^2) in the number of training
examples. Running it over the full 7,565-example train split takes hours,
which is out of budget for this machine. Instead:

1. The sentence-transformer *body* is fine-tuned (contrastive phase) on a
   deterministic, stratified subsample of the train split: up to
   ``SUBSAMPLE_CAP`` (default 64) examples per class, sampled with
   ``random.Random(SEED)`` (default 42) so the run is reproducible. Classes
   with fewer than the cap keep all their examples.
2. ``num_iterations=20`` bounds the number of contrastive pairs generated
   per example (the setfit library default of "generate every possible
   pair, then oversample" is combinatorially explosive once there are many
   classes) so pair generation and training stay fast even though the
   dataset has 15 environment classes.
3. After the contrastive fine-tune, the classification head (a
   scikit-learn ``LogisticRegression`` by default in setfit) is refit from
   scratch on embeddings of the *entire* train split, computed with the
   now-fine-tuned body. This is done via ``SetFitModel.fit(...)``, which
   for a non-differentiable (sklearn) head only embeds the given texts and
   calls ``model_head.fit`` -- it does not repeat the expensive pair
   generation / contrastive step. This keeps the SetFitWrapper contract
   (``predict_proba``, ``classes_``) and the CLI unchanged while letting
   the head see all 7,565 training examples instead of just the subsample.

If the installed setfit version's API differs from any of the above (e.g.
``TrainingArguments`` field names), adapt the calls -- the wrapper contract
must not change.
"""
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# MPS has no pinned-memory support; torch's DataLoader warns about this on
# every run on Apple Silicon even though nothing is actually broken.
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score

from common.schema import RoomRecord, read_jsonl
from common.textclean import build_text
from training.baseline import predict_with_proba

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUBSAMPLE_CAP = 64
SEED = 42
NUM_ITERATIONS = 20


class SetFitWrapper:
    """Duck-types sklearn's predict_proba/classes_ for shared eval code."""

    def __init__(self, model: SetFitModel):
        self.model = model
        self.classes_ = list(model.labels) if model.labels else \
            list(model.model_head.classes_)

    def predict_proba(self, texts: list[str]):
        import numpy as np
        return np.asarray(self.model.predict_proba(texts))


def _dataset(records: list[RoomRecord]) -> Dataset:
    return Dataset.from_dict({
        "text": [build_text(r.name, r.description) for r in records],
        "label": [r.environment for r in records]})


def _stratified_subsample(records: list[RoomRecord], cap: int = SUBSAMPLE_CAP,
                          seed: int = SEED) -> list[RoomRecord]:
    """Up to `cap` examples per class, deterministic given `seed`."""
    by_class: dict[str, list[RoomRecord]] = defaultdict(list)
    for r in records:
        by_class[r.environment].append(r)
    rng = random.Random(seed)
    out: list[RoomRecord] = []
    for env in sorted(by_class):
        group = by_class[env]
        out.extend(rng.sample(group, cap) if len(group) > cap else group)
    return out


def train_setfit(train: list[RoomRecord], epochs: int = 1) -> SetFitModel:
    labels = sorted({r.environment for r in train})
    subsample = _stratified_subsample(train)

    model = SetFitModel.from_pretrained(BASE_MODEL, labels=labels)
    args = TrainingArguments(batch_size=16, num_epochs=epochs,
                             num_iterations=NUM_ITERATIONS,
                             sampling_strategy="oversampling")
    Trainer(model=model, args=args, train_dataset=_dataset(subsample)).train()

    # Refit the head on the FULL train split using the fine-tuned body.
    # For a non-differentiable (sklearn) head, SetFitModel.fit only embeds
    # the texts and calls model_head.fit -- no contrastive pairs involved.
    full_texts = [build_text(r.name, r.description) for r in train]
    full_labels = [r.environment for r in train]
    model.fit(full_texts, full_labels, num_epochs=1)
    return model


def load_setfit(path: Path) -> SetFitWrapper:
    return SetFitWrapper(SetFitModel.from_pretrained(str(path)))


def main() -> None:
    train = read_jsonl(Path("data/splits/train.jsonl"))
    val = read_jsonl(Path("data/splits/val.jsonl"))
    model = train_setfit(train)
    Path("data/models/setfit").mkdir(parents=True, exist_ok=True)
    model.save_pretrained("data/models/setfit")
    wrapper = SetFitWrapper(model)
    preds = [p for p, _ in predict_with_proba(wrapper, val)]
    macro = f1_score([r.environment for r in val], preds, average="macro")
    print(f"val macro-F1: {macro:.3f}  (n_train={len(train)}, n_val={len(val)})")


if __name__ == "__main__":
    main()
