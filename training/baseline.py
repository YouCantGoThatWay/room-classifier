"""TF-IDF + logistic regression baseline. The number to beat."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from common.schema import RoomRecord, read_jsonl
from common.textclean import build_text


def _texts(records: list[RoomRecord]) -> list[str]:
    return [build_text(r.name, r.description) for r in records]


def train_baseline(train: list[RoomRecord]) -> Pipeline:
    for r in train:
        if r.tier == "eval":
            raise ValueError(f"{r.id}: eval-tier record in training input")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(_texts(train), [r.environment for r in train])
    return model


def predict_with_proba(model, records: list[RoomRecord]
                       ) -> list[tuple[str, float]]:
    proba = model.predict_proba(_texts(records))
    classes = list(model.classes_)
    return [(classes[row.argmax()], float(row.max())) for row in proba]


def main() -> None:
    train = read_jsonl(Path("data/splits/train.jsonl"))
    val = read_jsonl(Path("data/splits/val.jsonl"))
    model = train_baseline(train)
    preds = [p for p, _ in predict_with_proba(model, val)]
    macro = f1_score([r.environment for r in val], preds, average="macro")
    Path("data/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "data/models/baseline.joblib")
    print(f"val macro-F1: {macro:.3f}  (n_train={len(train)}, n_val={len(val)})")


if __name__ == "__main__":
    main()
