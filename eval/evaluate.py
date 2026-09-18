"""Metrics, confusion, abstention threshold sweep, report generation."""
import argparse
import json
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

from common.schema import RoomRecord, read_jsonl
from common.textclean import build_text


def metrics(y_true, preds, labels, threshold: float = 0.0) -> dict:
    covered = [(t, p) for t, (p, c) in zip(y_true, preds) if c >= threshold]
    coverage = len(covered) / len(y_true) if y_true else 0.0
    acc = (sum(t == p for t, p in covered) / len(covered)) if covered else 0.0
    y_pred = [p for p, _ in preds]
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class = {lab: {"precision": float(pr[i]), "recall": float(rc[i]),
                       "f1": float(f1[i]), "support": int(sup[i])}
                 for i, lab in enumerate(labels)}
    present = [l for l in labels if per_class[l]["support"] > 0]
    macro = (sum(per_class[l]["f1"] for l in present) / len(present)
             if present else 0.0)
    confusion: dict[str, dict[str, int]] = {}
    for t, p in zip(y_true, y_pred):
        confusion.setdefault(t, {})[p] = confusion.get(t, {}).get(p, 0) + 1
    return {"macro_f1": macro, "per_class": per_class,
            "confusion": confusion, "coverage": coverage,
            "accuracy_covered": acc}


def threshold_sweep(y_true, preds, labels) -> list[dict]:
    rows = []
    for i in range(19):
        th = round(i * 0.05, 2)
        m = metrics(y_true, preds, labels, threshold=th)
        rows.append({"threshold": th, "coverage": m["coverage"],
                     "accuracy_covered": m["accuracy_covered"]})
    return rows


def _load_model(which: str):
    if which == "baseline":
        import joblib
        return joblib.load("data/models/baseline.joblib")
    from training.train_setfit import load_setfit
    return load_setfit(Path("data/models/setfit"))


def _title_only(records: list[RoomRecord]) -> list[RoomRecord]:
    import copy
    out = []
    for r in records:
        c = copy.copy(r)
        c.description = c.name  # build_text still valid; body = title
        out.append(c)
    return out


def _md_table(m: dict) -> str:
    lines = ["| class | P | R | F1 | n |", "|---|---|---|---|---|"]
    for lab, s in m["per_class"].items():
        lines.append(f"| {lab} | {s['precision']:.2f} | {s['recall']:.2f} "
                     f"| {s['f1']:.2f} | {s['support']} |")
    return "\n".join(lines)


def main() -> None:
    from training.baseline import predict_with_proba
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "setfit"], required=True)
    a = ap.parse_args()
    model = _load_model(a.model)
    labels = sorted(model.classes_)
    report = {"model": a.model}
    md = [f"# Evaluation report — {a.model}\n"]
    for name, path in [("test (clean, held-out areas)", "data/splits/test.jsonl"),
                       ("unseen-genre (grey eval tier)", "data/labeled/eval.jsonl")]:
        recs = read_jsonl(Path(path))
        y = [r.environment for r in recs]
        preds = predict_with_proba(model, recs)
        m = metrics(y, preds, labels)
        report[name] = m
        md += [f"## {name}\n", f"macro-F1: **{m['macro_f1']:.3f}**\n",
               _md_table(m), ""]
        if name.startswith("test"):
            sweep = threshold_sweep(y, preds, labels)
            report["sweep"] = sweep
            md += ["## Abstention sweep\n",
                   "| threshold | coverage | acc@covered |", "|---|---|---|"]
            md += [f"| {r['threshold']:.2f} | {r['coverage']:.2f} "
                   f"| {r['accuracy_covered']:.2f} |" for r in sweep]
            md.append("")
            t_preds = predict_with_proba(model, _title_only(recs))
            tm = metrics(y, t_preds, labels)
            report["title_only_macro_f1"] = tm["macro_f1"]
            md.append(f"Title-only macro-F1: {tm['macro_f1']:.3f} "
                      f"(vs full {m['macro_f1']:.3f})\n")
    out = Path("data/eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report_{a.model}.json").write_text(json.dumps(report, indent=1))
    (out / f"report_{a.model}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
