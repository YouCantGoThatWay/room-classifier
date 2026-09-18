from eval.evaluate import metrics, threshold_sweep

Y = ["forest", "forest", "desert", "cave"]
P = [("forest", 0.9), ("desert", 0.4), ("desert", 0.8), ("cave", 0.6)]
LABELS = ["cave", "desert", "forest"]


def test_metrics_basic():
    m = metrics(Y, P, LABELS)
    assert 0 < m["macro_f1"] < 1
    assert m["per_class"]["forest"]["support"] == 2
    assert m["confusion"]["forest"]["desert"] == 1
    assert m["coverage"] == 1.0


def test_threshold_filters_low_confidence():
    m = metrics(Y, P, LABELS, threshold=0.5)
    assert m["coverage"] == 0.75          # the 0.4 prediction abstains
    assert m["accuracy_covered"] == 1.0   # remaining three are correct


def test_sweep_monotone_coverage():
    rows = threshold_sweep(Y, P, LABELS)
    covs = [r["coverage"] for r in rows]
    assert covs == sorted(covs, reverse=True)
    assert rows[0]["threshold"] == 0.0
