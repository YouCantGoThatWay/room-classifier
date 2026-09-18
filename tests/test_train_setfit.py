import pytest

from tests.test_baseline import _corpus, rec
from training.baseline import predict_with_proba


@pytest.mark.slow
def test_setfit_smoke(tmp_path):
    from training.train_setfit import SetFitWrapper, load_setfit, train_setfit
    model = train_setfit(_corpus(), epochs=1)
    wrapper = SetFitWrapper(model)
    preds = predict_with_proba(wrapper, [
        rec(99, "Pines and oaks form a dense green canopy overhead.",
            "forest")])
    assert preds[0][0] in {"forest", "desert"}
    model.save_pretrained(str(tmp_path / "m"))
    again = load_setfit(tmp_path / "m")
    assert list(again.classes_) == list(wrapper.classes_)
