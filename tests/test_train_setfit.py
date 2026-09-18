import pytest

from common.schema import RoomRecord
from tests.test_baseline import _corpus, rec
from training.baseline import predict_with_proba


def test_train_setfit_rejects_eval_tier_records():
    from training.train_setfit import train_setfit
    recs = _corpus()
    recs[0] = RoomRecord(id="s:a:eval0", source="s", tier="eval", world="s",
                         area="a", name="Room", description=recs[0].description,
                         environment=recs[0].environment)
    with pytest.raises(ValueError, match="eval-tier record"):
        train_setfit(recs)


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
