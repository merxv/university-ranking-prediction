import numpy as np
import pytest

from urps.features import TARGET, build_features
from urps.model import evaluate, fit_model, make_estimator, rank_band, temporal_split


@pytest.fixture(scope="module")
def feats(small_clean):
    return build_features(small_clean)


def test_temporal_split_has_no_overlap_and_no_future_in_train(feats):
    train, test = temporal_split(feats, 2021)
    assert train["year"].max() < test["year"].min()
    assert set(test["year"]) == {2021}


def test_temporal_split_rejects_unknown_year(feats):
    with pytest.raises(ValueError, match="Cannot split"):
        temporal_split(feats, 1999)


def test_evaluate_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert evaluate(y, y) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_evaluate_known_values():
    m = evaluate(np.array([0.0, 0.0]), np.array([1.0, 3.0]))
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5.0), abs=1e-4)


def test_persistence_predicts_last_year(feats):
    model = fit_model("persistence", feats)
    pred = model.predict(feats)["predicted_score"]
    assert np.allclose(pred, feats["lag1_total_score"])


@pytest.mark.parametrize("name", ["linear", "gbm"])
def test_models_fit_and_give_valid_intervals(feats, name):
    train, test = temporal_split(feats, 2021)
    model = fit_model(name, train, seed=1)
    pred = model.predict(test)
    assert model.interval > 0
    assert (pred["lower"] <= pred["predicted_score"]).all()
    assert (pred["predicted_score"] <= pred["upper"]).all()
    assert pred["predicted_score"].between(0, 100).all()
    assert evaluate(test[TARGET], pred["predicted_score"].to_numpy())["r2"] > 0.9


def test_gbm_is_deterministic_with_seed(feats):
    a = fit_model("gbm", feats, seed=3).predict(feats)["predicted_score"]
    b = fit_model("gbm", feats, seed=3).predict(feats)["predicted_score"]
    assert np.array_equal(a, b)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_estimator("deep-transformer")


def test_rank_band_labels():
    labels = rank_band(np.array([90.0, 80.0, 80.0, 10.0]), exact_limit=2, band=50)
    assert labels == ["1", "2", "2", "1-50"]
    many = rank_band(np.linspace(100, 0, 260))
    assert many[0] == "1" and many[199] == "200" and many[200] == "201-250" and many[-1] == "251-300"
