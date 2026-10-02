import numpy as np
import pytest

from urps.features import TARGET, build_features
from urps.model import (
    MODEL_NAMES,
    best_model,
    evaluate,
    fit_model,
    make_estimator,
    rank_band,
    temporal_split,
    to_absolute,
)


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
    m = evaluate(y, y)
    assert {k: m[k] for k in ("mae", "rmse", "r2", "rank_mae", "spearman")} == {
        "mae": 0.0,
        "rmse": 0.0,
        "r2": 1.0,
        "rank_mae": 0.0,
        "spearman": 1.0,
    }


def test_evaluate_known_values():
    m = evaluate(np.array([0.0, 0.0]), np.array([1.0, 3.0]))
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5.0), abs=1e-4)


def test_persistence_predicts_last_year(feats):
    model = fit_model("persistence", feats)
    pred = model.predict(feats)["predicted_score"]
    assert np.allclose(pred, feats["lag1_total_score"])


@pytest.mark.parametrize("name", ["ridge", "gbm"])
def test_models_fit_and_give_valid_intervals(feats, name):
    train, test = temporal_split(feats, 2021)
    model = fit_model(name, train, seed=1)
    pred = model.predict(test)
    assert model.interval > 0
    assert (pred["lower"] <= pred["predicted_score"]).all()
    assert (pred["predicted_score"] <= pred["upper"]).all()
    assert pred["predicted_score"].notna().all()
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


def test_r2_change_is_zero_for_persistence_and_one_for_perfect():
    last = np.array([50.0, 60.0, 70.0, 80.0])
    actual = np.array([51.0, 59.0, 72.0, 79.0])
    assert evaluate(actual, last, last)["r2_change"] <= 0.0  # predicts "no change"
    assert evaluate(actual, actual, last)["r2_change"] == 1.0
    assert "r2_change" not in evaluate(actual, last)


def test_rank_metrics():
    actual = np.array([90.0, 80.0, 70.0, 60.0])
    perfect = evaluate(actual, actual)
    assert perfect["rank_mae"] == 0.0 and perfect["spearman"] == 1.0
    swapped = evaluate(actual, np.array([80.0, 90.0, 70.0, 60.0]))
    assert swapped["rank_mae"] == pytest.approx(0.5)


def test_best_model_includes_baseline_and_prefers_it_on_ties():
    m = {"persistence": {"mae": 1.0}, "ridge": {"mae": 1.2}, "gbm": {"mae": 1.1}}
    assert best_model(m) == "persistence"
    m["ridge"]["mae"] = 0.9
    assert best_model(m) == "ridge"
    assert best_model({n: {"mae": 1.0} for n in MODEL_NAMES}) == "persistence"


def test_to_absolute_adds_offset_and_clips():
    import pandas as pd

    rel = pd.DataFrame(
        {"predicted_score": [-60.0, 0.0, 50.0], "lower": [-61.0, -1.0, 49.0], "upper": [-59.0, 1.0, 51.0]}
    )
    out = to_absolute(rel, 55.0)
    assert out["predicted_score"].tolist() == [0.0, 55.0, 100.0]
    assert out["lower"].tolist() == [0.0, 54.0, 100.0]
