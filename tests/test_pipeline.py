import json

import pytest

from urps.pipeline import forecast_university, run_experiment

from .conftest import REFERENCE


@pytest.fixture(scope="module")
def result(sample_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    return run_experiment(sample_path, out, seed=42), out


def test_outputs_are_written(result):
    _, out = result
    for name in ("metrics.json", "predictions.csv", "forecast.csv"):
        assert (out / name).exists(), name
    for fig in ("predicted_vs_actual.png", "feature_importance.png", "model_comparison.png"):
        assert (out / "figures" / fig).stat().st_size > 0


def test_models_beat_naive_baseline(result):
    res, _ = result
    assert res["mae_improvement_over_persistence_pct"] > 0
    assert res["metrics"][res["best_model"]]["r2"] > 0.85  # NFR1


def test_temporal_validation_is_used(result):
    res, _ = result
    assert max(res["train_years"]) < res["test_year"]


def test_provenance_is_recorded(result):
    res, _ = result
    prov = res["provenance"]
    assert len(prov["data_sha256"]) == 64
    assert prov["seed"] == 42


def test_regression_against_reference_metrics(result):
    """Regression test: metrics must stay close to the stored reference run."""
    res, _ = result
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert res["best_model"] == reference["best_model"]
    for model, ref in reference["metrics"].items():
        for metric in ("mae", "rmse", "r2"):
            assert res["metrics"][model][metric] == pytest.approx(ref[metric], abs=0.02), (model, metric)


def test_pipeline_is_reproducible(sample_path, tmp_path, result):
    res, _ = result
    again = run_experiment(sample_path, tmp_path, seed=42, figures=False)
    assert again["metrics"] == res["metrics"]


def test_forecast_university_is_labelled_as_estimate(sample_path, sample_university):
    fc = forecast_university(sample_path, f"  {sample_university.upper()} ")
    assert fc["university"] == sample_university
    assert fc["year"] == 2026
    assert fc["interval_90"][0] <= fc["predicted_score"] <= fc["interval_90"][1]
    assert "not a guarantee" in fc["note"]


def test_forecast_unknown_university(sample_path):
    with pytest.raises(ValueError, match="not found"):
        forecast_university(sample_path, "Hogwarts")


def test_level_r2_overstates_skill_compared_with_change_r2(result):
    """The naive baseline already has a high level R² but no skill on the yearly change."""
    res, _ = result
    m = res["metrics"]
    assert m["persistence"]["r2"] > 0.98
    assert abs(m["persistence"]["r2_change"]) < 0.05
    assert m[res["best_model"]]["r2_change"] > 0.1
