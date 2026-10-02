"""End-to-end experiment: clean -> features -> train -> evaluate -> report."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from urps import __version__, viz
from urps.data import load_clean
from urps.features import TARGET, build_features, future_features
from urps.model import (
    MODEL_NAMES,
    TrainedModel,
    best_model,
    evaluate,
    fit_model,
    rank_band,
    temporal_split,
    to_absolute,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_models(
    feats: pd.DataFrame, test_year: int, seed: int = 42
) -> tuple[dict[str, dict[str, float]], dict[str, TrainedModel], pd.DataFrame, pd.DataFrame]:
    """Fit every model on years before ``test_year`` and evaluate it on ``test_year``.

    Metrics are computed on the year-relative scale, i.e. after removing the common
    shift of the whole ranking scale, which no model can know in advance.
    """
    train, test = temporal_split(feats, test_year)
    metrics: dict[str, dict[str, float]] = {}
    trained: dict[str, TrainedModel] = {}
    for name in MODEL_NAMES:
        model = fit_model(name, train, seed=seed)
        pred = model.predict(test)
        m = evaluate(test[TARGET], pred["predicted_score"].to_numpy(), test["lag1_total_score"])
        inside = (test[TARGET] >= pred["lower"]) & (test[TARGET] <= pred["upper"])
        m["interval_coverage"] = round(float(inside.mean()), 4)
        metrics[name] = m
        trained[name] = model
    return metrics, trained, train, test


def forecast_next_cycle(clean: pd.DataFrame, model: TrainedModel) -> pd.DataFrame:
    """Forecast every university of the latest year for the next cycle, on the published scale."""
    future = future_features(clean)
    relative = model.predict(future)
    out = future[["university", "year"]].join(to_absolute(relative, future["prev_year_mean"]))
    out["predicted_rank"] = rank_band(relative["predicted_score"])
    return out.sort_values("predicted_score", ascending=False).reset_index(drop=True)


def run_experiment(
    data_path: str | Path,
    out_dir: str | Path = "reports",
    test_year: int | None = None,
    seed: int = 42,
    figures: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline and write metrics, predictions and figures to ``out_dir``."""
    data_path, out_dir = Path(data_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = load_clean(data_path)
    feats = build_features(clean)
    test_year = int(test_year or feats["year"].max())
    metrics, trained, train, test = evaluate_models(feats, test_year, seed)

    best = best_model(metrics)
    baseline_mae = metrics["persistence"]["mae"]
    improvement = round(100 * (baseline_mae - metrics[best]["mae"]) / baseline_mae, 2)

    # Test-year predictions of the best model (relative to the year average).
    best_pred = trained[best].predict(test)
    predictions = pd.DataFrame(
        {
            "university": test["university"],
            "year": test["year"],
            "actual_relative": test[TARGET],
            "predicted_relative": best_pred["predicted_score"],
            "lower": best_pred["lower"],
            "upper": best_pred["upper"],
            "actual_rank": rank_band(test[TARGET]),
            "predicted_rank": rank_band(best_pred["predicted_score"]),
        }
    )
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    # Forecast of the next, unpublished cycle with the best model refit on all years.
    forecast_next_cycle(clean, fit_model(best, feats, seed=seed)).to_csv(out_dir / "forecast.csv", index=False)

    cols = trained[best].features
    if best == "persistence":
        imp = pd.Series(1.0, index=["lag1_total_score"])
    else:
        importance = permutation_importance(
            trained[best].estimator,
            test[cols],
            test[TARGET],
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=seed,
        )
        imp = pd.Series(importance.importances_mean, index=cols).sort_values(ascending=False)

    if figures:
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        viz.plot_predicted_vs_actual(
            test[TARGET],
            best_pred["predicted_score"],
            f"{best} model, test year {test_year}",
            fig_dir / "predicted_vs_actual.png",
        )
        viz.plot_feature_importance(imp, fig_dir / "feature_importance.png")
        viz.plot_model_comparison(metrics, fig_dir / "model_comparison.png")

    yearly_shift = test["year_mean"] - test["prev_year_mean"]
    result: dict[str, Any] = {
        "test_year": test_year,
        "train_years": sorted(int(y) for y in train["year"].unique()),
        "n_train": len(train),
        "n_test": len(test),
        "metrics": metrics,
        "best_model": best,
        "mae_improvement_over_persistence_pct": improvement,
        "scale_shift_in_test_year": round(float(yearly_shift.iloc[0]), 4),
        "top_features": [str(f) for f in imp.head(5).index],
        "provenance": {
            "data_file": data_path.name,
            "data_sha256": _sha256(data_path),
            "urps_version": __version__,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "seed": seed,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def select_model(feats: pd.DataFrame, seed: int = 42) -> str:
    """Pick the model with the lowest MAE on the latest year held out (the baseline included)."""
    metrics, *_ = evaluate_models(feats, int(feats["year"].max()), seed)
    return best_model(metrics)


def forecast_university(data_path: str | Path, university: str, seed: int = 42) -> dict[str, Any]:
    """Forecast the next-cycle score and rank for one university (user story US2)."""
    clean = load_clean(data_path)
    feats = build_features(clean)
    name = select_model(feats, seed=seed)
    pred = forecast_next_cycle(clean, fit_model(name, feats, seed=seed))
    match = pred[pred["university"].str.casefold() == university.strip().casefold()]
    if match.empty:
        raise ValueError(f"University not found in the latest year: {university!r}")
    row = match.iloc[0]
    return {
        "university": str(row["university"]),
        "year": int(row["year"]),
        "predicted_score": round(float(row["predicted_score"]), 2),
        "interval_90": [round(float(row["lower"]), 2), round(float(row["upper"]), 2)],
        "predicted_rank": str(row["predicted_rank"]),
        "model": name,
        "note": "Estimate based on public indicators, not a guarantee of the actual ranking.",
    }
