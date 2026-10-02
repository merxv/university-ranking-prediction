"""Service layer used by the dashboard: train once, then answer user-story queries.

The dashboard never touches models or files directly (Assignment 2, layered
architecture); it calls these framework-independent functions instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.inspection import permutation_importance

from urps.data import PILLAR_WEIGHTS, PILLARS
from urps.features import TARGET, build_features
from urps.model import TrainedModel, best_model, fit_model
from urps.pipeline import evaluate_models, forecast_next_cycle

INDICATOR_LABELS = {
    "total_score": "Overall score",
    "teaching": "Teaching",
    "international": "International outlook",
    "research": "Research",
    "citations": "Citations",
    "income": "Industry income",
}
NOTE = "Predictions are estimates based on public indicators, not a guarantee of an actual ranking position."


def feature_label(name: str) -> str:
    """Turn a feature column name into a readable label, e.g. ``lag1_research`` -> ``Research (last year)``."""
    prefixes = {
        "lag1_": "last year",
        "delta_": "change over last year",
        "trend3_": "3-year trend",
        "histmean_": "historical mean",
    }
    for prefix, suffix in prefixes.items():
        if name.startswith(prefix):
            base = name[len(prefix) :]
            label = INDICATOR_LABELS.get(base, base.replace("_", " ").capitalize())
            return f"{label} ({suffix})"
    return name


@dataclass
class Bundle:
    """Everything the dashboard needs after one training run."""

    clean: pd.DataFrame
    test_year: int
    metrics: dict[str, dict[str, float]]
    test_predictions: pd.DataFrame
    importance: pd.Series
    model: TrainedModel  # best model refit on all years, used for forecasts
    forecast: pd.DataFrame


def build_bundle(clean: pd.DataFrame, seed: int = 42) -> Bundle:
    """Evaluate all models on the latest year, pick the best, refit it and forecast."""
    feats = build_features(clean)
    test_year = int(feats["year"].max())
    metrics, trained, _, test = evaluate_models(feats, test_year, seed)
    best = best_model(metrics)
    # Relative scale (score minus the year's average) for the evaluation charts.
    test_predictions = test[["university", "year", TARGET]].join(trained[best].predict(test))

    if best == "persistence":
        importance = pd.Series(1.0, index=["lag1_total_score"])
    else:
        cols = trained[best].features
        imp = permutation_importance(
            trained[best].estimator,
            test[cols],
            test[TARGET],
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=seed,
        )
        importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)

    final = fit_model(best, feats, seed=seed)
    return Bundle(
        clean=clean,
        test_year=test_year,
        metrics=metrics,
        test_predictions=test_predictions,
        importance=importance,
        model=final,
        forecast=forecast_next_cycle(clean, final),
    )


def latest_indicators(clean: pd.DataFrame, university: str) -> pd.Series:
    """Indicator values of a university in the latest published year."""
    latest = clean[(clean["university"] == university) & (clean["year"] == clean["year"].max())]
    if latest.empty:
        raise ValueError(f"University not found in the latest year: {university!r}")
    return latest.iloc[0]


def what_if(bundle: Bundle, university: str, changes: dict[str, float]) -> dict[str, float | str]:
    """Forecast the next cycle after changing the latest published indicators (FR9).

    The overall score of the edited year is adjusted with the THE pillar weights,
    so a change in one indicator propagates to every feature built from it.
    """
    unknown = set(changes) - set(PILLARS)
    if unknown:
        raise ValueError(f"Only ranking indicators can be changed, got {sorted(unknown)}")
    row = latest_indicators(bundle.clean, university)
    edited = bundle.clean.copy()
    mask = (edited["university"] == university) & (edited["year"] == row["year"])
    delta_total = 0.0
    for pillar, value in changes.items():
        if not 0 <= value <= 100:
            raise ValueError(f"{pillar} must be between 0 and 100, got {value}")
        old = row[pillar]
        if pd.notna(old):
            delta_total += PILLAR_WEIGHTS[pillar] * (value - float(old))
        edited.loc[mask, pillar] = value
    edited.loc[mask, "total_score"] = min(100.0, max(0.0, float(row["total_score"]) + delta_total))

    scenario = forecast_next_cycle(edited, bundle.model).set_index("university").loc[university]
    base = bundle.forecast.set_index("university").loc[university]
    return {
        "base_score": round(float(base["predicted_score"]), 2),
        "base_rank": str(base["predicted_rank"]),
        "scenario_score": round(float(scenario["predicted_score"]), 2),
        "scenario_rank": str(scenario["predicted_rank"]),
        "scenario_lower": round(float(scenario["lower"]), 2),
        "scenario_upper": round(float(scenario["upper"]), 2),
        "change": round(float(scenario["predicted_score"] - base["predicted_score"]), 2),
    }
