"""Models, temporal validation and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from urps.features import FEATURES, TARGET

MODEL_NAMES: tuple[str, ...] = ("persistence", "linear", "gbm")
INTERVAL_COVERAGE = 0.9


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Naive baseline: next year's score equals last year's score."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PersistenceRegressor:
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X["lag1_total_score"], dtype=float)


def make_estimator(name: str, seed: int = 42) -> BaseEstimator:
    """Create an unfitted estimator by name."""
    if name == "persistence":
        return PersistenceRegressor()
    if name == "linear":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LinearRegression()
        )
    if name == "gbm":
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, random_state=seed
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def temporal_split(features: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all years before ``test_year``, test on ``test_year`` only."""
    train = features[features["year"] < test_year]
    test = features[features["year"] == test_year]
    if train.empty or test.empty:
        years = sorted(features["year"].unique())
        raise ValueError(f"Cannot split on test year {test_year}; available target years: {years}")
    return train, test


def evaluate(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    last_year: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Return MAE, RMSE and R² rounded to 4 decimals.

    Scores differ far more *between* universities than from one year to the next,
    so R² of the score level is close to 1 even for the naive "same as last year"
    forecast. When ``last_year`` scores are given, ``r2_change`` is added: R² of the
    predicted year-over-year change, i.e. the share of the actual change the model
    explains. The persistence baseline scores about 0 on it by construction.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    metrics = {
        "mae": round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))), 4),
        "r2": round(float(r2_score(y_true_arr, y_pred_arr)), 4),
    }
    if last_year is not None:
        prev = np.asarray(last_year, dtype=float)
        metrics["r2_change"] = round(float(r2_score(y_true_arr - prev, y_pred_arr - prev)), 4)
    return metrics


@dataclass
class TrainedModel:
    """A fitted estimator together with its uncertainty interval half-width."""

    name: str
    estimator: BaseEstimator
    interval: float
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    train_years: list[int] = field(default_factory=list)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pred = np.clip(self.estimator.predict(X[self.features]), 0, 100)
        return pd.DataFrame(
            {
                "predicted_score": pred,
                "lower": np.clip(pred - self.interval, 0, 100),
                "upper": np.clip(pred + self.interval, 0, 100),
            },
            index=X.index,
        )


def usable_features(train: pd.DataFrame) -> list[str]:
    """Features with at least one observed value (short histories leave trend features empty)."""
    return [f for f in FEATURES if train[f].notna().any()]


def fit_model(name: str, train: pd.DataFrame, seed: int = 42) -> TrainedModel:
    """Fit a model; estimate a 90% interval on the last training year held out."""
    years = sorted(int(y) for y in train["year"].unique())
    interval = float("nan")
    if len(years) >= 2:
        calib_fit, calib = train[train["year"] < years[-1]], train[train["year"] == years[-1]]
        cols = usable_features(calib_fit)
        est = make_estimator(name, seed).fit(calib_fit[cols], calib_fit[TARGET])
        residuals = np.abs(calib[TARGET].to_numpy() - est.predict(calib[cols]))
        interval = float(np.quantile(residuals, INTERVAL_COVERAGE))
    cols = usable_features(train)
    estimator = make_estimator(name, seed).fit(train[cols], train[TARGET])
    return TrainedModel(name=name, estimator=estimator, interval=interval, features=cols, train_years=years)


def rank_band(scores: pd.Series | np.ndarray, exact_limit: int = 200, band: int = 50) -> list[str]:
    """Convert predicted scores of one cohort into THE-style rank labels."""
    ranks = pd.Series(np.asarray(scores)).rank(ascending=False, method="min").astype(int)
    labels = []
    for r in ranks:
        if r <= exact_limit:
            labels.append(str(r))
        else:
            lower = ((r - 1) // band) * band + 1
            labels.append(f"{lower}-{lower + band - 1}")
    return labels
