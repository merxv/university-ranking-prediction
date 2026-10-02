"""Leakage-free feature engineering.

Anti-leakage rule: a row that predicts the score of year ``t`` may only use
information published up to year ``t - 1``. Lags are built by joining on
``year - 1`` (not by positional shifting), so gaps in a university's history
never silently pair non-consecutive years.

All scores are first expressed *relative to the average of their ranking year*.
Ranking agencies rescale scores and change methodology between editions, which
moves the whole scale up or down by several points (in the public THE data the
average changes by -3.9 to +3.6 points per year). Such a common shift cannot be
predicted from history and does not change anyone's position, so models learn
and predict the relative score; the target year's average is kept separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from urps.data import PILLARS

TARGET = "target"  # overall score relative to the average of its ranking year
SCORE_COLUMNS: list[str] = [*PILLARS, "total_score"]
_LAG_SOURCE = [*PILLARS, "total_score", "student_staff_ratio", "international_students"]
_DELTA_SOURCE = [*PILLARS, "total_score"]

FEATURES: list[str] = (
    [f"lag1_{c}" for c in _LAG_SOURCE]
    + ["lag1_log_students"]
    + [f"delta_{c}" for c in _DELTA_SOURCE]
    + [f"trend3_{c}" for c in _DELTA_SOURCE]
    + [f"histmean_{c}" for c in _DELTA_SOURCE]
)


def relative_to_year(clean: pd.DataFrame) -> pd.DataFrame:
    """Subtract the yearly average from every score column (indicators and overall score)."""
    rel = clean.copy()
    for col in SCORE_COLUMNS:
        rel[col] = rel[col] - rel.groupby("year")[col].transform("mean")
    return rel


def year_means(clean: pd.DataFrame) -> pd.Series:
    """Average overall score per ranking year, used to convert relative scores back."""
    return clean.groupby("year")["total_score"].mean()


def _snapshot(clean: pd.DataFrame, offset: int, prefix: str) -> pd.DataFrame:
    """Return indicator values of year ``t - offset`` keyed by target year ``t``."""
    snap = clean[["university", "year", *_LAG_SOURCE, "num_students"]].copy()
    snap["year"] = snap["year"] + offset
    return snap.rename(columns={c: f"{prefix}{c}" for c in [*_LAG_SOURCE, "num_students"]})


def _history_mean(clean: pd.DataFrame) -> pd.DataFrame:
    """Mean of each indicator over all years up to and including ``t - 1``, keyed by ``t``."""
    hist = clean.sort_values("year")[["university", "year", *_DELTA_SOURCE]].copy()
    means = hist.groupby("university")[list(_DELTA_SOURCE)].expanding().mean().reset_index(level=0, drop=True)
    hist[list(_DELTA_SOURCE)] = means
    hist["year"] = hist["year"] + 1
    return hist.rename(columns={c: f"histmean_{c}" for c in _DELTA_SOURCE})


def _assemble(base: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    df = base.merge(_snapshot(clean, 1, "lag1_"), on=["university", "year"], how="inner")
    df = df.merge(_snapshot(clean, 2, "lag2_"), on=["university", "year"], how="left")
    df = df.merge(_snapshot(clean, 3, "lag3_"), on=["university", "year"], how="left")
    df = df.merge(_history_mean(clean), on=["university", "year"], how="left")
    df["lag1_log_students"] = np.log(df["lag1_num_students"])
    for c in _DELTA_SOURCE:
        df[f"delta_{c}"] = df[f"lag1_{c}"] - df[f"lag2_{c}"]
        # Average yearly change over the last three published cycles (momentum).
        df[f"trend3_{c}"] = (df[f"lag1_{c}"] - df[f"lag3_{c}"]) / 2
    return df


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Build the supervised table: one row per (university, year) with a known target.

    ``target`` is the overall score relative to its year's average; ``year_mean``
    (average of year ``t``) and ``prev_year_mean`` (year ``t - 1``) allow converting
    back to the published scale. The first observed year of each university is
    dropped because it has no previous year to learn from.
    """
    rel = relative_to_year(clean)
    base = rel[["university", "year", "total_score"]].rename(columns={"total_score": TARGET})
    df = _assemble(base, rel)
    means = year_means(clean)
    df["year_mean"] = df["year"].map(means)
    df["prev_year_mean"] = (df["year"] - 1).map(means)
    cols = ["university", "year", *FEATURES, TARGET, "year_mean", "prev_year_mean"]
    return df[cols].sort_values(["year", "university"]).reset_index(drop=True)


def future_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Features for the next, not yet published cycle (``max(year) + 1``).

    ``prev_year_mean`` is the average of the latest published year: the forecast
    assumes no common shift of the scale, because such shifts are not predictable.
    """
    rel = relative_to_year(clean)
    next_year = int(clean["year"].max()) + 1
    latest = rel.loc[rel["year"] == next_year - 1, ["university"]].copy()
    latest["year"] = next_year
    df = _assemble(latest, rel)
    df["prev_year_mean"] = float(year_means(clean).loc[next_year - 1])
    cols = ["university", "year", *FEATURES, "prev_year_mean"]
    return df[cols].sort_values("university").reset_index(drop=True)
