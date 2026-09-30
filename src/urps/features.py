"""Leakage-free feature engineering.

Anti-leakage rule: a row that predicts the score of year ``t`` may only use
information published up to year ``t - 1``. Lags are built by joining on
``year - 1`` (not by positional shifting), so gaps in a university's history
never silently pair non-consecutive years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from urps.data import PILLARS

TARGET = "target"
_LAG_SOURCE = [*PILLARS, "total_score", "student_staff_ratio", "international_students"]
_DELTA_SOURCE = [*PILLARS, "total_score"]

FEATURES: list[str] = (
    [f"lag1_{c}" for c in _LAG_SOURCE]
    + ["lag1_log_students"]
    + [f"delta_{c}" for c in _DELTA_SOURCE]
    + [f"trend3_{c}" for c in _DELTA_SOURCE]
    + [f"histmean_{c}" for c in _DELTA_SOURCE]
)


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

    The first observed year of each university is dropped because it has no
    previous year to learn from.
    """
    base = clean[["university", "year", "total_score"]].rename(columns={"total_score": TARGET})
    df = _assemble(base, clean)
    return df[["university", "year", *FEATURES, TARGET]].sort_values(["year", "university"]).reset_index(drop=True)


def future_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Features for the next, not yet published cycle (``max(year) + 1``)."""
    next_year = int(clean["year"].max()) + 1
    latest = clean.loc[clean["year"] == next_year - 1, ["university"]].copy()
    latest["year"] = next_year
    df = _assemble(latest, clean)
    return df[["university", "year", *FEATURES]].sort_values("university").reset_index(drop=True)
