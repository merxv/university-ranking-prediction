"""Report figures (static PNG files)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_predicted_vs_actual(y_true: pd.Series, y_pred: pd.Series, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.7)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1, label="perfect prediction")
    ax.set_xlabel("Actual score (relative to year average)")
    ax.set_ylabel("Predicted score (relative to year average)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(importance: pd.Series, path: Path, top: int = 10) -> Path:
    data = importance.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(data.index, data.values)
    ax.set_xlabel("Permutation importance (increase in MAE)")
    ax.set_title("Indicator influence on the prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(metrics: dict[str, dict[str, float]], path: Path) -> Path:
    names = list(metrics)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(names, [metrics[n]["mae"] for n in names])
    ax.set_ylabel("MAE on test year (relative score)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_trend(clean: pd.DataFrame, university: str, path: Path) -> Path:
    hist = clean[clean["university"] == university].sort_values("year")
    if hist.empty:
        raise ValueError(f"University not found: {university!r}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for col in ("total_score", "research", "citations", "teaching"):
        ax.plot(hist["year"], hist[col], marker="o", label=col)
    ax.set_xlabel("Ranking year")
    ax.set_ylabel("Score (0-100)")
    ax.set_title(university)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
