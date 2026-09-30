"""Loading and cleaning of raw ranking tables (THE format).

Raw ranking files are written for humans, not for models: ranks contain ties
and bands, numbers contain separators and percent signs, and the same
university is spelled differently across years. This module turns such a file
into one tidy row per (university, year) with numeric columns only.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

from urps.schema import validate_clean

PILLARS: tuple[str, ...] = ("teaching", "international", "research", "citations", "income")

# THE methodology weights used to compose the overall score.
PILLAR_WEIGHTS: dict[str, float] = {
    "teaching": 0.30,
    "international": 0.075,
    "research": 0.30,
    "citations": 0.30,
    "income": 0.025,
}

RAW_COLUMNS: tuple[str, ...] = (
    "world_rank",
    "university_name",
    "country",
    *PILLARS,
    "total_score",
    "num_students",
    "student_staff_ratio",
    "international_students",
    "year",
)

MISSING_TOKENS = {"", "-", "--", "n/a", "na", "nan", "none"}

_RANK_RE = re.compile(r"^=?\s*(\d+)\s*(?:[-–—]\s*(\d+)|(\+))?$")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_TOKENS


def parse_rank(value: object) -> tuple[int, int]:
    """Parse a published rank into an inclusive ``(lower, upper)`` range.

    >>> parse_rank("=45")
    (45, 45)
    >>> parse_rank("201-250")
    (201, 250)
    >>> parse_rank("1001+")
    (1001, 1001)
    """
    text = str(value).strip()
    match = _RANK_RE.match(text)
    if not match:
        raise ValueError(f"Unrecognised rank value: {value!r}")
    lower = int(match.group(1))
    upper = int(match.group(2)) if match.group(2) else lower
    if lower < 1 or upper < lower:
        raise ValueError(f"Invalid rank range: {value!r}")
    return lower, upper


def parse_number(value: object) -> float:
    """Parse numbers such as ``"20,152"``, ``"25%"`` or ``"-"`` (missing -> NaN)."""
    if _is_missing(value):
        return math.nan
    text = str(value).strip().replace(",", "").rstrip("%").strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Unrecognised numeric value: {value!r}") from exc


def normalise_name(name: object) -> str:
    """Return a comparison key: trimmed, single-spaced and case-folded."""
    return " ".join(str(name).split()).casefold()


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a raw ranking CSV as strings and check that required columns exist."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")
    return raw


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw THE-format table into a validated, tidy table."""
    df = pd.DataFrame()
    keys = raw["university_name"].map(normalise_name)
    # Use the most frequent original spelling of each university as its label.
    spelled = raw["university_name"].map(lambda s: " ".join(str(s).split()))
    canonical = spelled.groupby(keys).agg(lambda s: s.value_counts().index[0])
    df["university"] = keys.map(canonical)
    df["country"] = raw["country"].str.strip()
    df["year"] = raw["year"].map(parse_number).astype(int)

    ranks = raw["world_rank"].map(parse_rank)
    df["rank_lower"] = ranks.map(lambda r: r[0]).astype(int)
    df["rank_upper"] = ranks.map(lambda r: r[1]).astype(int)

    numeric = [*PILLARS, "total_score", "num_students", "student_staff_ratio", "international_students"]
    for col in numeric:
        df[col] = raw[col].map(parse_number).astype(float)

    # Rows without a published overall score (THE prints "-" below the top 200 in
    # some years) cannot serve as training targets and are dropped.
    df = (
        df.dropna(subset=["total_score"])
        .drop_duplicates(subset=["university", "year"], keep="first")
        .sort_values(["university", "year"])
        .reset_index(drop=True)
    )
    return validate_clean(df)


def load_clean(path: str | Path) -> pd.DataFrame:
    """Shortcut: read, clean and validate a raw ranking file."""
    return clean(load_raw(path))
