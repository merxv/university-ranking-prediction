"""Declarative schema for the cleaned ranking table (requirement NFR7).

The pipeline stops on a schema violation instead of training on bad data.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

# Indicators may be unpublished for some universities ("-" in the raw file);
# the overall score is the prediction target and therefore mandatory.
_score = pa.Column(float, pa.Check.in_range(0, 100), nullable=True)

CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "university": pa.Column(str, pa.Check.str_length(min_value=2)),
        "country": pa.Column(str),
        "year": pa.Column(int, pa.Check.in_range(1990, 2100)),
        "rank_lower": pa.Column(int, pa.Check.ge(1)),
        "rank_upper": pa.Column(int, pa.Check.ge(1)),
        "teaching": _score,
        "international": _score,
        "research": _score,
        "citations": _score,
        "income": _score,
        "total_score": pa.Column(float, pa.Check.in_range(0, 100), nullable=False),
        "num_students": pa.Column(float, pa.Check.gt(0), nullable=True),
        "student_staff_ratio": pa.Column(float, pa.Check.gt(0), nullable=True),
        "international_students": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
    },
    checks=[
        pa.Check(lambda df: df["rank_upper"] >= df["rank_lower"], error="rank_upper < rank_lower"),
    ],
    unique=["university", "year"],
    strict=True,
    coerce=True,
)


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the cleaned table; raises ``pandera.errors.SchemaError`` on failure."""
    return CLEAN_SCHEMA.validate(df)
