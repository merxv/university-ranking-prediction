from pathlib import Path

import pandas as pd
import pytest

from urps.data import clean
from urps.synthetic import generate_the_table

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "the_rankings_sample.csv"
REFERENCE = Path(__file__).parent / "data" / "reference_metrics.json"


@pytest.fixture(scope="session")
def sample_path() -> Path:
    return SAMPLE


@pytest.fixture(scope="session")
def small_raw() -> pd.DataFrame:
    """A small synthetic raw table: 40 universities x 6 years."""
    return generate_the_table(n_universities=40, years=range(2016, 2022), seed=7)


@pytest.fixture(scope="session")
def small_clean(small_raw: pd.DataFrame) -> pd.DataFrame:
    return clean(small_raw)


@pytest.fixture(scope="session")
def sample_university(sample_path: Path) -> str:
    """Name of a university present in the latest year of the committed sample."""
    df = clean(pd.read_csv(sample_path, dtype=str, keep_default_na=False))
    return str(df.loc[df["year"] == df["year"].max(), "university"].sort_values().iloc[0])
