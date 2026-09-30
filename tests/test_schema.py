import pandera.errors
import pytest

from urps.schema import validate_clean


def test_valid_table_passes(small_clean):
    assert len(validate_clean(small_clean)) == len(small_clean)


def test_score_out_of_range_is_rejected(small_clean):
    bad = small_clean.copy()
    bad.loc[0, "research"] = 140.0
    with pytest.raises(pandera.errors.SchemaError):
        validate_clean(bad)


def test_missing_mandatory_score_is_rejected(small_clean):
    bad = small_clean.copy()
    bad.loc[0, "total_score"] = float("nan")
    with pytest.raises(pandera.errors.SchemaError):
        validate_clean(bad)


def test_duplicate_university_year_is_rejected(small_clean):
    bad = small_clean.copy()
    bad.loc[1, ["university", "year"]] = bad.loc[0, ["university", "year"]].to_numpy()
    with pytest.raises(pandera.errors.SchemaError):
        validate_clean(bad)


def test_inverted_rank_band_is_rejected(small_clean):
    bad = small_clean.copy()
    bad.loc[0, "rank_lower"], bad.loc[0, "rank_upper"] = 250, 201
    with pytest.raises(pandera.errors.SchemaError):
        validate_clean(bad)


def test_unexpected_column_is_rejected(small_clean):
    bad = small_clean.assign(secret_internal_kpi=1.0)
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        validate_clean(bad)
