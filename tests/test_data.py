import math

import pandas as pd
import pytest

from urps.data import RAW_COLUMNS, clean, load_clean, load_raw, normalise_name, parse_number, parse_rank


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", (1, 1)),
        ("=45", (45, 45)),
        (" 7 ", (7, 7)),
        ("201-250", (201, 250)),
        ("201–250", (201, 250)),  # en dash, as in copied web tables
        ("1001+", (1001, 1001)),
    ],
)
def test_parse_rank_valid(raw, expected):
    assert parse_rank(raw) == expected


@pytest.mark.parametrize("raw", ["", "-", "abc", "0", "250-201", "=", "12.5"])
def test_parse_rank_invalid(raw):
    with pytest.raises(ValueError):
        parse_rank(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("20,152", 20152.0), ("25%", 25.0), (" 61.4 ", 61.4), ("0", 0.0), (7, 7.0)],
)
def test_parse_number_valid(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["-", "", "n/a", None, float("nan")])
def test_parse_number_missing_is_nan(raw):
    assert math.isnan(parse_number(raw))


def test_parse_number_rejects_text():
    with pytest.raises(ValueError, match="Unrecognised numeric value"):
        parse_number("twelve")


def test_normalise_name_merges_spelling_variants():
    assert normalise_name("  University  of Marlow ") == normalise_name("UNIVERSITY OF MARLOW")


def test_clean_merges_name_variants(small_raw, small_clean):
    # Raw data contains upper-case and padded names; cleaning must not create extra universities.
    assert small_raw["university_name"].nunique() > 40
    assert small_clean["university"].nunique() == 40


def test_clean_removes_duplicates_and_keeps_one_row_per_year(small_clean):
    assert not small_clean.duplicated(["university", "year"]).any()
    assert (small_clean.groupby("university")["year"].count() == 6).all()


def test_clean_keeps_missing_income_as_nan(small_raw, small_clean):
    assert (small_raw["income"] == "-").any()
    assert small_clean["income"].isna().any()
    assert small_clean["teaching"].notna().all()


def test_clean_produces_numeric_columns(small_clean):
    assert pd.api.types.is_integer_dtype(small_clean["rank_lower"])
    assert pd.api.types.is_float_dtype(small_clean["total_score"])
    assert (small_clean["rank_upper"] >= small_clean["rank_lower"]).all()


def test_load_raw_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"world_rank": ["1"], "university_name": ["X"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_raw(path)


def test_load_clean_reads_committed_sample(sample_path):
    df = load_clean(sample_path)
    assert set(RAW_COLUMNS) - {"world_rank", "university_name"} <= set(df.columns)
    assert df["year"].nunique() == 10


def test_clean_raises_on_invalid_rank(small_raw):
    broken = small_raw.copy()
    broken.loc[0, "world_rank"] = "top ten"
    with pytest.raises(ValueError, match="Unrecognised rank"):
        clean(broken)


def test_rows_without_overall_score_are_dropped(small_raw):
    raw = small_raw.drop_duplicates().reset_index(drop=True)
    raw.loc[0, "total_score"] = "-"
    raw.loc[1, "research"] = "-"
    cleaned = clean(raw)
    assert len(cleaned) == len(clean(small_raw)) - 1
    assert cleaned["research"].isna().sum() == 1


def test_extra_raw_columns_are_ignored(small_raw):
    # The public THE file has additional columns such as female_male_ratio.
    cleaned = clean(small_raw.assign(female_male_ratio="48 : 52"))
    assert "female_male_ratio" not in cleaned.columns


def test_load_raw_accepts_uploaded_bytes(sample_path):
    # The dashboard passes uploaded files as in-memory bytes.
    import io

    df = load_raw(io.BytesIO(sample_path.read_bytes()))
    assert len(df) > 0 and "world_rank" in df.columns


def test_upload_demo_file_is_valid():
    from pathlib import Path

    demo = Path(__file__).resolve().parents[1] / "data" / "sample" / "the_rankings_upload_demo.csv"
    df = load_clean(demo)
    assert df["university"].nunique() == 60
    assert sorted(df["year"].unique()) == list(range(2019, 2026))
