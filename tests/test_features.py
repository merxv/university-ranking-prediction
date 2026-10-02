import pandas as pd
import pytest

from urps.data import PILLARS
from urps.features import FEATURES, TARGET, build_features, future_features, relative_to_year


@pytest.fixture(scope="module")
def feats(small_clean):
    return build_features(small_clean)


def test_first_year_has_no_rows(feats, small_clean):
    assert feats["year"].min() == small_clean["year"].min() + 1
    assert len(feats) == 40 * 5


def test_no_missing_lag1_values(feats):
    lag1 = [c for c in FEATURES if c.startswith("lag1_") and c not in ("lag1_income",)]
    assert feats[lag1].notna().all().all()


def test_lag_features_come_from_previous_year(feats, small_clean):
    """Leakage test: lag1 values equal the (year-relative) values of year t-1, target equals year t."""
    ref = relative_to_year(small_clean).set_index(["university", "year"])
    for _, row in feats.sample(25, random_state=0).iterrows():
        prev = ref.loc[(row["university"], row["year"] - 1)]
        cur = ref.loc[(row["university"], row["year"])]
        assert row["lag1_total_score"] == pytest.approx(prev["total_score"])
        assert row["lag1_research"] == pytest.approx(prev["research"])
        assert row[TARGET] == pytest.approx(cur["total_score"])


def test_features_do_not_contain_current_year_values(feats, small_clean):
    """No feature column may be identical to the target (a sign of leakage)."""
    for col in FEATURES:
        assert not feats[col].equals(feats[TARGET]), col


def test_history_mean_uses_only_past_years(feats, small_clean):
    uni = feats["university"].iloc[0]
    row = feats[(feats["university"] == uni)].sort_values("year").iloc[-1]
    rel = relative_to_year(small_clean)
    past = rel[(rel["university"] == uni) & (rel["year"] < row["year"])]
    assert row["histmean_total_score"] == pytest.approx(past["total_score"].mean())


def test_gap_in_history_is_not_bridged(small_clean):
    uni = small_clean["university"].iloc[0]
    gapped = small_clean[~((small_clean["university"] == uni) & (small_clean["year"] == 2018))]
    f = build_features(gapped)
    years = set(f.loc[f["university"] == uni, "year"])
    # 2018 is missing, so 2018 (no target) and 2019 (no t-1) must both be absent.
    assert 2018 not in years and 2019 not in years
    assert 2020 in years


def test_future_features_target_next_year(small_clean):
    fut = future_features(small_clean)
    assert (fut["year"] == small_clean["year"].max() + 1).all()
    assert len(fut) == 40
    assert TARGET not in fut.columns


def test_relative_scores_have_zero_mean_per_year(small_clean):
    rel = relative_to_year(small_clean)
    means = rel.groupby("year")[[*PILLARS, "total_score"]].mean()
    assert means.abs().max().max() < 1e-9


def test_common_shift_of_a_year_does_not_change_features_or_target(small_clean):
    """A methodology change that moves every score of one year by the same amount is not predictable
    and does not change anyone's position, so features and targets must be unaffected by it."""
    shifted = small_clean.copy()
    in_2019 = shifted["year"] == 2019
    for col in [*PILLARS, "total_score"]:
        shifted.loc[in_2019, col] = shifted.loc[in_2019, col] + 7.5
    a, b = build_features(small_clean), build_features(shifted)
    cols = [*FEATURES, TARGET]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, atol=1e-9)
    # ...while the yearly averages, used to convert back to the published scale, record the shift
    assert (b.loc[b["year"] == 2019, "year_mean"] - a.loc[a["year"] == 2019, "year_mean"]).iloc[0] == pytest.approx(7.5)


def test_future_features_carry_latest_year_mean(small_clean):
    fut = future_features(small_clean)
    latest = small_clean.loc[small_clean["year"] == small_clean["year"].max(), "total_score"].mean()
    assert fut["prev_year_mean"].iloc[0] == pytest.approx(latest)
