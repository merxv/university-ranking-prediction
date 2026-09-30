import pytest

from urps.features import FEATURES, TARGET, build_features, future_features


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
    """Leakage test: lag1 values equal the published values of year t-1, target equals year t."""
    ref = small_clean.set_index(["university", "year"])
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
    past = small_clean[(small_clean["university"] == uni) & (small_clean["year"] < row["year"])]
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
