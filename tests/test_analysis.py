import pytest

from urps.analysis import build_bundle, feature_label, latest_indicators, what_if
from urps.model import MODEL_NAMES


@pytest.fixture(scope="module")
def bundle(small_clean):
    return build_bundle(small_clean, seed=1)


def test_bundle_contents(bundle, small_clean):
    assert set(bundle.metrics) == set(MODEL_NAMES)
    assert bundle.test_year == small_clean["year"].max()
    assert len(bundle.forecast) == small_clean["university"].nunique()
    assert (bundle.forecast["year"] == small_clean["year"].max() + 1).all()
    assert bundle.forecast["predicted_score"].is_monotonic_decreasing
    assert bundle.model.name in MODEL_NAMES
    assert not bundle.importance.empty


def test_latest_indicators_unknown_university(small_clean):
    with pytest.raises(ValueError, match="not found"):
        latest_indicators(small_clean, "Unknown University")


def test_what_if_without_effective_change_equals_base(bundle):
    uni = bundle.forecast["university"].iloc[5]
    current = float(latest_indicators(bundle.clean, uni)["research"])
    result = what_if(bundle, uni, {"research": current})
    assert result["scenario_score"] == pytest.approx(result["base_score"])
    assert result["change"] == pytest.approx(0.0)


def test_what_if_better_indicators_raise_the_forecast(bundle):
    uni = bundle.forecast["university"].iloc[-1]  # weakest university, room to improve
    row = latest_indicators(bundle.clean, uni)
    better = {p: min(100.0, float(row[p]) + 20) for p in ("research", "citations", "teaching")}
    result = what_if(bundle, uni, better)
    assert result["change"] > 0
    assert result["scenario_lower"] <= result["scenario_score"] <= result["scenario_upper"]


def test_what_if_does_not_modify_original_data(bundle):
    uni = bundle.forecast["university"].iloc[0]
    before = bundle.clean.copy()
    what_if(bundle, uni, {"teaching": 1.0})
    assert bundle.clean.equals(before)


@pytest.mark.parametrize(
    ("changes", "message"),
    [({"total_score": 90.0}, "Only ranking indicators"), ({"research": 140.0}, "between 0 and 100")],
)
def test_what_if_rejects_invalid_changes(bundle, changes, message):
    uni = bundle.forecast["university"].iloc[0]
    with pytest.raises(ValueError, match=message):
        what_if(bundle, uni, changes)


def test_feature_label():
    assert feature_label("lag1_research") == "Research (last year)"
    assert feature_label("trend3_total_score") == "Overall score (3-year trend)"
    assert feature_label("lag1_log_students") == "Log students (last year)"
    assert feature_label("other") == "other"


def test_persistence_fallback_has_trivial_importance(small_clean):
    # On 40 universities no model beats the naive forecast, so it is selected honestly.
    b = build_bundle(small_clean, seed=1)
    if b.model.name == "persistence":
        assert list(b.importance.index) == ["lag1_total_score"]
