import pytest

from urps.viz import plot_trend


def test_plot_trend_writes_png(small_clean, tmp_path):
    uni = small_clean["university"].iloc[0]
    path = plot_trend(small_clean, uni, tmp_path / "trend.png")
    assert path.stat().st_size > 0


def test_plot_trend_unknown_university(small_clean, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        plot_trend(small_clean, "Unknown", tmp_path / "x.png")
