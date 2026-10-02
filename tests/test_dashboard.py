"""Smoke tests of the Streamlit dashboard with Streamlit's headless AppTest runner."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from urps.cli import main  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "src" / "urps" / "dashboard.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(DASHBOARD, default_timeout=120)
    at.run()
    return at


def test_dashboard_renders_without_errors(app):
    assert not app.exception
    assert not app.error
    assert app.title[0].value == "University Ranking Prediction System"
    assert [t.label for t in app.tabs] == ["Trends", "Forecast & what-if", "Indicator influence", "Model evaluation"]


def test_dashboard_shows_dataset_summary(app):
    labels = {m.label: m.value for m in app.metric}
    assert labels["Universities"] == "300"
    assert labels["Ranking years"] == "2016 - 2025"


def test_what_if_slider_updates_scenario(app):
    slider = next(s for s in app.slider if s.label == "Research")
    slider.set_value(100.0).run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Scenario score" in labels


def test_app_command_starts_streamlit(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.call", lambda cmd: calls.append(cmd) or 0)
    assert main(["app", "--port", "8600"]) == 0
    assert calls[0][1:4] == ["-m", "streamlit", "run"]
    assert calls[0][-1] == "8600"


def test_trends_table_preselects_top_three(app):
    assert not app.exception
    assert not app.warning
    assert any("Tick up to" in c.value for c in app.caption)
