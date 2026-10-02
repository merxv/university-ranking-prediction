# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Streamlit dashboard (`urps app`): trends, next-cycle forecast with what-if sliders, indicator
  influence, model evaluation, CSV upload and export.
- `urps.analysis` service layer used by the dashboard (framework-independent, fully tested).
- Smaller synthetic file `data/sample/the_rankings_upload_demo.csv` for trying the upload.
- Optional dependency group `app` (streamlit, plotly).
- Metric `r2_change` (R² of the predicted year-over-year change) in CLI, reports, dashboard and the
  CI quality gate, because level R² is near 1 even for the naive baseline.
- Rank metrics: mean rank error and Spearman correlation.
- Year-shift invariance test: moving every score of one year by a constant changes no feature or target.

### Changed
- Scores are modeled relative to the average of their ranking year. On the real THE data the whole
  scale moves by up to ±4 points between editions, which made every model worse than the naive forecast.
- Plain linear regression replaced by ridge regression (`ridge`), which does not overfit on ~180
  universities per year.
- The naive baseline now competes in model selection; if no model beats it, forecasts use it and the
  dashboard explains why.
- Trends tab: universities are chosen with checkboxes in a searchable table instead of a multiselect
  dropdown that stayed open while picking; at most 8 lines are drawn.

## [0.1.0] - 2026-09-30

### Added
- Loading and cleaning of raw THE-format ranking tables: tied ranks, rank bands, separators,
  percentages, missing values, name variants and duplicates.
- pandera schema validation of the cleaned table.
- Leakage-free features from year t-1 (lags, yearly change, 3-year trend, historical mean).
- Persistence baseline, linear regression and gradient boosting with temporal validation,
  90% prediction intervals and THE-style rank labels.
- `urps` command-line interface: `generate`, `run`, `predict`.
- Metrics, predictions, next-cycle forecast, figures and provenance written by `urps run`.
- Synthetic sample dataset for tests and CI.
- Test suite (unit, schema, leakage, property-based, regression, CLI) with coverage gate.
- GitHub Actions CI (quality, test matrix, smoke run) and tag-based release workflow.
