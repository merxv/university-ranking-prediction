# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
