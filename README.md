# University Ranking Prediction System (URPS)

[![CI](https://github.com/merxv/university-ranking-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/merxv/university-ranking-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

A small, reproducible Python tool that turns historical university ranking tables into
**next-cycle score and rank predictions**, with uncertainty intervals and indicator influence.
It is the practical part of Assignment 3 (*Software Development and Integration*) of the Applied
Software Development course, building on the project charter, plan (Assignment 1) and architecture
(Assignment 2) of the University Ranking Prediction System.

> Predictions are estimates based on public indicators. They explain correlation, not causation,
> and are not a guarantee of any university's actual ranking.

## What it does

| Step | Module | Requirement |
|------|--------|-------------|
| Read raw THE-format CSV, parse tied ranks (`=45`), rank bands (`201-250`), `20,152`, `25%`, `-` | `urps.data` | FR1, FR2 |
| Merge name variants, drop duplicates and rows without an overall score | `urps.data` | FR2 |
| Validate the cleaned table against a declarative schema; stop on violation | `urps.schema` | NFR7 |
| Build features **only from year t-1**: lags, yearly change, 3-year trend, historical mean | `urps.features` | FR3 |
| Express scores relative to their year's average; train a persistence baseline, ridge regression and gradient boosting; temporal split | `urps.features`, `urps.model` | FR4 |
| Report MAE, RMSE, R², R² of yearly change, 90% interval coverage; forecast next cycle with rank labels | `urps.pipeline` | FR5, FR7 |
| Permutation importance and figures | `urps.viz` | FR8, FR6 |

User stories covered: US1 (trend plots), US2 (next-cycle forecast), US3 (indicator influence).

## Quick start

```bash
git clone https://github.com/merxv/university-ranking-prediction.git
cd university-ranking-prediction
python -m venv .venv
.venv/Scripts/activate            # Windows; `source .venv/bin/activate` on Linux/macOS
pip install -e ".[app]" -c constraints.txt
```

### Interactive dashboard

```bash
urps app
```

This opens <http://localhost:8501> with four tabs:

| Tab | What you can do | User story |
|-----|-----------------|------------|
| Trends | Compare universities over the years by overall score or any indicator | US1, FR6, FR11 |
| Forecast & what-if | Next-cycle score, 90% interval and rank; move sliders to change this year's indicators and see the effect | US2, FR5, FR9 |
| Indicator influence | Which indicators drive the prediction (permutation importance) | US3, FR8 |
| Model evaluation | MAE / RMSE / R² of all models on the held-out year, predicted vs actual, CSV export | FR7, FR10 |

The sidebar switches between the built-in synthetic sample and **your own CSV upload**
(try `data/sample/the_rankings_upload_demo.csv`, or real data, see [Data](#data)).

![Dashboard](docs/figures/dashboard.png)

### Command line

Run the whole pipeline on the committed sample:

```bash
urps run --data data/sample/the_rankings_sample.csv --out reports
```

```
Test year 2025: best model = ridge
Common shift of the score scale in the test year: +0.06 (removed)
  persistence  MAE=0.958  rank error=4.9  Spearman=0.997  R2=0.995  R2 of yearly change=-0.000
  ridge        MAE=0.816  rank error=4.4  Spearman=0.998  R2=0.996  R2 of yearly change=+0.275
  gbm          MAE=0.886  rank error=4.6  Spearman=0.998  R2=0.996  R2 of yearly change=+0.140
MAE improvement over persistence baseline: 14.8%
```

> **Why is R² ≈ 0.99 even for the naive baseline?** Scores differ far more *between* universities
> (standard deviation ≈ 17 points) than *from one year to the next* (≈ 1 point). R² compares a model
> with predicting the average score of all universities, so simply repeating last year's score already
> gives R² = 0.995. This is not leakage (see the leakage tests), but it means level R² says little about
> skill. The pipeline therefore also reports **R² of yearly change**: the share of the actual
> year-over-year change the model explains. The baseline scores ≈ 0 by construction; the ridge model
> explains about 27% of the change, which is the honest measure of what it adds. **Rank error** (how
> many places a predicted position is off) and **Spearman** correlation measure what users care about.

### Results on real data

On the public Times Higher Education data 2011–2016 (`timesData.csv`, see [Data](#data)):

| Test year | Common scale shift | Persistence MAE / rank error | Best model | Best MAE / rank error |
|-----------|-------------------:|-----------------------------:|------------|----------------------:|
| 2014 | −3.84 | 1.91 / 9.8 | persistence | — |
| 2015 | +1.21 | 1.24 / 7.7 | persistence | — |
| 2016 | +3.58 | 3.23 / 17.5 | ridge | 3.18 / 17.2 |

Two lessons came from the real data. First, THE rescales scores between editions (and changed its
methodology in 2016), shifting the **whole scale** by up to ±4 points. That shift is unpredictable and
does not change anyone's position, so all scores are modeled **relative to the average of their year**;
without this, every model was worse than the naive forecast (MAE 4.4–5.4 in 2016). Second, with about
180 universities per year and only five year-to-year transitions, positions are so stable that the
naive *same position as last year* forecast is very hard to beat. The pipeline therefore lets the
baseline compete: when no model beats it on the held-out year, forecasts use it, and the dashboard says so.

Forecast one university for the next, unpublished cycle:

```bash
urps predict --data data/sample/the_rankings_sample.csv --university "National Fairhaven University"
```

```json
{
  "university": "National Fairhaven University",
  "year": 2026,
  "predicted_score": 99.05,
  "interval_90": [97.43, 100.0],
  "predicted_rank": "1",
  "model": "ridge",
  "note": "Estimate based on public indicators, not a guarantee of the actual ranking."
}
```

Other commands: `urps generate` (write a new synthetic table), `urps --help`.

### Outputs of `urps run`

| File | Content |
|------|---------|
| `reports/metrics.json` | Metrics per model, best model, improvement over baseline, top features, provenance (data SHA-256, versions, seed) |
| `reports/predictions.csv` | Test-year predictions with 90% interval and predicted rank |
| `reports/forecast.csv` | Next-cycle forecast for every university |
| `reports/figures/*.png` | Predicted vs actual, feature importance, model comparison |

<p align="center">
  <img src="docs/figures/predicted_vs_actual.png" width="45%" alt="Predicted vs actual">
  <img src="docs/figures/feature_importance.png" width="50%" alt="Feature importance">
</p>

## Data

`data/sample/the_rankings_sample.csv` is a **synthetic** table (300 universities × 10 years) generated
by `urps.synthetic`. It copies the column layout and the formatting problems of the public Times Higher
Education data, but no real ranking data is redistributed. Metrics above therefore show that the
pipeline works; they say nothing about how predictable real rankings are.

`data/sample/the_rankings_upload_demo.csv` is a smaller synthetic file (60 universities, 2019–2025)
for trying the upload feature of the dashboard.

**Real data.** The public Kaggle dataset
[World University Rankings](https://www.kaggle.com/datasets/mylesoneill/world-university-rankings)
(Times Higher Education 2011–2016) has a file `timesData.csv` with the expected layout. A free Kaggle
account is needed to download it. Put the file into `data/raw/` (git-ignored) and either upload it in
the dashboard or run `urps run --data data/raw/timesData.csv`. Universities without a published overall
score (positions below 200 in that file) are dropped automatically. The loader was written for this
layout but has not been tested on the real file in CI.

Any CSV with the columns `world_rank, university_name, country, teaching, international, research,
citations, income, total_score, num_students, student_staff_ratio, international_students, year`
works; extra columns are ignored. Respect the terms of use of the data source.

## Project structure

```
├── src/urps/            # package: data, schema, features, model, pipeline, analysis, dashboard, viz, cli, synthetic
├── tests/               # pytest suite + reference metrics for the regression test
├── data/sample/         # synthetic sample used by tests and CI
├── docs/                # technology justification and figures
├── .github/workflows/   # ci.yml (integration), release.yml (delivery)
├── pyproject.toml       # metadata, dependencies, pytest/ruff/mypy configuration
└── constraints.txt      # pinned versions for reproducible installs
```

## Testing

```bash
pip install -e ".[dev,app]" -c constraints.txt
pytest
```

The suite (92 tests, coverage gate 80%, currently ~99%) contains:

- **unit tests** for parsers and cleaning (tied ranks, bands, separators, name variants, duplicates);
- **schema tests**: out-of-range, missing target, duplicates and unexpected columns are rejected;
- **leakage tests**: lag features equal year t-1 values, history gaps are never bridged;
- **property-based tests** (Hypothesis): parser round-trips, rank labels are monotonic in score;
- **regression test**: metrics must stay within tolerance of `tests/data/reference_metrics.json`;
- **reproducibility test**: two runs with the same seed give identical metrics;
- **service tests** for forecasts and what-if scenarios;
- **dashboard smoke tests** with Streamlit's headless `AppTest` (renders, slider updates the scenario);
- **CLI tests** and doctests.

## CI/CD

| Workflow | Trigger | Stages |
|----------|---------|--------|
| [`ci.yml`](.github/workflows/ci.yml) | push to `main`, every pull request | **quality** (ruff lint + format, mypy) → **test** (Ubuntu/Windows × Python 3.11/3.12, coverage) → **smoke** (full pipeline on the sample; quality gate: R² ≥ 0.85, explains part of the yearly change, beats the baseline; reports uploaded as artifact) |
| [`release.yml`](.github/workflows/release.yml) | tag `v*` | tag = package version check → tests → build sdist/wheel → GitHub Release with packages and reports |

`main` is protected: changes arrive through pull requests with green checks (GitHub Flow, see
[CONTRIBUTING.md](CONTRIBUTING.md)). Dependabot keeps actions and dependencies up to date.

## Technology choices

Python 3.11+, pandas, scikit-learn, pandera, matplotlib, Streamlit + Plotly, pytest + Hypothesis,
ruff, mypy, GitHub Actions. The reasoning (weighted decision matrix, alternatives considered) is in
[docs/technology-justification.md](docs/technology-justification.md).

## Limitations

- The sample data is synthetic; real ranking methodologies change over time and need per-source normalisation.
- Permutation importance is shared between strongly correlated features
  (e.g. last year's score and its historical mean); read it as a ranking of influence, not as effect sizes.
- The 90% interval is estimated from one held-out calibration year and is approximate.
- Not included yet (planned in the architecture of Assignment 2): MLflow tracking, DVC.

## Citation and licence

Cite via [CITATION.cff](CITATION.cff). Released under the [MIT License](LICENSE).
Author: Kulbossynov Alisher.
