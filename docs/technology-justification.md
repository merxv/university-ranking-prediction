# Justification of chosen technologies

This document summarises why each technology was chosen. The full analysis is in the
theoretical report of Assignment 3 (section 2).

## Constraints that drive the choice

- One developer, eight weeks, no paid services (Assignment 1 budget).
- Small tabular dataset: about 300–1,500 universities × 8–10 annual cycles.
- Reproducibility is a required KPI (NFR2); transparency of predictions (NFR3).
- Public GitHub repository with CI/CD required by the assignment.

## Language: Python

Weighted decision matrix (score 1–5):

| Criterion | Weight | Python | R | Julia | Java |
|-----------|-------:|------:|--:|------:|-----:|
| ML and data library ecosystem | 25% | 5 | 4 | 3 | 2 |
| Developer competence | 20% | 5 | 3 | 2 | 3 |
| Community and documentation | 15% | 5 | 4 | 2 | 5 |
| Visualisation and dashboards | 15% | 5 | 4 | 3 | 2 |
| CI/CD and tooling integration | 10% | 5 | 4 | 3 | 5 |
| Performance | 10% | 3 | 3 | 5 | 4 |
| Licence and cost | 5% | 5 | 5 | 5 | 5 |
| **Weighted total** | | **4.80** | 3.75 | 2.95 | 3.30 |

## Libraries and tools

| Area | Choice | Why | Alternative considered |
|------|--------|-----|------------------------|
| Tabular data | pandas | De-facto standard; groupby/merge make lag features explicit | Polars (faster, but unnecessary for this data size) |
| Validation | pandera | Declarative schema in Python; fails fast on bad data (NFR7) | Great Expectations (heavier setup) |
| Models | scikit-learn | One API for baseline, ridge regression and gradient boosting; built-in permutation importance | XGBoost/LightGBM (extra dependency, no gain on small data) |
| Figures | matplotlib | Static PNGs for reports, no server needed | Plotly (planned for the dashboard) |
| CLI | argparse (stdlib) | No extra dependency for three sub-commands | Click, Typer |
| Tests | pytest, pytest-cov, Hypothesis | Fixtures, parametrisation, coverage gate, property-based tests for scientific invariants | unittest |
| Code quality | ruff, mypy | ruff replaces flake8 + isort + black in one fast tool; mypy catches type errors | flake8 + black |
| Environment | venv + `constraints.txt` | Pinned versions without extra tooling; works in CI on Windows and Linux | Conda, Poetry, Docker |
| Version control | Git + GitHub | Distributed, standard; issues, PRs, branch protection, free Actions for public repos | GitLab, Bitbucket |
| CI/CD | GitHub Actions | No infrastructure; integrated with pull requests; OS × Python matrix | GitLab CI, Jenkins |

## Observations from the experiments

The architecture of Assignment 2 expected gradient boosting to be the final model. On the synthetic
sample the **regularized linear model (ridge) wins** (MAE 0.82 vs 0.89 for gradient boosting and 0.96
for the naive baseline): with a few thousand rows and features that are nearly linear in last year's
values, a simple model generalizes better. Plain least squares was replaced by ridge after it overfit
badly on the real THE data (about 180 universities per year).

On the real data, scores also had to be expressed relative to their year's average, because THE
rescales the whole scale between editions, and the naive forecast turned out to be very hard to beat.
The pipeline therefore compares all candidates, including the baseline, on a held-out year and uses
the best one instead of hard-coding a model.
