# Contributing

## Workflow (GitHub Flow)

1. Open or pick an issue; every change starts from a backlog item.
2. Create a short-lived branch from `main`: `feature/<issue>-<short-name>` or `fix/<issue>-<short-name>`.
3. Commit in small steps using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `refactor:`, `chore:`.
4. Open a pull request that references the issue (`Closes #12`) and fill in the checklist.
5. Merge only when all CI checks are green. `main` is protected.
6. Releases are created by pushing a tag `vX.Y.Z` that matches `urps.__version__`.

## Local setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -e ".[dev]" -c constraints.txt
```

## Checks to run before pushing

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Changing results

The regression test compares metrics with `tests/data/reference_metrics.json`. If a change is
*intended* to alter model results, regenerate the reference and explain why in the pull request.
