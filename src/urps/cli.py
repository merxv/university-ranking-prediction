"""Command-line interface: ``urps generate | run | predict | app``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from urps import __version__


def _cmd_generate(args: argparse.Namespace) -> int:
    from urps.synthetic import generate_the_table

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    years = range(args.first_year, args.first_year + args.years)
    generate_the_table(args.universities, years, seed=args.seed).to_csv(out, index=False)
    print(f"Synthetic table written to {out}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from urps.pipeline import run_experiment

    result = run_experiment(args.data, args.out, args.test_year, args.seed, figures=not args.no_figures)
    best = result["best_model"]
    print(f"Test year {result['test_year']}: best model = {best}")
    for name, m in result["metrics"].items():
        print(f"  {name:<12} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}")
    print(f"MAE improvement over persistence baseline: {result['mae_improvement_over_persistence_pct']}%")
    print(f"Reports written to {args.out}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from urps.pipeline import forecast_university

    print(json.dumps(forecast_university(args.data, args.university, args.seed), indent=2))
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print('error: the dashboard needs extra packages: pip install -e ".[app]"', file=sys.stderr)
        return 2
    dashboard = Path(__file__).with_name("dashboard.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard), "--server.port", str(args.port)]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="urps", description="University Ranking Prediction System")
    parser.add_argument("--version", action="version", version=f"urps {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a synthetic THE-format ranking table")
    gen.add_argument("--out", default="data/sample/the_rankings_sample.csv")
    gen.add_argument("--universities", type=int, default=300)
    gen.add_argument("--first-year", type=int, default=2016)
    gen.add_argument("--years", type=int, default=10)
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=_cmd_generate)

    run = sub.add_parser("run", help="clean data, train and evaluate models, write reports")
    run.add_argument("--data", required=True, help="raw ranking CSV in THE format")
    run.add_argument("--out", default="reports")
    run.add_argument("--test-year", type=int, default=None, help="held-out year (default: latest)")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--no-figures", action="store_true")
    run.set_defaults(func=_cmd_run)

    pred = sub.add_parser("predict", help="forecast the next cycle for one university")
    pred.add_argument("--data", required=True)
    pred.add_argument("--university", required=True)
    pred.add_argument("--seed", type=int, default=42)
    pred.set_defaults(func=_cmd_predict)

    app = sub.add_parser("app", help="open the interactive Streamlit dashboard in the browser")
    app.add_argument("--port", type=int, default=8501)
    app.set_defaults(func=_cmd_app)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
