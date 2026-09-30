import json

from urps.cli import main


def test_generate_and_run(tmp_path, capsys):
    data = tmp_path / "raw.csv"
    assert main(["generate", "--out", str(data), "--universities", "30", "--years", "5"]) == 0
    assert data.exists()
    out = tmp_path / "reports"
    assert main(["run", "--data", str(data), "--out", str(out), "--no-figures"]) == 0
    printed = capsys.readouterr().out
    assert "best model" in printed
    assert json.loads((out / "metrics.json").read_text())["n_test"] == 30


def test_predict_prints_json(sample_path, sample_university, capsys):
    assert main(["predict", "--data", str(sample_path), "--university", sample_university]) == 0
    assert json.loads(capsys.readouterr().out)["year"] == 2026


def test_errors_return_exit_code_2(tmp_path, capsys):
    assert main(["run", "--data", str(tmp_path / "missing.csv")]) == 2
    assert "error" in capsys.readouterr().err
