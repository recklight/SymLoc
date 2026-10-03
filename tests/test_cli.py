"""End-to-end smoke tests of every command.

The original analysis had no interface, so nothing could be run without editing
source.  These tests exercise each command the way a user would and check that
it leaves the artefacts it promises behind.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from symloc.cli import app

runner = CliRunner()

#: Keeps every command fast enough to run in the suite.
SMALL = [
    "--set",
    "detection.n_trials_per_class=60",
    "--set",
    "detection.n_splits=3",
    "--set",
    "montecarlo.n_trials=10",
]


def _invoke(*arguments: str):
    """Run a command and surface the traceback if it failed."""
    result = runner.invoke(app, list(arguments))
    if result.exit_code != 0 and result.exception is not None:
        raise AssertionError(f"command failed: {arguments}\n{result.output}") from result.exception
    return result


def test_help_lists_every_command():
    result = _invoke("--help")
    for command in (
        "run",
        "sweep",
        "figure",
        "reproduce-legacy",
        "inspect-fig",
        "prepare",
        "train",
        "evaluate",
        "predict",
    ):
        assert command in result.output


def test_run_writes_tables_figures_config_and_manifest(tmp_path):
    result = _invoke("run", "--set", f"output.results_dir={tmp_path}", "--label", "demo")
    assert result.exit_code == 0

    folders = list(tmp_path.iterdir())
    assert len(folders) == 1
    root = folders[0]
    assert root.name.startswith("demo-")

    assert (root / "config.yaml").is_file()
    assert (root / "manifest.json").is_file()
    assert (root / "tables" / "metrics.csv").is_file()
    assert (root / "tables" / "curves.csv").is_file()
    for name in ("distributions", "peak_bars", "peak_ratios", "single_ap_likelihoods"):
        assert (root / "figures" / f"{name}.png").is_file(), name
        assert (root / "figures" / f"{name}.pdf").is_file(), name

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "adjacent_peak_ratios" in manifest["notes"]
    assert "evidence_collapse" in manifest["notes"]
    assert manifest["notes"]["radio_map_degenerate"] is True


def test_run_can_skip_figures(tmp_path):
    _invoke("run", "--set", f"output.results_dir={tmp_path}", "--no-figures")
    root = next(tmp_path.iterdir())
    assert not any((root / "figures").iterdir())


def test_run_reports_the_headline_numbers(tmp_path):
    result = _invoke("run", "--set", f"output.results_dir={tmp_path}")
    assert "peaks" in result.output
    assert "peak ratios" in result.output
    assert "evidence" in result.output


def test_run_accepts_a_config_file(project_root, tmp_path):
    result = _invoke(
        "run",
        "--config",
        str(project_root / "configs" / "one_attack.yaml"),
        "--set",
        f"output.results_dir={tmp_path}",
    )
    assert result.exit_code == 0
    assert "one_attack" in result.output


def test_sweep_writes_a_long_table_and_a_heatmap(tmp_path):
    result = _invoke(
        "sweep",
        "--set",
        f"output.results_dir={tmp_path}",
        "--set",
        "sweep.attack_offsets_db=[-10.0, -20.0]",
        "--metric",
        "js_divergence",
    )
    assert result.exit_code == 0
    root = next(tmp_path.iterdir())
    assert (root / "tables" / "sweep.csv").is_file()
    assert (root / "tables" / "pivot_js_divergence.csv").is_file()
    assert (root / "figures" / "heatmap_js_divergence.png").is_file()


def test_sweep_rejects_an_unknown_metric(tmp_path):
    result = runner.invoke(
        app,
        [
            "sweep",
            "--set",
            f"output.results_dir={tmp_path}",
            "--set",
            "sweep.attack_offsets_db=[-10.0]",
            "--metric",
            "not_a_metric",
        ],
    )
    assert result.exit_code != 0


def test_figure_regenerates_plots_without_tables(tmp_path):
    result = _invoke("figure", "--set", f"output.results_dir={tmp_path}")
    assert result.exit_code == 0
    root = next(tmp_path.iterdir())
    assert (root / "figures" / "distributions.png").is_file()
    assert not (root / "tables").exists() or not any((root / "tables").iterdir())


def test_reproduce_legacy_reports_the_defect_and_its_size():
    result = _invoke("reproduce-legacy")
    assert result.exit_code == 0
    assert "0.0305445" in result.output
    assert "0.0283886" in result.output
    assert "7.59%" in result.output
    assert "HIGHER than the control" in result.output


def test_reproduce_legacy_can_write_a_comparison_figure(tmp_path):
    result = _invoke("reproduce-legacy", "--output", str(tmp_path))
    assert result.exit_code == 0
    assert (tmp_path / "sign_bug_comparison.png").is_file()


def test_inspect_fig_names_the_format_of_a_mislabelled_file(tmp_path):
    path = tmp_path / "wrong.fig"
    path.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n")
    result = runner.invoke(app, ["inspect-fig", str(path)])
    assert result.exit_code == 1
    assert "Encapsulated PostScript" in result.output


def test_inspect_fig_reads_a_real_figure(tmp_path):
    """A handle-graphics tree written by scipy is read back curve for curve."""
    import numpy as np
    from scipy.io import savemat

    path = tmp_path / "plot.fig"
    line = {
        "type": "graph2d.lineseries",
        "properties": {
            "XData": np.arange(5.0),
            "YData": np.array([0.1, 0.3, 0.9, 0.3, 0.1]),
            "DisplayName": "ESP (M=0)",
        },
        "children": np.array([], dtype=object),
    }
    axes = {
        "type": "axes",
        "properties": {},
        "children": np.array([line], dtype=object),
    }
    savemat(str(path), {"hgS_070000": {"type": "figure", "properties": {}, "children": axes}})

    result = _invoke("inspect-fig", str(path))
    assert "ESP (M=0)" in result.output
    assert "0.9" in result.output


def test_prepare_writes_the_labelled_dataset(tmp_path):
    result = _invoke("prepare", *SMALL, "--set", f"output.results_dir={tmp_path}")
    assert result.exit_code == 0
    root = next(tmp_path.iterdir())
    assert (root / "tables" / "dataset.csv").is_file()
    assert "samples" in result.output

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["notes"]["n_samples"] == 120
    assert len(manifest["notes"]["feature_names"]) == manifest["notes"]["n_features"]


def test_train_then_predict_round_trips_through_a_saved_model(tmp_path):
    model_path = tmp_path / "detector.joblib"
    trained = _invoke("train", *SMALL, "--output", str(model_path))
    assert trained.exit_code == 0
    assert model_path.is_file()

    # Magnitudes: a leading minus would be parsed as an option by the shell.
    scored = _invoke("predict", "70,70,110", "--model", str(model_path))
    assert scored.exit_code == 0
    assert "verdict" in scored.output
    assert "position" in scored.output
    assert "-70.0" in scored.output

    # The explicit signed form works after a "--" separator and gives the same
    # verdict; only the log line noting the sign convention differs.
    signed = _invoke("predict", "--model", str(model_path), "--", "-70,-70,-110")
    assert signed.exit_code == 0

    def _verdict_lines(text: str) -> list[str]:
        return [
            line
            for line in text.splitlines()
            if line.startswith(("readings", "verdict", "position"))
        ]

    assert _verdict_lines(signed.output) == _verdict_lines(scored.output)


def test_predict_without_a_model_says_what_to_run_first(tmp_path):
    result = runner.invoke(app, ["predict", "70,70,70", "--model", str(tmp_path / "absent.joblib")])
    assert result.exit_code == 1
    assert "symloc train" in result.output


def test_predict_rejects_unparseable_readings(tmp_path):
    model_path = tmp_path / "detector.joblib"
    _invoke("train", *SMALL, "--output", str(model_path))
    result = runner.invoke(app, ["predict", "not,numbers", "--model", str(model_path)])
    assert result.exit_code != 0


def test_predict_rejects_mixed_sign_readings(tmp_path):
    """Neither convention: taken at face value this scores +70 dBm in silence."""
    model_path = tmp_path / "detector.joblib"
    _invoke("train", *SMALL, "--output", str(model_path))
    result = runner.invoke(app, ["predict", "70,-70,90", "--model", str(model_path)])
    assert result.exit_code != 0
    assert "positive dBm" in result.output


def test_evaluate_reports_the_full_classification_metrics(tmp_path):
    result = _invoke("evaluate", *SMALL, "--set", f"output.results_dir={tmp_path}")
    assert result.exit_code == 0
    for token in ("UAR", "sensitivity", "specificity", "ROC-AUC", "confusion"):
        assert token in result.output

    root = next(tmp_path.iterdir())
    assert (root / "tables" / "cv_folds.csv").is_file()
    assert (root / "figures" / "confusion_matrix.png").is_file()
    assert (root / "figures" / "roc_curve.png").is_file()

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "uar_mean" in manifest["notes"]["summary"]
    assert len(manifest["notes"]["confusion"]) == 2


def test_a_malformed_override_is_rejected(tmp_path):
    result = runner.invoke(app, ["run", "--set", "no_equals_sign"])
    assert result.exit_code != 0


def test_an_unknown_setting_is_rejected(tmp_path):
    result = runner.invoke(app, ["run", "--set", "grid.step_dB=0.5"])
    assert result.exit_code != 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("false", False),
        ("none", None),
        ("0.25", 0.25),
        ("[1, 2]", [1, 2]),
        ("hello", "hello"),
    ],
)
def test_override_values_are_coerced_from_their_text(raw, expected):
    from symloc.cli import _coerce

    assert _coerce(raw) == expected
