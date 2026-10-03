"""Result persistence, run manifests, figure reading and parameter sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symloc.config import Config
from symloc.figio import (
    FigCurve,
    FigDocument,
    FigFormatError,
    compare_to_reference,
    detect_format,
    read_fig,
)
from symloc.io_ import config_hash, prepare_run, write_manifest, write_table
from symloc.sweep import SweepError, build_sweep_configs, discriminability_pivot, run_sweep


class TestRunPersistence:
    def test_the_hash_is_stable_and_sensitive(self):
        """Same settings, same folder; different settings, different folder."""
        assert config_hash(Config()) == config_hash(Config())
        assert config_hash(Config()) != config_hash(Config().override({"grid.step_db": 0.25}))
        assert len(config_hash(Config())) == 12

    def test_the_run_folder_is_named_after_its_configuration(self, tmp_path):
        """A filename can never disagree with its parameters if it is derived from them."""
        config = Config().override({"output.results_dir": str(tmp_path)})
        paths = prepare_run(config, "demo")
        assert paths.root.name == f"demo-{config_hash(config)}"
        assert paths.figures.is_dir() and paths.tables.is_dir()

    def test_an_awkward_label_is_made_safe(self, tmp_path):
        config = Config().override({"output.results_dir": str(tmp_path)})
        paths = prepare_run(config, "my run/2012?")
        assert "/" not in paths.root.name and "?" not in paths.root.name

    def test_tables_are_written_as_csv_and_read_back(self, tmp_path):
        frame = pd.DataFrame({"m": [0, 1], "value": [0.5, 0.25]})
        written = write_table(frame, tmp_path / "metrics", write_parquet=False)
        assert [p.suffix for p in written] == [".csv"]
        pd.testing.assert_frame_equal(pd.read_csv(written[0]), frame)

    def test_a_missing_parquet_engine_degrades_to_csv(self, tmp_path, monkeypatch, caplog):
        """The CSV is always written, so a missing optional dependency is not fatal."""

        def _refuse(*args, **kwargs):
            raise ImportError("no parquet engine")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", _refuse)
        with caplog.at_level("WARNING"):
            written = write_table(pd.DataFrame({"a": [1]}), tmp_path / "t", write_parquet=True)
        assert [p.suffix for p in written] == [".csv"]
        assert "Parquet" in caplog.text

    def test_the_manifest_records_settings_versions_and_seeds(self, tmp_path):
        """The record whose absence makes the archived figures unreproducible."""
        config = Config()
        path = write_manifest(tmp_path / "manifest.json", config, extra={"note": np.float64(1.5)})
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["config"] == config.to_dict()
        assert payload["config_hash"] == config_hash(config)
        assert payload["seeds"]["montecarlo"] == config.montecarlo.seed
        assert payload["environment"]["numpy"] == np.__version__
        assert "python" in payload["environment"]
        assert payload["created_utc"].endswith("+00:00")
        # numpy scalars must survive JSON serialisation.
        assert payload["notes"]["note"] == 1.5

    def test_a_saved_configuration_reproduces_the_same_hash(self, tmp_path):
        config = Config().override({"scenario.combiner": "trimmed"})
        path = config.save(tmp_path / "config.yaml")
        assert config_hash(Config.from_yaml(path)) == config_hash(config)


class TestFigReader:
    def test_eps_wearing_a_fig_extension_is_named_for_what_it_is(self, tmp_path):
        """The archive contains one such file; an opaque parser error is useless."""
        path = tmp_path / "wrong.fig"
        path.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%Creator: test\n")
        assert detect_format(path) == "eps"
        with pytest.raises(FigFormatError, match="Encapsulated PostScript"):
            read_fig(path)

    def test_an_unrecognised_file_is_rejected(self, tmp_path):
        path = tmp_path / "junk.fig"
        path.write_bytes(b"not a figure at all")
        assert detect_format(path) == "unknown"
        with pytest.raises(FigFormatError, match="MAT-file"):
            read_fig(path)

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FigFormatError, match="not found"):
            read_fig(tmp_path / "absent.fig")

    def test_a_data_matfile_is_distinguished_from_a_figure(self, tmp_path):
        """A MAT-file with no handle-graphics tree is data, not a plot."""
        from scipy.io import savemat

        path = tmp_path / "data.fig"
        savemat(str(path), {"some_array": np.arange(5.0)})
        assert detect_format(path) == "mat5"
        with pytest.raises(FigFormatError, match="data MAT-file"):
            read_fig(path)

    def test_peaks_are_keyed_by_legend_entry_and_unnamed_curves_survive(self):
        document = FigDocument(
            path=Path("demo.fig"),
            curves=(
                FigCurve("ESP (M=0)", np.arange(3.0), np.array([0.1, 0.4, 0.1])),
                FigCurve("", np.arange(3.0), np.array([0.2, 0.2, 0.2])),
            ),
        )
        peaks = document.peaks()
        assert peaks["ESP (M=0)"] == pytest.approx(0.4)
        # A line drawn by hand at the prompt has no name but is still recorded.
        assert peaks["curve_1"] == pytest.approx(0.2)
        assert "2 curve(s)" in document.summary()

    def test_comparison_against_expected_peaks_reports_per_curve(self):
        document = FigDocument(
            Path("demo.fig"),
            (FigCurve("M=0", np.arange(3.0), np.array([0.1, 0.4, 0.1])),),
        )
        report = compare_to_reference(document, {"M=0": 0.4, "missing": 1.0})
        assert report["M=0"]["matches"] == 1.0
        assert report["missing"]["matches"] == 0.0


class TestSweep:
    def test_the_axes_expand_to_a_cartesian_product(self):
        config = Config().override(
            {
                "sweep.attack_offsets_db": (-10.0, -20.0),
                "sweep.sigmas_db": (4.0, 8.0),
                "sweep.combiners": ("esp", "trimmed"),
            }
        )
        points = build_sweep_configs(config)
        assert len(points) == 2 * 2 * 2
        coordinates = {tuple(sorted(c.items())) for c, _ in points}
        assert len(coordinates) == 8

    def test_sweeping_the_access_point_count_keeps_one_honest(self):
        """M must always leave something to localise with."""
        config = Config().override(
            {"sweep.attack_offsets_db": (-20.0,), "sweep.n_ap_values": (2, 5, 9)}
        )
        for coordinates, point in build_sweep_configs(config):
            scenario = point.build_scenario()
            assert scenario.n_ap == coordinates["n_ap"]
            assert 1 <= scenario.n_attacked <= scenario.n_ap - 1

    def test_sweeping_sigma_applies_it_to_every_access_point(self):
        config = Config().override({"sweep.attack_offsets_db": (-20.0,), "sweep.sigmas_db": (7.0,)})
        _, point = build_sweep_configs(config)[0]
        assert {ap.sigma_db for ap in point.scenario.aps} == {7.0}

    def test_scenario_names_encode_their_coordinates(self):
        config = Config().override({"sweep.attack_offsets_db": (-20.0,)})
        _, point = build_sweep_configs(config)[0]
        assert "off-20dB" in point.scenario.name

    def test_a_sweep_produces_one_long_format_table(self):
        config = Config().override({"sweep.attack_offsets_db": (-10.0, -20.0)})
        frame = run_sweep(config)
        assert frame.columns.tolist() == [
            "attack_offset_db",
            "sigma_db",
            "n_ap",
            "combiner",
            "scenario",
            "m",
            "metric",
            "value",
        ]
        assert sorted(frame["attack_offset_db"].unique()) == [-20.0, -10.0]
        assert "log_evidence_gap" in set(frame["metric"])

    def test_the_pivot_is_attack_strength_by_order(self):
        config = Config().override({"sweep.attack_offsets_db": (-10.0, -20.0, -30.0)})
        table = discriminability_pivot(run_sweep(config), "js_divergence")
        assert table.shape == (3, 3)
        assert table.index.name == "attack_offset_db"

    def test_a_stronger_attack_moves_the_posterior_further_from_the_control(self):
        """The property a discriminability heatmap is supposed to show."""
        config = Config().override({"sweep.attack_offsets_db": (-5.0, -15.0, -30.0)})
        table = discriminability_pivot(run_sweep(config), "js_divergence")
        column = table[0]  # M = 0
        assert column.loc[-30.0] > column.loc[-15.0] > column.loc[-5.0]

    def test_an_unknown_metric_lists_the_available_ones(self):
        config = Config().override({"sweep.attack_offsets_db": (-10.0,)})
        with pytest.raises(SweepError, match="available metrics"):
            discriminability_pivot(run_sweep(config), "not_a_metric")

    def test_a_sweep_of_one_access_point_is_rejected(self):
        from symloc.config import ConfigError

        with pytest.raises(ConfigError, match="at least two access points"):
            Config().override({"sweep.n_ap_values": (1,)})
