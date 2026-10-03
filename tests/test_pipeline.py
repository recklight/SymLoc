"""The end-to-end scenario pipeline, including the properties it must preserve."""

from __future__ import annotations

import numpy as np
import pytest

from symloc.config import Config
from symloc.pipeline import analyse_scenario, curves_to_frame, evidence_collapse


def test_produces_one_curve_per_requested_order(config):
    result = analyse_scenario(config)
    assert result.m_values == (0, 1, 2)
    for curve in result.curves.values():
        assert curve.values.shape == (result.scenario.grid.size,)
        assert curve.values.sum() == pytest.approx(1.0)


def test_the_metric_table_is_tidy_and_complete(config):
    result = analyse_scenario(config)
    frame = result.metrics
    assert frame.columns.tolist() == ["scenario", "combiner", "m", "metric", "value"]
    assert not frame.empty
    for metric in ("peak", "shannon_entropy", "js_divergence", "log_evidence_gap"):
        assert metric in set(frame["metric"]), metric


def test_the_control_is_computed_by_the_same_code_path(config):
    """The honest baseline comes from the scenario with attacks removed.

    In the original this curve was a separate hard-coded expression, so a
    difference between it and the attacked curves could come from either the
    parameters or the code.  Here only the parameters can differ.
    """
    result = analyse_scenario(config)
    honest = result.scenario.honest()
    assert honest.n_attacked == 0
    assert [ap.name for ap in honest.aps] == [ap.name for ap in result.scenario.aps]
    assert result.baseline.values.sum() == pytest.approx(1.0)


def test_removing_the_attack_makes_every_curve_match_its_control(config):
    """With no tampering the attacked and honest curves are identical."""
    honest_config = config.override({"scenario.attacked_indices": ()})
    result = analyse_scenario(honest_config)
    for order in result.m_values:
        assert np.allclose(
            result.curves[order].values, result.baseline_curves[order].values, rtol=1e-12
        )
    collapse = evidence_collapse(result)
    for order in result.m_values:
        assert collapse[f"log_evidence_gap_m{order}"] == pytest.approx(0.0, abs=1e-12)


def test_an_attack_always_reduces_the_absolute_evidence(config):
    """Tampering can only make the readings fit the map worse, never better.

    This is the claim the whole study rests on, and it is invisible after
    per-curve normalisation -- which is why the pipeline keeps log_evidence.
    """
    result = analyse_scenario(config)
    collapse = evidence_collapse(result)
    for order in result.m_values:
        assert collapse[f"log_evidence_gap_m{order}"] < 0.0
        assert 0.0 < collapse[f"evidence_ratio_m{order}"] < 1.0


def test_higher_m_loses_less_evidence_to_the_attack(config):
    """Tolerating more tampered access points is what M is for."""
    collapse = evidence_collapse(analyse_scenario(config))
    gaps = [collapse[f"log_evidence_gap_m{m}"] for m in (0, 1, 2)]
    assert gaps[0] < gaps[1] < gaps[2]


def test_sharpness_decreases_as_m_increases(config):
    """Robustness is bought with sharpness: the trade-off M actually makes."""
    result = analyse_scenario(config)
    entropies = [
        result.metrics.query("m == @m and metric == 'shannon_entropy'")["value"].item()
        for m in result.m_values
    ]
    assert entropies == sorted(entropies), "a more tolerant combiner should spread out"


def test_localization_error_is_reported_only_when_the_truth_is_known(config):
    without = analyse_scenario(config)
    assert "localization_error" not in set(without.metrics["metric"])

    with_truth = analyse_scenario(config.override({"scenario.true_index": 30}))
    assert "localization_error" in set(with_truth.metrics["metric"])


def test_curves_to_frame_covers_every_curve_and_the_control(config):
    result = analyse_scenario(config)
    frame = curves_to_frame(result)
    assert frame.columns.tolist() == ["scenario", "curve", "m", "x", "value"]
    assert set(frame["m"]) == {-1, 0, 1, 2}
    assert len(frame) == (len(result.m_values) + 1) * result.scenario.grid.size


def test_notes_record_the_degenerate_radio_map_caveat(config):
    """A uniform map means the abscissa is not a location, and the run says so."""
    result = analyse_scenario(config)
    assert result.notes["radio_map_degenerate"] is True
    assert "not a location on a floor" in str(result.notes["radio_map_caveat"])


def test_a_pathloss_scenario_is_not_flagged_as_degenerate(project_root):
    config = Config.from_yaml(project_root / "configs" / "pathloss_floor.yaml")
    result = analyse_scenario(config)
    assert result.notes["radio_map_degenerate"] is False
    assert "radio_map_caveat" not in result.notes


def test_a_pathloss_scenario_localizes_exactly_when_nothing_is_tampered_with(project_root):
    """The geometric pipeline is correct: honest readings recover the true point."""
    config = Config.from_yaml(project_root / "configs" / "pathloss_floor.yaml")
    honest = config.override({"scenario.attacked_indices": ()})
    result = analyse_scenario(honest)
    errors = result.metrics.query("metric == 'localization_error'")["value"]
    assert len(errors) == result.scenario.n_ap
    assert errors.to_numpy() == pytest.approx(0.0)


def test_attack_strength_can_be_overridden_per_call(config):
    """The sweep driver varies the offset without rebuilding the configuration."""
    mild = analyse_scenario(config, attack_offset_db=-2.0)
    severe = analyse_scenario(config, attack_offset_db=-40.0)
    mild_gap = evidence_collapse(mild)["log_evidence_gap_m0"]
    severe_gap = evidence_collapse(severe)["log_evidence_gap_m0"]
    assert severe_gap < mild_gap


def test_every_registered_combiner_runs_through_the_pipeline(config):
    from symloc.combiners import available_combiners

    for name in available_combiners():
        result = analyse_scenario(config.override({"scenario.combiner": name}))
        assert len(result.curves) >= 1
        for curve in result.curves.values():
            assert np.all(np.isfinite(curve.values))
            assert curve.values.sum() == pytest.approx(1.0)


def test_adjacent_peak_measures_match_the_metric_table(config):
    result = analyse_scenario(config)
    ratios = result.adjacent_peak_ratios()
    assert set(ratios) == {"rho_0_1", "rho_1_2"}
    tabulated = result.metrics.query("m == 0 and metric == 'peak_ratio_vs_m1'")["value"].item()
    assert tabulated == pytest.approx(ratios["rho_0_1"])
