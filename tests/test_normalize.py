"""Normalisation modes, and the diagnostics for what normalising discards."""

from __future__ import annotations

import numpy as np
import pytest

from symloc.ap import APSpec, Scenario
from symloc.grid import Grid
from symloc.normalize import (
    Curve,
    NormalizeError,
    check_truncation,
    grid_resolution_note,
    normalize_log,
    to_pmf,
    truncated_mass,
)


def test_sum_mode_gives_a_mass_function(grid):
    values, _ = normalize_log(np.random.default_rng(0).normal(size=grid.size), mode="sum")
    assert values.sum() == pytest.approx(1.0)
    assert np.all(values >= 0)


def test_density_mode_integrates_to_one(grid):
    values, _ = normalize_log(
        np.random.default_rng(0).normal(size=grid.size), mode="density", step=grid.step
    )
    assert (values * grid.step).sum() == pytest.approx(1.0)


def test_none_mode_preserves_absolute_magnitude(grid):
    log_score = np.full(grid.size, -12.0)
    values, log_total = normalize_log(log_score, mode="none")
    assert np.allclose(values, np.exp(-12.0))
    assert log_total == pytest.approx(np.log(grid.size) - 12.0)


def test_sum_mode_peaks_scale_with_the_step_and_density_peaks_do_not():
    """The reason the grid step belongs in every manifest."""
    coarse = Grid.from_dbm_sweep(-40.0, 40.0, 1.0)
    fine = Grid.from_dbm_sweep(-40.0, 40.0, 0.25)

    def bell(g: Grid) -> np.ndarray:
        return -0.5 * (g.values / 8.0) ** 2

    coarse_mass, _ = normalize_log(bell(coarse), mode="sum")
    fine_mass, _ = normalize_log(bell(fine), mode="sum")
    assert coarse_mass.max() / fine_mass.max() == pytest.approx(4.0, rel=0.01)

    coarse_density, _ = normalize_log(bell(coarse), mode="density", step=coarse.step)
    fine_density, _ = normalize_log(bell(fine), mode="density", step=fine.step)
    assert coarse_density.max() == pytest.approx(fine_density.max(), rel=0.01)


def test_log_evidence_records_the_magnitude_that_normalisation_removes(grid):
    """Two curves of identical shape but different magnitude stay distinguishable."""
    weak = Curve.from_log_score("weak", grid, np.full(grid.size, -30.0))
    strong = Curve.from_log_score("strong", grid, np.zeros(grid.size))
    assert np.allclose(weak.values, strong.values)
    assert weak.log_evidence - strong.log_evidence == pytest.approx(-30.0)


def test_density_mode_requires_a_step():
    with pytest.raises(NormalizeError, match="needs the grid step"):
        normalize_log(np.zeros(4), mode="density")


def test_unknown_mode_lists_the_alternatives():
    with pytest.raises(NormalizeError, match="sum"):
        normalize_log(np.zeros(4), mode="nonsense")


def test_a_curve_with_no_mass_anywhere_is_an_error_not_a_nan():
    """The failure the linear domain produces silently is raised here instead."""
    with pytest.raises(NormalizeError, match="zero probability"):
        normalize_log(np.full(5, -np.inf))


def test_malformed_curves_are_rejected():
    with pytest.raises(NormalizeError, match="one-dimensional"):
        normalize_log(np.zeros((2, 3)))
    with pytest.raises(NormalizeError, match="empty"):
        normalize_log(np.array([]))
    with pytest.raises(NormalizeError, match="nan"):
        normalize_log(np.array([0.0, np.nan]))
    with pytest.raises(NormalizeError, match="negative"):
        to_pmf(np.array([1.0, -1.0]))


def test_to_pmf_handles_exact_zeros(grid):
    """A zero is a legitimate likelihood; only a negative one is not."""
    values = np.zeros(4)
    values[1] = 2.0
    assert to_pmf(values).tolist() == [0.0, 1.0, 0.0, 0.0]


def test_truncated_mass_matches_the_gaussian_tail(grid):
    """The fraction off the grid is exactly computable, so it is computed."""
    from scipy.stats import norm

    scenario = Scenario.build("t", grid, [APSpec("A", -70.0, 5.0)])
    low, high = grid.span
    expected = 1.0 - (norm.cdf((high + 70.0) / 5.0) - norm.cdf((low + 70.0) / 5.0))
    assert truncated_mass(scenario)[0] == pytest.approx(expected)


def test_an_observation_far_off_the_grid_loses_nearly_all_its_mass(grid):
    scenario = Scenario.build("t", grid, [APSpec("A", -70.0, 5.0), APSpec("B", -99.0, 5.0)])
    fractions = truncated_mass(scenario)
    assert fractions[0] < 0.05
    assert fractions[1] > 0.99


def test_truncation_is_measured_against_the_radio_map_not_the_axis_units():
    """On a floor plan the axis is a point index, so the dBm bounds come from the map.

    Comparing a reading in dBm against a reference point index would be a unit
    error that reports 100% truncation for every access point.
    """
    from symloc.pathloss import pathloss_radio_map

    floor = Grid.rectangular(10.0, 10.0, 2.0)
    radio_map = pathloss_radio_map(
        floor, np.array([[0.0, 0.0], [10.0, 10.0]]), ["A", "B"], reference_rssi_dbm=-40.0
    )
    means = radio_map.means_dbm
    aps = [
        APSpec("A", float(means[0].mean()), 6.0),
        APSpec("B", float(means[1].mean()), 6.0),
    ]
    scenario = Scenario.build("floor", floor, aps, radio_map=radio_map)
    fractions = truncated_mass(scenario)
    assert np.all(fractions < 0.9), "a reading inside the modelled means is not truncated away"


def test_check_truncation_warns_only_above_the_threshold(grid, caplog):
    scenario = Scenario.build("t", grid, [APSpec("A", -70.0, 5.0), APSpec("B", -110.0, 5.0)])
    with caplog.at_level("WARNING"):
        report = check_truncation(scenario, warn_above=0.5)
    assert set(report) == {"A", "B"}
    assert "B" in caplog.text
    assert "modelled means" in caplog.text


def test_grid_resolution_note_states_how_peaks_depend_on_the_step(grid):
    assert "0.5 dBm grid" in grid_resolution_note(grid, "sum")
    assert "invariant" in grid_resolution_note(grid, "density")
    assert "unnormalised" in grid_resolution_note(grid, "none")


def test_curve_reports_its_own_argmax_on_the_grid_axis(grid):
    log_score = np.full(grid.size, -50.0)
    log_score[10] = 0.0
    curve = Curve.from_log_score("spike", grid, log_score)
    assert curve.argmax_index == 10
    assert curve.argmax_value == pytest.approx(grid.values[10])
    assert curve.peak == pytest.approx(curve.values.max())
