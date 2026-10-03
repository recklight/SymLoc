"""Divergences and sharpness measures, against closed-form Gaussian results.

Discrete metrics on a fine grid converge to their continuous counterparts, and
for two Gaussians those counterparts are known exactly.  Checking against the
analytic value catches errors a self-consistency test would not -- a metric
implemented with the wrong constant factor still agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from symloc.grid import Grid
from symloc.metrics import (
    MetricError,
    argmax_offset,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    curve_metrics,
    effective_support_width,
    excess_kurtosis,
    full_width_half_maximum,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    peak_difference,
    peak_ratio,
    peak_to_secondary_ratio,
    renyi_entropy,
    shannon_entropy,
    symmetric_kl,
    total_variation,
)
from symloc.normalize import Curve


def _gaussian_pmf(grid: Grid, mean: float, sigma: float) -> np.ndarray:
    """A Gaussian sampled on the grid and normalised to a mass function."""
    values = np.exp(-0.5 * ((grid.values - mean) / sigma) ** 2)
    return values / values.sum()


def _analytic_kl(m1: float, s1: float, m2: float, s2: float) -> float:
    """Closed-form KL divergence between two univariate Gaussians."""
    return float(np.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5)


def _analytic_bhattacharyya(m1: float, s1: float, m2: float, s2: float) -> float:
    """Closed-form Bhattacharyya distance between two univariate Gaussians."""
    first = 0.25 * np.log(0.25 * (s1**2 / s2**2 + s2**2 / s1**2 + 2.0))
    second = 0.25 * (m1 - m2) ** 2 / (s1**2 + s2**2)
    return float(first + second)


@pytest.mark.parametrize(
    ("m1", "s1", "m2", "s2"),
    [(0.0, 5.0, 0.0, 8.0), (-70.0, 6.0, -64.0, 6.0), (10.0, 3.0, -4.0, 7.0)],
)
def test_kl_divergence_matches_the_analytic_gaussian_value(fine_grid, m1, s1, m2, s2):
    p = _gaussian_pmf(fine_grid, m1, s1)
    q = _gaussian_pmf(fine_grid, m2, s2)
    assert kl_divergence(p, q) == pytest.approx(_analytic_kl(m1, s1, m2, s2), rel=1e-4)


@pytest.mark.parametrize(
    ("m1", "s1", "m2", "s2"),
    [(0.0, 5.0, 0.0, 9.0), (-70.0, 6.0, -60.0, 6.0), (2.0, 4.0, -6.0, 5.0)],
)
def test_bhattacharyya_distance_matches_the_analytic_value(fine_grid, m1, s1, m2, s2):
    p = _gaussian_pmf(fine_grid, m1, s1)
    q = _gaussian_pmf(fine_grid, m2, s2)
    expected = _analytic_bhattacharyya(m1, s1, m2, s2)
    assert bhattacharyya_distance(p, q) == pytest.approx(expected, rel=1e-4)


def test_kl_is_asymmetric_and_the_symmetrised_form_adds_both(fine_grid):
    """Reporting one direction alone hides a genuine asymmetry."""
    p = _gaussian_pmf(fine_grid, 0.0, 4.0)
    q = _gaussian_pmf(fine_grid, 0.0, 9.0)
    forward, backward = kl_divergence(p, q), kl_divergence(q, p)
    assert forward != pytest.approx(backward, rel=1e-3)
    assert symmetric_kl(p, q) == pytest.approx(forward + backward, rel=1e-12)


def test_kl_is_infinite_where_the_reference_has_no_support():
    """An impossible candidate stays impossible; it is not smoothed away."""
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == np.inf


def test_js_divergence_is_symmetric_bounded_and_finite_on_disjoint_support():
    p, q = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))
    assert js_divergence(p, q) == pytest.approx(np.log(2.0))
    assert js_divergence(p, q, base=2) == pytest.approx(1.0)


def test_overlap_measures_agree_with_each_other(fine_grid):
    """Hellinger, Bhattacharyya and the coefficient are one quantity in three forms."""
    p = _gaussian_pmf(fine_grid, -70.0, 6.0)
    q = _gaussian_pmf(fine_grid, -66.0, 7.0)
    coefficient = bhattacharyya_coefficient(p, q)
    assert hellinger_distance(p, q) == pytest.approx(np.sqrt(1.0 - coefficient))
    assert bhattacharyya_distance(p, q) == pytest.approx(-np.log(coefficient))


def test_identical_distributions_give_zero_divergence_and_unit_overlap(fine_grid):
    p = _gaussian_pmf(fine_grid, -70.0, 6.0)
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert total_variation(p, p) == pytest.approx(0.0, abs=1e-12)
    assert bhattacharyya_coefficient(p, p) == pytest.approx(1.0)
    # The square root magnifies the round-off in BC: sqrt(2e-16) is about 1.5e-8.
    assert hellinger_distance(p, p) == pytest.approx(0.0, abs=1e-7)


def test_entropy_is_maximal_for_a_uniform_distribution():
    n = 32
    uniform = np.full(n, 1.0 / n)
    assert shannon_entropy(uniform) == pytest.approx(np.log(n))
    assert shannon_entropy(uniform, base=2) == pytest.approx(np.log2(n))
    spiked = np.zeros(n)
    spiked[3] = 1.0
    assert shannon_entropy(spiked) == pytest.approx(0.0, abs=1e-12)


def test_renyi_entropy_reduces_to_shannon_as_alpha_approaches_one(fine_grid):
    p = _gaussian_pmf(fine_grid, 0.0, 5.0)
    assert renyi_entropy(p, alpha=1.0 + 1e-9) == pytest.approx(shannon_entropy(p), rel=1e-5)
    # Collision entropy weights the peak more, so it is never larger.
    assert renyi_entropy(p, alpha=2.0) <= shannon_entropy(p) + 1e-12


def test_full_width_half_maximum_recovers_the_gaussian_value(fine_grid):
    """FWHM of a Gaussian is 2*sqrt(2*ln 2)*sigma."""
    sigma = 6.0
    bell = np.exp(-0.5 * (fine_grid.values / sigma) ** 2)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert full_width_half_maximum(bell, fine_grid) == pytest.approx(expected, rel=1e-3)


def test_full_width_half_maximum_reports_nan_for_a_truncated_curve():
    """A curve that never falls back below half maximum has no measurable width."""
    grid = Grid.from_dbm_sweep(-10.0, 10.0, 0.5)
    ramp = np.linspace(0.6, 1.0, grid.size)
    assert np.isnan(full_width_half_maximum(ramp, grid))


def test_excess_kurtosis_is_zero_for_a_gaussian(fine_grid):
    bell = np.exp(-0.5 * (fine_grid.values / 8.0) ** 2)
    assert excess_kurtosis(bell, fine_grid) == pytest.approx(0.0, abs=1e-3)


def test_peak_to_secondary_ratio_finds_a_competing_mode():
    """An ambiguous posterior is exactly one with a strong second peak."""
    assert peak_to_secondary_ratio([0.0, 1.0, 0.0, 0.5, 0.0]) == pytest.approx(2.0)
    assert peak_to_secondary_ratio([0.0, 1.0, 0.0]) == np.inf


def test_peak_to_secondary_ratio_counts_a_mode_at_the_grid_edge():
    """A spoofed reading pushes mass to the edge, which is where the rival mode sits."""
    assert peak_to_secondary_ratio([0.5, 0.2, 1.0, 0.2, 0.45]) == pytest.approx(2.0)
    # The tallest point itself may be the boundary; the ratio must still be
    # measured from it, not between two lesser interior bumps.
    assert peak_to_secondary_ratio([1.0, 0.5, 0.3, 0.4, 0.2]) == pytest.approx(2.5)


def test_peak_to_secondary_ratio_treats_a_flat_top_as_one_mode():
    """Two candidates tying for the peak is one mode, not zero."""
    assert peak_to_secondary_ratio([0.0, 1.0, 1.0, 0.0, 0.5, 0.0]) == pytest.approx(2.0)
    assert peak_to_secondary_ratio([1.0, 1.0, 1.0]) == np.inf


def test_peak_to_secondary_ratio_reports_a_truncated_posterior_as_ambiguous():
    """The regression this metric exists for: a rival mode running off the grid edge."""
    grid = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    values = np.exp(-0.5 * ((grid.values + 70.0) / 1.0) ** 2)
    values += 0.8 * np.exp(-0.5 * ((grid.values + 92.0) / 4.0) ** 2)
    ratio = peak_to_secondary_ratio(values)
    assert np.isfinite(ratio)
    assert ratio == pytest.approx(values.max() / values[0])


def test_effective_support_width_scales_with_the_grid_step():
    """The width is reported in axis units, so it tracks the step."""
    coarse = Grid.from_dbm_sweep(-10.0, 10.0, 1.0)
    fine = Grid.from_dbm_sweep(-10.0, 10.0, 0.5)
    for grid in (coarse, fine):
        flat = np.ones(grid.size)
        assert effective_support_width(flat, grid, fraction=1.0) == pytest.approx(
            grid.size * grid.step
        )


def test_peak_ratio_and_difference_reproduce_the_original_measures():
    p = np.array([0.1, 0.4, 0.1])
    q = np.array([0.2, 0.2, 0.2])
    # Both are normalised internally, so q's peak is 1/3.
    assert peak_ratio(p, q) == pytest.approx((0.4 / 0.6) / (1.0 / 3.0))
    assert peak_difference(p, q) == pytest.approx(0.4 / 0.6 - 1.0 / 3.0)


def test_argmax_offset_reports_dbm_on_a_sweep_and_metres_on_a_floor():
    sweep = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    curve = np.zeros(sweep.size)
    curve[30] = 1.0
    assert argmax_offset(curve, sweep, true_index=28) == pytest.approx(1.0)

    floor = Grid.rectangular(4.0, 4.0, 2.0)
    curve = np.zeros(floor.size)
    curve[0] = 1.0
    # Point 0 is (0,0) and point 4 is (2,2): 2*sqrt(2) metres apart.
    assert argmax_offset(curve, floor, true_index=4) == pytest.approx(2.0 * np.sqrt(2.0))


def test_curve_metrics_includes_the_evidence_gap_that_normalisation_hides(grid):
    """The absolute likelihood collapse survives into the metric table."""
    weak = Curve.from_log_score("weak", grid, np.full(grid.size, -20.0))
    strong = Curve.from_log_score("strong", grid, np.zeros(grid.size))
    metrics = curve_metrics(weak, reference=strong)
    # Both curves are flat, so they normalise to the same shape and every
    # divergence is zero -- yet one carries e^-20 times the evidence of the
    # other, and only log_evidence_gap records that.
    assert metrics["log_evidence_gap"] == pytest.approx(-20.0)
    assert metrics["js_divergence"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["peak_ratio_to_reference"] == pytest.approx(1.0)
    assert "kl_to_reference" in metrics and "kl_from_reference" in metrics


def test_malformed_distributions_are_rejected():
    with pytest.raises(MetricError, match="negative"):
        kl_divergence([-0.1, 1.1], [0.5, 0.5])
    with pytest.raises(MetricError, match="share a grid"):
        kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])
    with pytest.raises(MetricError, match="sums to zero"):
        shannon_entropy([0.0, 0.0])
    with pytest.raises(MetricError, match="one-dimensional"):
        shannon_entropy([[0.5, 0.5]])
