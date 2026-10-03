"""The per-access-point Gaussian likelihood, against closed-form values."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from symloc.ap import APSpec, Scenario
from symloc.likelihood import (
    LikelihoodError,
    likelihood,
    log_likelihood,
    log_likelihood_for_observations,
    scenario_log_likelihood,
)


def test_matches_the_scipy_normal_density(grid):
    """The hand-written log density equals scipy's, to full precision."""
    observations = np.array([-70.0, -78.0, -63.0])
    sigma = np.array([5.0, 8.0, 10.0])
    means = np.tile(grid.values, (3, 1))

    computed = log_likelihood(observations, means, sigma)
    expected = norm.logpdf(observations[:, None], loc=means, scale=sigma[:, None])
    assert np.allclose(computed, expected, rtol=1e-12, atol=0.0)


def test_peak_sits_where_the_mean_equals_the_observation(grid):
    """The likelihood peaks at the candidate whose mean matches the reading."""
    observations = np.array([-70.0])
    means = np.tile(grid.values, (1, 1))
    values = likelihood(observations, means, np.array([6.0]))
    assert grid.values[int(np.argmax(values[0]))] == pytest.approx(-70.0)


def test_peak_height_is_the_gaussian_normalisation(grid):
    """At its peak the density is 1/(sigma*sqrt(2*pi))."""
    sigma = 7.5
    means = np.tile(grid.values, (1, 1))
    values = likelihood(np.array([-70.0]), means, np.array([sigma]))
    assert values.max() == pytest.approx(1.0 / (sigma * np.sqrt(2.0 * np.pi)))


def test_negating_both_observation_and_mean_changes_nothing(grid):
    """The decorative pair of minus signs in the original is a no-op.

    A Gaussian is symmetric about its mean, so evaluating it at -r against a
    mean of -mu gives the same density as r against mu.  This is why the
    original expression was not wrong -- and why dropping only one of the two
    signs, as it did in one term, was.
    """
    observations = np.array([-70.0, -88.0])
    sigma = np.array([8.0, 10.0])
    means = np.tile(grid.values, (2, 1))

    straight = log_likelihood(observations, means, sigma)
    negated = log_likelihood(-observations, -means, sigma)
    assert np.allclose(straight, negated, rtol=1e-12)


def test_dropping_one_sign_annihilates_the_term(grid):
    """Negating the observation but not the mean gives a vanishing density.

    This is the exact shape of the defect in the archived analysis: the term
    falls from about 4e-2 to about 8e-39 and disappears from the sum.
    """
    means = np.tile(grid.values, (1, 1))
    correct = likelihood(np.array([-70.0]), means, np.array([10.0])).max()
    defective = likelihood(np.array([70.0]), means, np.array([10.0])).max()
    assert correct == pytest.approx(0.0398942, rel=1e-5)
    assert defective < 1e-30


def test_batched_form_agrees_with_the_single_vector_form(three_ap_scenario):
    """Scoring a batch gives the same numbers as scoring each vector alone."""
    scenario = three_ap_scenario
    observations = np.vstack([scenario.observed_rssi_dbm, scenario.observed_rssi_dbm - 3.0])
    batched = log_likelihood_for_observations(observations, scenario.radio_map, scenario.sigma_db)
    for index, row in enumerate(observations):
        single = log_likelihood(row, scenario.radio_map.means_dbm, scenario.sigma_db)
        assert np.allclose(batched[index], single, rtol=1e-12)


def test_scenario_helper_returns_one_row_per_access_point(three_ap_scenario):
    block = scenario_log_likelihood(three_ap_scenario)
    assert block.shape == (3, three_ap_scenario.grid.size)
    assert np.all(np.isfinite(block))


def test_zero_sigma_is_rejected_rather_than_producing_infinities(grid):
    """A zero standard deviation would collapse the density to a delta."""
    means = np.tile(grid.values, (2, 1))
    with pytest.raises(LikelihoodError, match="strictly positive"):
        log_likelihood(np.array([-70.0, -70.0]), means, np.array([5.0, 0.0]))


def test_shape_mismatches_are_reported_precisely(grid):
    means = np.tile(grid.values, (3, 1))
    with pytest.raises(LikelihoodError, match="observations but the radio map"):
        log_likelihood(np.array([-70.0, -70.0]), means, np.array([5.0, 5.0]))
    with pytest.raises(LikelihoodError, match="standard deviations"):
        log_likelihood(np.array([-70.0] * 3), means, np.array([5.0]))
    with pytest.raises(LikelihoodError, match="one-dimensional"):
        log_likelihood(np.array([[-70.0]]), means, np.array([5.0]))


def test_non_finite_observation_is_rejected(grid):
    means = np.tile(grid.values, (1, 1))
    with pytest.raises(LikelihoodError, match="finite"):
        log_likelihood(np.array([np.nan]), means, np.array([5.0]))


def test_an_observation_outside_the_grid_is_flagged(grid, caplog):
    """A reading beyond the swept range is warned about, not silently used."""
    aps = [APSpec("A", -70.0, 8.0), APSpec("B", -99.0, 8.0)]
    scenario = Scenario.build("edge", grid, aps)
    assert scenario.out_of_grid_observations() == ("B",)
    with caplog.at_level("WARNING"):
        scenario_log_likelihood(scenario)
    assert "outside the candidate grid" in caplog.text
