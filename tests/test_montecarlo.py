"""Repeated trials, their summaries, and the reproducibility the seed buys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from symloc.combiners import combine_many
from symloc.config import Config
from symloc.montecarlo import (
    MonteCarloError,
    batched_combine,
    resolve_true_index,
    run_montecarlo,
    sample_observations,
    summarise,
)


def test_batched_combine_matches_combining_each_trial_separately(rng):
    """The folding trick that vectorises the batch must not change any number."""
    n_trials, n_ap, n_points = 6, 4, 9
    log_p = rng.normal(size=(n_trials, n_ap, n_points))
    orders = (0, 1, 2, 3)

    batched = batched_combine(log_p, orders)
    for trial in range(n_trials):
        single = combine_many(log_p[trial], orders)
        for order in orders:
            assert np.allclose(batched[order][trial], single[order], rtol=1e-12)


def test_batched_combine_rejects_the_wrong_rank():
    with pytest.raises(MonteCarloError, match="3-D"):
        batched_combine(np.zeros((3, 4)), (0,))


def test_the_true_index_falls_back_to_the_honest_signal_level(config):
    scenario = config.build_scenario()
    assert scenario.grid.values[resolve_true_index(scenario)] == pytest.approx(-70.0)


def test_an_explicit_true_index_is_honoured(config):
    scenario = config.override({"scenario.true_index": 7}).build_scenario()
    assert resolve_true_index(scenario) == 7


def test_sampled_observations_are_centred_on_the_means_plus_the_attack(config, rng):
    """Honest access points scatter about the map; tampered ones are shifted."""
    scenario = config.build_scenario()
    draws = sample_observations(scenario, 40000, rng)
    assert draws.shape == (40000, scenario.n_ap)

    index = resolve_true_index(scenario)
    expected = scenario.radio_map.means_dbm[:, index] + np.array(
        [ap.attack_offset_db if ap.attacked else 0.0 for ap in scenario.aps]
    )
    assert np.allclose(draws.mean(axis=0), expected, atol=0.2)
    assert np.allclose(draws.std(axis=0), scenario.sigma_db, rtol=0.05)


def test_sampling_is_reproducible_from_the_seed(config):
    scenario = config.build_scenario()
    first = sample_observations(scenario, 20, np.random.default_rng(99))
    second = sample_observations(scenario, 20, np.random.default_rng(99))
    assert np.array_equal(first, second)


def test_zero_trials_is_rejected(config, rng):
    with pytest.raises(MonteCarloError, match="at least 1"):
        sample_observations(config.build_scenario(), 0, rng)


def test_the_whole_run_is_reproducible_from_the_configured_seed(config):
    """Same seed, same table -- which is what makes a reported interval checkable."""
    first = run_montecarlo(config)
    second = run_montecarlo(config)
    pd.testing.assert_frame_equal(first.summary, second.summary)


def test_a_different_seed_gives_a_different_sample(config):
    first = run_montecarlo(config)
    second = run_montecarlo(config.override({"montecarlo.seed": 12345}))
    assert not np.allclose(first.samples["value"].to_numpy(), second.samples["value"].to_numpy())


def test_the_summary_brackets_the_mean_with_its_interval(config):
    result = run_montecarlo(config)
    peaks = result.summary.query("metric == 'peak'")
    assert not peaks.empty
    for _, row in peaks.iterrows():
        assert row["ci_low"] <= row["mean"] <= row["ci_high"]
        assert row["n_trials"] == config.montecarlo.n_trials


def test_the_summary_has_one_row_per_metric_and_order(config):
    result = run_montecarlo(config)
    counts = result.samples.groupby(["m", "metric"]).size().unique()
    assert counts.tolist() == [config.montecarlo.n_trials]
    assert len(result.summary) == result.samples[["m", "metric"]].drop_duplicates().shape[0]


def test_infinities_are_excluded_rather_than_poisoning_the_mean():
    """An unimodal curve has no secondary peak, and that must not become inf."""
    frame = pd.DataFrame(
        {
            "trial": [0, 1, 2],
            "m": [0, 0, 0],
            "metric": ["peak_to_secondary_ratio"] * 3,
            "value": [2.0, np.inf, 4.0],
        }
    )
    summary = summarise(frame, "demo", "esp")
    assert summary["mean"].item() == pytest.approx(3.0)
    assert summary["n_trials"].item() == 2


def test_the_summary_columns_are_stable(config):
    result = run_montecarlo(config)
    assert result.summary.columns.tolist() == [
        "scenario",
        "combiner",
        "m",
        "metric",
        "mean",
        "std",
        "ci_low",
        "ci_high",
        "n_trials",
    ]


def test_a_wider_confidence_level_gives_a_wider_interval():
    frame = pd.DataFrame(
        {
            "trial": list(range(100)),
            "m": [0] * 100,
            "metric": ["peak"] * 100,
            "value": np.linspace(0.0, 1.0, 100),
        }
    )
    narrow = summarise(frame, "d", "esp", confidence=0.50)
    wide = summarise(frame, "d", "esp", confidence=0.99)
    narrow_width = narrow["ci_high"].item() - narrow["ci_low"].item()
    wide_width = wide["ci_high"].item() - wide["ci_low"].item()
    assert wide_width > narrow_width


def test_an_invalid_confidence_level_is_rejected():
    frame = pd.DataFrame({"trial": [0], "m": [0], "metric": ["peak"], "value": [1.0]})
    with pytest.raises(MonteCarloError, match=r"\(0, 1\)"):
        summarise(frame, "d", "esp", confidence=1.5)


def _uniform_sigma_config(config: Config, sigma_db: float) -> Config:
    """The same three access points, all at one shadowing standard deviation."""
    return config.override(
        {
            "montecarlo.n_trials": 60,
            "scenario.aps": [
                {"name": f"AP{index}", "true_rssi_dbm": -70.0, "sigma_db": sigma_db}
                for index in (1, 2, 3)
            ],
        }
    )


def test_more_shadowing_widens_the_spread_of_the_localization_error(config):
    """The run-to-run variation the original study could not report at all."""
    quiet = run_montecarlo(_uniform_sigma_config(config, 2.0))
    noisy = run_montecarlo(_uniform_sigma_config(config, 12.0))
    quiet_spread = quiet.summary.query("metric == 'localization_error'")["std"].max()
    noisy_spread = noisy.summary.query("metric == 'localization_error'")["std"].max()
    assert noisy_spread > quiet_spread


def test_montecarlo_result_carries_its_provenance(config):
    result = run_montecarlo(config)
    assert result.seed == config.montecarlo.seed
    assert result.n_trials == config.montecarlo.n_trials
    assert result.scenario_name == config.scenario.name
