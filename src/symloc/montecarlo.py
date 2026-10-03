"""Repeated trials with resampled observation noise, and confidence intervals.

The original study evaluated one deterministic observation vector per setting.
That makes every number it reports a sample of size one: when two orders of
:math:`M` differ by 8% in peak height, nothing in the analysis says whether a
second draw of the same experiment would have reversed the ordering.

Here each trial draws a fresh observation vector from the log-normal shadowing
model at the true position, applies the attack to the tampered access points,
and runs it through the same combiners and the same metrics.  Reporting the
mean together with a percentile interval over trials turns those numbers into
estimates with stated uncertainty.

The whole batch is scored in one vectorised pass: the ``(n_trials, n_ap,
n_points)`` block of log likelihoods is folded to ``(n_ap, n_trials *
n_points)`` so the elementary-symmetric recurrence of :mod:`symloc.combiners`
runs once for every trial and candidate at the same time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ap import Scenario
from .combiners import combine_many
from .config import Config
from .likelihood import log_likelihood_for_observations
from .metrics import argmax_offset, curve_metrics
from .normalize import Curve

_LOGGER = logging.getLogger(__name__)


class MonteCarloError(ValueError):
    """Raised when a Monte-Carlo run is misconfigured."""


@dataclass(frozen=True, slots=True, eq=False)
class MonteCarloResult:
    """Metric distributions over repeated trials.

    Attributes
    ----------
    scenario_name : str
        Scenario the trials were drawn for.
    n_trials : int
        Number of independent observation vectors.
    seed : int
        Seed that produced them; re-running with it reproduces the table.
    summary : pandas.DataFrame
        Columns ``scenario``, ``combiner``, ``m``, ``metric``, ``mean``,
        ``std``, ``ci_low``, ``ci_high``, ``n_trials``.
    samples : pandas.DataFrame
        Per-trial values, columns ``trial``, ``m``, ``metric``, ``value``.
    """

    scenario_name: str
    n_trials: int
    seed: int
    summary: pd.DataFrame
    samples: pd.DataFrame


def resolve_true_index(scenario: Scenario) -> int:
    """Index of the true position, falling back to the honest signal level.

    On a degenerate radio map there is no geometry, so the best available
    stand-in for "the true position" is the candidate whose shared mean matches
    what the honest access points report.  On a path-loss map the scenario is
    expected to carry a real ``true_index``, and the fallback picks the
    reference point whose predicted means best match the honest readings.

    Parameters
    ----------
    scenario : Scenario
        The scenario.

    Returns
    -------
    int
        Index into the grid.

    Examples
    --------
    >>> from symloc.config import Config
    >>> scenario = Config().build_scenario()
    >>> float(scenario.grid.values[resolve_true_index(scenario)])
    -70.0
    """
    if scenario.true_index is not None:
        return int(scenario.true_index)
    honest = np.array([ap.true_rssi_dbm for ap in scenario.aps], dtype=np.float64)
    if scenario.radio_map.is_degenerate:
        return scenario.grid.nearest_index(float(np.mean(honest)))
    residual = scenario.radio_map.means_dbm - honest[:, None]
    return int(np.argmin(np.sum(residual**2, axis=0)))


def sample_observations(
    scenario: Scenario,
    n_trials: int,
    rng: np.random.Generator,
    true_index: int | None = None,
) -> np.ndarray:
    """Draw observation vectors for a scenario, attacks included.

    Honest access points are drawn about the mean the radio map predicts at the
    true position; tampered ones have their attack offset added afterwards, so
    the attacker shifts a real reading rather than replacing the noise model.

    Parameters
    ----------
    scenario : Scenario
        The scenario to sample.
    n_trials : int
        Number of vectors to draw.
    rng : numpy.random.Generator
        Seeded generator.
    true_index : int, optional
        Index of the true position; defaults to :func:`resolve_true_index`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_trials, n_ap)`` observations in dBm.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.config import Config
    >>> scenario = Config().build_scenario()
    >>> sample_observations(scenario, 10, np.random.default_rng(0)).shape
    (10, 3)
    """
    if n_trials < 1:
        raise MonteCarloError(f"n_trials must be at least 1, got {n_trials}")
    index = resolve_true_index(scenario) if true_index is None else int(true_index)
    means = scenario.radio_map.means_dbm[:, index]
    sigma = scenario.sigma_db
    draws = rng.normal(loc=means, scale=sigma, size=(int(n_trials), scenario.n_ap))
    offsets = np.array(
        [ap.attack_offset_db if ap.attacked else 0.0 for ap in scenario.aps],
        dtype=np.float64,
    )
    return draws + offsets


def batched_combine(
    log_p: np.ndarray,
    m_values: tuple[int, ...],
    strategy: str = "esp",
) -> dict[int, np.ndarray]:
    """Combine a whole batch of trials in one pass.

    Parameters
    ----------
    log_p : numpy.ndarray
        Shape ``(n_trials, n_ap, n_points)`` log likelihoods.
    m_values : tuple of int
        Orders to evaluate.
    strategy : str, optional
        Combiner name.

    Returns
    -------
    dict of int to numpy.ndarray
        Log scores of shape ``(n_trials, n_points)``, keyed by order.

    Examples
    --------
    >>> import numpy as np
    >>> block = np.zeros((4, 3, 7))
    >>> batched_combine(block, (0, 1))[0].shape
    (4, 7)
    """
    array = np.asarray(log_p, dtype=np.float64)
    if array.ndim != 3:
        raise MonteCarloError(
            f"log_p must be 3-D (n_trials, n_ap, n_points), got shape {array.shape}"
        )
    n_trials, n_ap, n_points = array.shape
    # The recurrence is element-wise along its second axis, so trials and
    # candidates can be flattened together and unfolded afterwards.
    folded = np.moveaxis(array, 1, 0).reshape(n_ap, n_trials * n_points)
    scores = combine_many(folded, m_values, strategy=strategy)
    return {order: value.reshape(n_trials, n_points) for order, value in scores.items()}


def run_montecarlo(
    config: Config,
    scenario: Scenario | None = None,
    attack_offset_db: float | None = None,
) -> MonteCarloResult:
    """Run repeated trials and summarise every metric with a confidence interval.

    Parameters
    ----------
    config : Config
        Settings; ``config.montecarlo`` supplies the trial count, seed and
        confidence level.
    scenario : Scenario, optional
        Scenario to sample; built from ``config`` when omitted.
    attack_offset_db : float, optional
        Overrides the configured attack strength.

    Returns
    -------
    MonteCarloResult
        Per-trial samples and their summary.

    Examples
    --------
    >>> from symloc.config import Config
    >>> cfg = Config().override({"montecarlo.n_trials": 20})
    >>> result = run_montecarlo(cfg)
    >>> result.summary.columns.tolist()[:5]
    ['scenario', 'combiner', 'm', 'metric', 'mean']
    >>> peak = result.summary.query("metric == 'peak' and m == 0").iloc[0]
    >>> bool(peak["ci_low"] <= peak["mean"] <= peak["ci_high"])
    True
    """
    if scenario is None:
        scenario = config.build_scenario(attack_offset_db=attack_offset_db)

    settings = config.montecarlo
    rng = np.random.default_rng(settings.seed)
    true_index = resolve_true_index(scenario)

    orders = config.scenario.m_values or tuple(range(scenario.n_ap))
    orders = tuple(order for order in orders if 0 <= order <= scenario.n_ap - 1)

    observations = sample_observations(scenario, settings.n_trials, rng, true_index)
    log_p = log_likelihood_for_observations(observations, scenario.radio_map, scenario.sigma_db)
    scores = batched_combine(log_p, orders, strategy=config.scenario.combiner)

    # The same draws with the tampering undone: pairing each trial with its own
    # honest counterpart removes the sampling noise from the comparison.
    attack_offsets = np.array(
        [ap.attack_offset_db if ap.attacked else 0.0 for ap in scenario.aps],
        dtype=np.float64,
    )
    honest_observations = observations - attack_offsets
    honest_log_p = log_likelihood_for_observations(
        honest_observations, scenario.radio_map, scenario.sigma_db
    )
    honest_scores = batched_combine(honest_log_p, orders, strategy=config.scenario.combiner)

    mode = config.normalize.mode
    rows: list[dict[str, object]] = []
    for trial in range(settings.n_trials):
        for order in orders:
            curve = Curve.from_log_score(
                label=f"ESP (M={order})",
                grid=scenario.grid,
                log_score=scores[order][trial],
                mode=mode,
            )
            reference = Curve.from_log_score(
                label=f"honest (M={order})",
                grid=scenario.grid,
                log_score=honest_scores[order][trial],
                mode=mode,
            )
            for metric, value in curve_metrics(curve, reference=reference).items():
                rows.append({"trial": trial, "m": order, "metric": metric, "value": value})
            rows.append(
                {
                    "trial": trial,
                    "m": order,
                    "metric": "localization_error",
                    "value": abs(argmax_offset(curve.values, scenario.grid, true_index)),
                }
            )

    samples = pd.DataFrame(rows, columns=["trial", "m", "metric", "value"])
    summary = summarise(
        samples,
        scenario_name=scenario.name,
        combiner=config.scenario.combiner,
        confidence=settings.confidence,
    )

    _LOGGER.info(
        "Monte-Carlo: %d trials for scenario %r at seed %d",
        settings.n_trials,
        scenario.name,
        settings.seed,
    )
    return MonteCarloResult(
        scenario_name=scenario.name,
        n_trials=settings.n_trials,
        seed=settings.seed,
        summary=summary,
        samples=samples,
    )


def summarise(
    samples: pd.DataFrame,
    scenario_name: str,
    combiner: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Mean, standard deviation and percentile interval of every metric.

    The interval is the empirical percentile interval over trials rather than a
    normal approximation, because several of these metrics are bounded or
    heavily skewed and a symmetric interval would run outside their range.

    Parameters
    ----------
    samples : pandas.DataFrame
        Per-trial values with columns ``trial``, ``m``, ``metric``, ``value``.
    scenario_name : str
        Name carried into the output.
    combiner : str
        Combiner name carried into the output.
    confidence : float, optional
        Confidence level in ``(0, 1)``.

    Returns
    -------
    pandas.DataFrame
        One row per ``(m, metric)``.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame(
    ...     {"trial": [0, 1], "m": [0, 0], "metric": ["peak", "peak"], "value": [1.0, 3.0]}
    ... )
    >>> summarise(frame, "demo", "esp")["mean"].tolist()
    [2.0]
    """
    if not 0 < confidence < 1:
        raise MonteCarloError(f"confidence must lie in (0, 1), got {confidence!r}")
    tail = 0.5 * (1.0 - confidence)

    # Infinities (an unimodal curve has no secondary peak) would poison the
    # mean; they are recorded as missing and excluded from the statistics.
    finite = samples.replace([np.inf, -np.inf], np.nan)
    grouped = finite.groupby(["m", "metric"], sort=True)["value"]
    summary = grouped.agg(
        mean="mean",
        std="std",
        n_trials="count",
        ci_low=lambda values: values.quantile(tail),
        ci_high=lambda values: values.quantile(1.0 - tail),
    ).reset_index()
    summary.insert(0, "combiner", combiner)
    summary.insert(0, "scenario", scenario_name)
    return summary[
        ["scenario", "combiner", "m", "metric", "mean", "std", "ci_low", "ci_high", "n_trials"]
    ]
