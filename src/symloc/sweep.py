"""Multi-dimensional parameter sweeps, in one tidy table.

Changing a parameter in the original workflow meant editing a constant,
re-running the script and saving a figure by hand, so each setting cost a
manual cycle and only a dozen of them were ever explored.  Here the axes --
attack strength, shadowing standard deviation, access point count, order of
:math:`M` and combining strategy -- are a Cartesian product driven from the
configuration, and the whole grid of settings produces one long-format table.

Long format is what makes the result usable: every row is one
``(offset, sigma, n_ap, combiner, M, metric)`` observation, so a heatmap, a
line plot or a pivot table is a groupby away and no result has to be copied
between a console and a plotting script.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from itertools import product

import pandas as pd

from .config import Config, ScenarioConfig
from .pipeline import analyse_scenario, evidence_collapse

_LOGGER = logging.getLogger(__name__)


class SweepError(ValueError):
    """Raised when a sweep cannot be built from the configuration."""


def _scenario_with_n_ap(base: ScenarioConfig, n_ap: int) -> ScenarioConfig:
    """Rebuild a scenario for a different access point count.

    Access point templates are cycled, so a sweep over count keeps the mix of
    standard deviations the configuration described instead of inventing new
    ones.  The same number of access points stays tampered with, clipped to
    leave at least one honest.
    """
    if n_ap < 2:
        raise SweepError(f"a scenario needs at least two access points, got {n_ap}")
    templates = base.aps
    aps = tuple(
        replace(templates[index % len(templates)], name=f"AP{index + 1}") for index in range(n_ap)
    )
    n_attacked = min(max(base.n_attacked, 1), n_ap - 1)
    attacked = tuple(range(n_ap - n_attacked, n_ap))
    return replace(base, aps=aps, attacked_indices=attacked, m_values=())


def _scenario_with_sigma(base: ScenarioConfig, sigma_db: float) -> ScenarioConfig:
    """Apply one shadowing standard deviation to every access point."""
    aps = tuple(replace(ap, sigma_db=float(sigma_db)) for ap in base.aps)
    return replace(base, aps=aps)


def build_sweep_configs(config: Config) -> list[tuple[dict[str, object], Config]]:
    """Expand the sweep axes into concrete configurations.

    Parameters
    ----------
    config : Config
        Settings; ``config.sweep`` holds the axes.

    Returns
    -------
    list of tuple
        Pairs of ``(coordinates, config)``, where ``coordinates`` names the
        point of the sweep grid that configuration represents.

    Examples
    --------
    >>> from symloc.config import Config
    >>> cfg = Config().override({"sweep.attack_offsets_db": (-10.0, -20.0)})
    >>> points = build_sweep_configs(cfg)
    >>> len(points), sorted(points[0][0])
    (2, ['attack_offset_db', 'combiner', 'n_ap', 'sigma_db'])
    """
    settings = config.sweep
    offsets = settings.attack_offsets_db
    sigmas = settings.sigmas_db or (None,)
    counts = settings.n_ap_values or (None,)
    combiners = settings.combiners or (config.scenario.combiner,)

    out: list[tuple[dict[str, object], Config]] = []
    for offset, sigma, count, combiner in product(offsets, sigmas, counts, combiners):
        scenario = config.scenario
        if count is not None:
            scenario = _scenario_with_n_ap(scenario, int(count))
        if sigma is not None:
            scenario = _scenario_with_sigma(scenario, float(sigma))
        scenario = replace(
            scenario,
            attack_offset_db=float(offset),
            combiner=str(combiner),
            name=_point_name(config.scenario.name, offset, sigma, count, combiner),
        )
        if settings.m_values:
            valid = tuple(m for m in settings.m_values if 0 <= m <= len(scenario.aps) - 1)
            scenario = replace(scenario, m_values=valid)

        coordinates: dict[str, object] = {
            "attack_offset_db": float(offset),
            "sigma_db": float(sigma) if sigma is not None else None,
            "n_ap": int(count) if count is not None else len(scenario.aps),
            "combiner": str(combiner),
        }
        out.append((coordinates, replace(config, scenario=scenario)))
    return out


def _point_name(
    base: str,
    offset: float,
    sigma: float | None,
    count: int | None,
    combiner: str,
) -> str:
    """Scenario name encoding the sweep coordinates, for traceable filenames."""
    parts = [base, f"off{offset:+.0f}dB", combiner]
    if sigma is not None:
        parts.append(f"sigma{sigma:g}")
    if count is not None:
        parts.append(f"n{count}")
    return "__".join(parts)


def run_sweep(config: Config) -> pd.DataFrame:
    """Evaluate every point of the sweep grid and concatenate the results.

    Parameters
    ----------
    config : Config
        Settings; ``config.sweep`` holds the axes.

    Returns
    -------
    pandas.DataFrame
        Long format, columns ``attack_offset_db``, ``sigma_db``, ``n_ap``,
        ``combiner``, ``scenario``, ``m``, ``metric`` and ``value``.

    Examples
    --------
    >>> from symloc.config import Config
    >>> cfg = Config().override({"sweep.attack_offsets_db": (-10.0, -20.0)})
    >>> frame = run_sweep(cfg)
    >>> set(["attack_offset_db", "m", "metric", "value"]).issubset(frame.columns)
    True
    >>> sorted(frame["attack_offset_db"].unique().tolist())
    [-20.0, -10.0]
    """
    points = build_sweep_configs(config)
    _LOGGER.info("sweeping %d parameter combination(s)", len(points))

    frames: list[pd.DataFrame] = []
    for coordinates, point_config in points:
        result = analyse_scenario(point_config)
        frame = result.metrics.copy()

        collapse = evidence_collapse(result)
        extra = pd.DataFrame(
            [
                {
                    "scenario": result.scenario.name,
                    "combiner": result.combiner,
                    "m": int(key.rsplit("m", 1)[1]),
                    "metric": key.rsplit("_m", 1)[0],
                    "value": float(value),
                }
                for key, value in collapse.items()
            ]
        )
        frame = pd.concat([frame, extra], ignore_index=True)

        for name, value in coordinates.items():
            frame[name] = value
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    ordered = [
        "attack_offset_db",
        "sigma_db",
        "n_ap",
        "combiner",
        "scenario",
        "m",
        "metric",
        "value",
    ]
    return combined[ordered]


def discriminability_pivot(
    frame: pd.DataFrame,
    metric: str = "js_divergence",
) -> pd.DataFrame:
    """Pivot a sweep result into an ``(attack strength) x (M)`` table.

    This is the view the original analysis never had: one number per cell
    saying how far the attacked posterior has moved from the honest one, across
    the whole attack-strength range at once.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`run_sweep`.
    metric : str, optional
        Metric to tabulate.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by attack offset, columns by :math:`M`.

    Raises
    ------
    SweepError
        If the metric is not present in the table.

    Examples
    --------
    >>> from symloc.config import Config
    >>> cfg = Config().override({"sweep.attack_offsets_db": (-10.0, -20.0)})
    >>> table = discriminability_pivot(run_sweep(cfg), "peak")
    >>> table.shape
    (2, 3)
    """
    available = sorted(frame["metric"].unique())
    if metric not in available:
        raise SweepError(
            f"metric {metric!r} is not in this sweep; available metrics are {available}"
        )
    subset = frame[(frame["metric"] == metric) & (frame["m"] >= 0)]
    return subset.pivot_table(
        index="attack_offset_db",
        columns="m",
        values="value",
        aggfunc="mean",
    )
