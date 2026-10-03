"""One scenario, end to end: likelihoods, combiners, distributions, metrics.

Everything the original analysis did by copying a block of code and editing the
constants at the top happens here once, driven by a :class:`~symloc.config.Config`.
The honest control, the attacked cases and every order of :math:`M` all travel
the same path, so a difference between two curves can only come from the
parameters and never from one of them having been computed by different code.

The result of a run is a tidy long-format table -- one row per
``(scenario, combiner, M, metric)`` -- which is what makes the sweep in
:mod:`symloc.sweep` a matter of concatenation rather than of bookkeeping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd

from .ap import Scenario
from .combiners import combine_many
from .config import Config
from .likelihood import scenario_log_likelihood
from .metrics import argmax_offset, curve_metrics, peak_difference, peak_ratio
from .normalize import Curve, check_truncation, grid_resolution_note

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioResult:
    """Everything one scenario produced.

    Attributes
    ----------
    scenario : Scenario
        The scenario that was evaluated.
    curves : dict of int to Curve
        Normalised curves keyed by :math:`M`.
    baseline : Curve
        The unattacked control, computed from the same scenario with every
        attack removed and combined at :math:`M = 0`.
    baseline_curves : dict of int to Curve
        The honest control at every evaluated order, so that an attacked
        curve is always compared against the same combiner rather than
        against a differently-shaped one.
    combiner : str
        Strategy used for ``curves``.
    truncation : dict of str to float
        Fraction of each access point's likelihood mass lying off the grid.
    metrics : pandas.DataFrame
        Tidy table with columns ``scenario``, ``combiner``, ``m``, ``metric``
        and ``value``.
    notes : dict
        Provenance carried into the run manifest, including the grid
        resolution note that makes a reported peak interpretable.
    """

    scenario: Scenario
    curves: dict[int, Curve]
    baseline: Curve
    baseline_curves: dict[int, Curve]
    combiner: str
    truncation: dict[str, float]
    metrics: pd.DataFrame
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def m_values(self) -> tuple[int, ...]:
        """Orders present in ``curves``, ascending."""
        return tuple(sorted(self.curves))

    def peaks(self) -> dict[int, float]:
        """Peak height of every curve, keyed by :math:`M`.

        Examples
        --------
        >>> from symloc.config import Config
        >>> result = analyse_scenario(Config())
        >>> sorted(result.peaks()) == [0, 1, 2]
        True
        """
        return {m: curve.peak for m, curve in self.curves.items()}

    def adjacent_peak_ratios(self) -> dict[str, float]:
        """Peak ratios between neighbouring orders, the original ``rho``.

        Returned keyed by ``"rho_0_1"``, ``"rho_1_2"`` and so on, so a result
        can be compared directly against the archived figures.

        Examples
        --------
        >>> from symloc.config import Config
        >>> sorted(analyse_scenario(Config()).adjacent_peak_ratios())
        ['rho_0_1', 'rho_1_2']
        """
        orders = self.m_values
        return {
            f"rho_{low}_{high}": peak_ratio(self.curves[low].values, self.curves[high].values)
            for low, high in pairwise(orders)
        }

    def adjacent_peak_differences(self) -> dict[str, float]:
        """Peak differences between neighbouring orders, the original ``Delta``."""
        orders = self.m_values
        return {
            f"delta_{low}_{high}": peak_difference(
                self.curves[low].values, self.curves[high].values
            )
            for low, high in pairwise(orders)
        }


def analyse_scenario(
    config: Config,
    scenario: Scenario | None = None,
    attack_offset_db: float | None = None,
) -> ScenarioResult:
    """Run one scenario through the whole pipeline.

    Parameters
    ----------
    config : Config
        Settings; supplies the grid, the combiner, the orders of :math:`M` and
        the normalisation mode.
    scenario : Scenario, optional
        A ready-made scenario.  When omitted it is built from ``config``.
    attack_offset_db : float, optional
        Overrides the configured attack strength, which is what the sweep
        driver varies.

    Returns
    -------
    ScenarioResult
        Curves, control, diagnostics and the tidy metric table.

    Examples
    --------
    >>> from symloc.config import Config
    >>> result = analyse_scenario(Config())
    >>> result.metrics.columns.tolist()
    ['scenario', 'combiner', 'm', 'metric', 'value']
    >>> bool((result.metrics["value"].notna()).any())
    True
    """
    if scenario is None:
        scenario = config.build_scenario(attack_offset_db=attack_offset_db)

    truncation = check_truncation(scenario, warn_above=config.normalize.warn_truncation_above)

    orders = config.scenario.m_values or tuple(range(scenario.n_ap))
    orders = tuple(order for order in orders if 0 <= order <= scenario.n_ap - 1)
    if not orders:
        raise ValueError(
            f"scenario {scenario.name!r} has {scenario.n_ap} access points, which "
            "leaves no valid order of M to evaluate"
        )

    log_p = scenario_log_likelihood(scenario)
    log_scores = combine_many(log_p, orders, strategy=config.scenario.combiner)

    mode = config.normalize.mode
    curves = {
        order: Curve.from_log_score(
            label=_curve_label(config.scenario.combiner, order),
            grid=scenario.grid,
            log_score=score,
            mode=mode,
            metadata={"m": order, "combiner": config.scenario.combiner},
        )
        for order, score in log_scores.items()
    }

    # The control: the same access points with every attack removed, combined
    # with the plain product.  Produced by this same code path, not a special
    # case written out separately.
    honest = scenario.honest()
    honest_log_p = scenario_log_likelihood(honest)
    honest_scores = combine_many(honest_log_p, orders, strategy=config.scenario.combiner)
    baseline_curves = {
        order: Curve.from_log_score(
            label=f"no attacked AP ({_curve_label(config.scenario.combiner, order)})",
            grid=scenario.grid,
            log_score=score,
            mode=mode,
            metadata={"m": order, "combiner": config.scenario.combiner, "role": "baseline"},
        )
        for order, score in honest_scores.items()
    }
    # The legacy control is the plain product of the honest access points,
    # which is what the archived figures plotted alongside the ESP curves.
    baseline = Curve.from_log_score(
        label="Bayesian (no attacked AP)",
        grid=scenario.grid,
        log_score=honest_log_p.sum(axis=0),
        mode=mode,
        metadata={"m": 0, "combiner": "product", "role": "baseline"},
    )

    metrics = _build_metric_table(scenario, config, curves, baseline)

    notes: dict[str, object] = {
        "grid_resolution": grid_resolution_note(scenario.grid, mode),
        "normalize_mode": mode,
        "n_ap": scenario.n_ap,
        "n_attacked": scenario.n_attacked,
        "attacked": [ap.name for ap in scenario.aps if ap.attacked],
        "observed_rssi_dbm": scenario.observed_rssi_dbm.tolist(),
        "sigma_db": scenario.sigma_db.tolist(),
        "radio_map_degenerate": scenario.radio_map.is_degenerate,
        "truncated_mass": truncation,
    }
    if scenario.radio_map.is_degenerate:
        notes["radio_map_caveat"] = (
            "every access point shares one mean per candidate, so the horizontal "
            "axis is a mean signal level in dBm and not a location on a floor"
        )

    _LOGGER.info(
        "scenario %r: %d access points (%d attacked), M in %s, combiner %r",
        scenario.name,
        scenario.n_ap,
        scenario.n_attacked,
        list(orders),
        config.scenario.combiner,
    )
    return ScenarioResult(
        scenario=scenario,
        curves=curves,
        baseline=baseline,
        baseline_curves=baseline_curves,
        combiner=config.scenario.combiner,
        truncation=truncation,
        metrics=metrics,
        notes=notes,
    )


def _curve_label(combiner: str, order: int) -> str:
    """Legend label for a curve, spelled consistently everywhere it appears."""
    if combiner == "esp":
        return f"ESP (M={order})"
    if combiner == "product":
        return "product (M=0)"
    if combiner == "sum":
        return "sum (M=N-1)"
    return f"{combiner} (M={order})"


def _build_metric_table(
    scenario: Scenario,
    config: Config,
    curves: dict[int, Curve],
    baseline: Curve,
) -> pd.DataFrame:
    """Assemble the tidy long-format metric table for one scenario."""
    rows: list[dict[str, object]] = []

    def _add(order: int, metric: str, value: float) -> None:
        rows.append(
            {
                "scenario": scenario.name,
                "combiner": config.scenario.combiner,
                "m": order,
                "metric": metric,
                "value": float(value),
            }
        )

    for order in sorted(curves):
        curve = curves[order]
        for metric, value in curve_metrics(curve, reference=baseline).items():
            _add(order, metric, value)
        if scenario.true_index is not None:
            _add(
                order,
                "localization_error",
                abs(argmax_offset(curve.values, scenario.grid, scenario.true_index)),
            )

    orders = sorted(curves)
    for low, high in pairwise(orders):
        _add(low, f"peak_ratio_vs_m{high}", peak_ratio(curves[low].values, curves[high].values))
        _add(
            low,
            f"peak_difference_vs_m{high}",
            peak_difference(curves[low].values, curves[high].values),
        )

    for metric, value in curve_metrics(baseline).items():
        rows.append(
            {
                "scenario": scenario.name,
                "combiner": "baseline",
                "m": -1,
                "metric": metric,
                "value": float(value),
            }
        )

    frame = pd.DataFrame(rows, columns=["scenario", "combiner", "m", "metric", "value"])
    return frame


def curves_to_frame(result: ScenarioResult) -> pd.DataFrame:
    """Long-format table of every curve value, for plotting or archiving.

    Parameters
    ----------
    result : ScenarioResult
        A finished scenario.

    Returns
    -------
    pandas.DataFrame
        Columns ``scenario``, ``curve``, ``m``, ``x``, ``value``.

    Examples
    --------
    >>> from symloc.config import Config
    >>> frame = curves_to_frame(analyse_scenario(Config()))
    >>> frame.columns.tolist()
    ['scenario', 'curve', 'm', 'x', 'value']
    """
    frames: list[pd.DataFrame] = []
    axis = result.scenario.grid.values
    for order in result.m_values:
        curve = result.curves[order]
        frames.append(
            pd.DataFrame(
                {
                    "scenario": result.scenario.name,
                    "curve": curve.label,
                    "m": order,
                    "x": axis,
                    "value": curve.values,
                }
            )
        )
    frames.append(
        pd.DataFrame(
            {
                "scenario": result.scenario.name,
                "curve": result.baseline.label,
                "m": -1,
                "x": axis,
                "value": result.baseline.values,
            }
        )
    )
    return pd.concat(frames, ignore_index=True)


def evidence_collapse(result: ScenarioResult) -> dict[str, float]:
    """How much absolute evidence the attack removed, in nats and as a ratio.

    This is the quantity grid normalisation destroys, and the one that actually
    supports the claim that tampering breaks a naive-Bayes fingerprint match.
    Each combined score is compared against the same combiner applied to the
    honest access points.

    Parameters
    ----------
    result : ScenarioResult
        A finished scenario.

    Returns
    -------
    dict of str to float
        ``log_evidence_gap_m<M>`` in nats, plus the equivalent ratio.

    Examples
    --------
    >>> from symloc.config import Config
    >>> gaps = evidence_collapse(analyse_scenario(Config()))
    >>> gaps["log_evidence_gap_m0"] < 0
    True
    """
    out: dict[str, float] = {}
    for order in result.m_values:
        honest = result.baseline_curves.get(order, result.baseline)
        gap = float(result.curves[order].log_evidence - honest.log_evidence)
        out[f"log_evidence_gap_m{order}"] = gap
        out[f"evidence_ratio_m{order}"] = float(np.exp(gap))
    return out
