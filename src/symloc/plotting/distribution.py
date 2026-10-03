"""Overlay of the normalised curves, on an axis that matches the data.

The single most misleading feature of the archived figures is the horizontal
axis: it was labelled "Reference location" and ticked 0 to 30, while the data
underneath ran from -60 to -85 dBm.  The labels were forced without setting the
tick positions, so they landed on whatever ticks were chosen automatically and
repeated.  A reader could not recover a single true abscissa from the figure.

Here the axis label and unit travel with the :class:`~symloc.grid.Grid`, so a
curve computed on a dBm sweep is labelled in dBm and one computed over
reference points is labelled by point index.  Nothing about the axis is typed
in at plotting time, and no tick is ever overridden.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..normalize import Curve, grid_resolution_note
from ..pipeline import ScenarioResult
from .style import BASELINE_COLOR, BASELINE_STYLE, curve_color, curve_marker, style_context

_LOGGER = logging.getLogger(__name__)


def plot_distributions(
    result: ScenarioResult,
    show_baseline: bool = True,
    annotate_peaks: bool = True,
    title: str | None = None,
) -> Figure:
    """Plot every curve of a scenario, with the control drawn distinctly.

    Parameters
    ----------
    result : ScenarioResult
        A finished scenario.
    show_baseline : bool, optional
        Whether to draw the unattacked control.
    annotate_peaks : bool, optional
        Whether to mark each curve's peak with its height.
    title : str, optional
        Figure title; a description of the scenario is used when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, left open for the caller to save or close.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from symloc.config import Config
    >>> from symloc.pipeline import analyse_scenario
    >>> figure = plot_distributions(analyse_scenario(Config()))
    >>> len(figure.axes)
    1
    >>> import matplotlib.pyplot as plt; plt.close(figure)
    """
    grid = result.scenario.grid
    axis_values = grid.values

    with style_context():
        figure, axes = plt.subplots()

        for index, order in enumerate(result.m_values):
            curve = result.curves[order]
            axes.plot(
                axis_values,
                curve.values,
                color=curve_color(index),
                marker=curve_marker(index),
                markevery=max(1, grid.size // 12),
                label=curve.label,
            )
            if annotate_peaks:
                _annotate_peak(axes, curve, curve_color(index))

        if show_baseline:
            axes.plot(
                axis_values,
                result.baseline.values,
                color=BASELINE_COLOR,
                linestyle=BASELINE_STYLE,
                linewidth=1.4,
                label=result.baseline.label,
            )
            if annotate_peaks:
                _annotate_peak(axes, result.baseline, BASELINE_COLOR)

        axes.set_xlabel(grid.axis_label)
        axes.set_ylabel(_y_label(result))
        axes.set_title(title if title is not None else _default_title(result))
        axes.legend(loc="best")

        note = grid_resolution_note(grid, result.curves[result.m_values[0]].mode)
        figure.text(0.01, -0.03, note, fontsize=7, color="#555555", ha="left")
        if result.scenario.radio_map.is_degenerate:
            figure.text(
                0.01,
                -0.08,
                "degenerate radio map: the abscissa is a shared mean signal level, "
                "not a position on a floor",
                fontsize=7,
                color="#555555",
                ha="left",
            )

    return figure


def _annotate_peak(axes, curve: Curve, color: str) -> None:
    """Mark a curve's peak with a dot and its height."""
    index = curve.argmax_index
    x = curve.grid.values[index]
    y = curve.values[index]
    axes.plot([x], [y], marker="o", markersize=5, color=color, zorder=5)
    axes.annotate(
        f"{y:.4f}",
        xy=(x, y),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color=color,
    )


def _y_label(result: ScenarioResult) -> str:
    """Axis label naming what the values actually are."""
    mode = result.curves[result.m_values[0]].mode
    unit = result.scenario.grid.unit
    if mode == "sum":
        return "probability mass per grid point"
    if mode == "density":
        return f"probability density (1/{unit})"
    return "unnormalised likelihood"


def _default_title(result: ScenarioResult) -> str:
    """Title stating the parameters, so a saved figure is self-describing."""
    scenario = result.scenario
    attacked = [ap.name for ap in scenario.aps if ap.attacked]
    if attacked:
        offsets = {ap.attack_offset_db for ap in scenario.aps if ap.attacked}
        offset = f"{offsets.pop():+.0f} dB" if len(offsets) == 1 else "mixed"
        attack = f"{len(attacked)} attacked ({', '.join(attacked)}, {offset})"
    else:
        attack = "no attacked AP"
    return f"{scenario.name}: {scenario.n_ap} APs, {attack}"


def plot_single_ap_likelihoods(result: ScenarioResult) -> Figure:
    """Plot each access point's own likelihood, before any combining.

    This is the diagnostic that makes an attack visible at a glance: a tampered
    access point's curve peaks somewhere the others do not, or lies almost flat
    because its reading falls outside the grid entirely.

    Parameters
    ----------
    result : ScenarioResult
        A finished scenario.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from symloc.config import Config
    >>> from symloc.pipeline import analyse_scenario
    >>> figure = plot_single_ap_likelihoods(analyse_scenario(Config()))
    >>> len(figure.axes)
    1
    >>> import matplotlib.pyplot as plt; plt.close(figure)
    """
    from ..likelihood import scenario_log_likelihood

    scenario = result.scenario
    grid = scenario.grid
    log_p = scenario_log_likelihood(scenario)

    with style_context():
        figure, axes = plt.subplots()
        for index, ap in enumerate(scenario.aps):
            values = np.exp(log_p[index])
            label = f"{ap.name} ({ap.observed_rssi_dbm:.0f} dBm)"
            if ap.attacked:
                label += " [attacked]"
            axes.plot(
                grid.values,
                values,
                color=curve_color(index),
                marker=curve_marker(index),
                markevery=max(1, grid.size // 12),
                linestyle="--" if ap.attacked else "-",
                label=label,
            )
        axes.set_xlabel(grid.axis_label)
        axes.set_ylabel("likelihood density (1/dB)")
        axes.set_title(f"{scenario.name}: per-AP likelihoods")
        axes.legend(loc="best")
    return figure
