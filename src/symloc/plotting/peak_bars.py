"""Bar charts of peak heights and of the adjacent-order ratios.

These reproduce the archived bar figures, with two differences.  The values are
printed on the bars, so the figure carries its own numbers instead of leaving
them in a workspace variable that was never written down; and the caption
states the grid step, because a peak of a mass function is only meaningful
together with the resolution that produced it.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..normalize import grid_resolution_note
from ..pipeline import ScenarioResult
from .style import BASELINE_COLOR, curve_color, style_context

_LOGGER = logging.getLogger(__name__)


def plot_peak_bars(result: ScenarioResult, show_baseline: bool = True) -> Figure:
    """Bar chart of each curve's peak height.

    Parameters
    ----------
    result : ScenarioResult
        A finished scenario.
    show_baseline : bool, optional
        Whether to include the unattacked control as an extra bar.

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
    >>> figure = plot_peak_bars(analyse_scenario(Config()))
    >>> len(figure.axes)
    1
    >>> import matplotlib.pyplot as plt; plt.close(figure)
    """
    orders = result.m_values
    labels = [result.curves[order].label for order in orders]
    values = [result.curves[order].peak for order in orders]
    colors = [curve_color(index) for index in range(len(orders))]

    if show_baseline:
        labels.append(result.baseline.label)
        values.append(result.baseline.peak)
        colors.append(BASELINE_COLOR)

    with style_context():
        figure, axes = plt.subplots()
        positions = np.arange(len(labels))
        bars = axes.bar(positions, values, color=colors, width=0.62)
        for bar, value in zip(bars, values, strict=True):
            axes.annotate(
                f"{value:.5f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, rotation=20, ha="right")
        axes.set_ylabel("peak of the normalised curve")
        axes.set_title(f"{result.scenario.name}: peak heights")
        axes.set_ylim(0, max(values) * 1.18)

        mode = result.curves[orders[0]].mode
        figure.text(
            0.01,
            -0.06,
            grid_resolution_note(result.scenario.grid, mode),
            fontsize=7,
            color="#555555",
            ha="left",
        )
    return figure


def plot_peak_ratios(result: ScenarioResult) -> Figure:
    """Bar chart of the adjacent-order peak ratios and differences.

    These are the original discriminability measures, kept so a result can be
    compared against the archived figures directly.  They are plotted side by
    side because the ratio and the difference can disagree about which pair of
    orders separates best, and reporting only one hides that.

    Parameters
    ----------
    result : ScenarioResult
        A finished scenario.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with two panels.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from symloc.config import Config
    >>> from symloc.pipeline import analyse_scenario
    >>> figure = plot_peak_ratios(analyse_scenario(Config()))
    >>> len(figure.axes)
    2
    >>> import matplotlib.pyplot as plt; plt.close(figure)
    """
    ratios = result.adjacent_peak_ratios()
    differences = result.adjacent_peak_differences()

    with style_context():
        figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 4.0))

        _bar_panel(
            left,
            [key.replace("rho_", "M=").replace("_", " vs M=") for key in ratios],
            list(ratios.values()),
            ylabel=r"peak ratio $\rho$",
            fmt="{:.5f}",
        )
        left.axhline(1.0, color="#888888", linewidth=0.8, linestyle=":")
        left.set_title("ratio of adjacent peaks")

        _bar_panel(
            right,
            [key.replace("delta_", "M=").replace("_", " vs M=") for key in differences],
            list(differences.values()),
            ylabel=r"peak difference $\Delta$",
            fmt="{:.5f}",
        )
        right.set_title("difference of adjacent peaks")

        figure.suptitle(f"{result.scenario.name}: discriminability between orders")
    return figure


def _bar_panel(axes, labels: list[str], values: list[float], ylabel: str, fmt: str) -> None:
    """Draw one labelled bar panel."""
    positions = np.arange(len(labels))
    bars = axes.bar(positions, values, color=[curve_color(i) for i in range(len(labels))])
    for bar, value in zip(bars, values, strict=True):
        axes.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=15, ha="right")
    axes.set_ylabel(ylabel)
    if values:
        axes.set_ylim(0, max(values) * 1.2)


def plot_montecarlo_intervals(
    summary,
    metric: str = "peak",
    title: str | None = None,
) -> Figure:
    """Metric mean and confidence interval against :math:`M`.

    The measure the original study could not report at all: whether the
    difference between two orders survives resampling the observation noise.

    Parameters
    ----------
    summary : pandas.DataFrame
        The ``summary`` table of a :class:`~symloc.montecarlo.MonteCarloResult`.
    metric : str, optional
        Metric to plot.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from symloc.config import Config
    >>> from symloc.montecarlo import run_montecarlo
    >>> result = run_montecarlo(Config().override({"montecarlo.n_trials": 20}))
    >>> figure = plot_montecarlo_intervals(result.summary)
    >>> len(figure.axes)
    1
    >>> import matplotlib.pyplot as plt; plt.close(figure)
    """
    subset = summary[summary["metric"] == metric].sort_values("m")
    if subset.empty:
        available = sorted(summary["metric"].unique())
        raise ValueError(f"metric {metric!r} is not in this summary; available: {available}")

    orders = subset["m"].to_numpy()
    means = subset["mean"].to_numpy(dtype=float)
    low = subset["ci_low"].to_numpy(dtype=float)
    high = subset["ci_high"].to_numpy(dtype=float)
    errors = np.vstack([means - low, high - means])

    with style_context():
        figure, axes = plt.subplots()
        axes.errorbar(
            orders,
            means,
            yerr=np.abs(errors),
            marker="o",
            capsize=4,
            color=curve_color(0),
            linestyle="-",
        )
        axes.set_xticks(orders)
        axes.set_xlabel("M (number of tampered APs tolerated)")
        axes.set_ylabel(metric.replace("_", " "))
        n_trials = int(subset["n_trials"].max())
        axes.set_title(
            title if title is not None else f"{metric} over {n_trials} trials, with 95% interval"
        )
    return figure
