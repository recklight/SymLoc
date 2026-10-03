"""Figure generation, with the axis contract that the original plots lacked.

Every figure here derives its axis labels from the :class:`~symloc.grid.Grid` the
data was computed on, takes its colours from one palette, and states the grid
resolution in its caption, so a saved figure can be read without its
configuration file beside it.  Nothing fixes a tick, a limit or a legend entry
to a particular number of curves.
"""

from __future__ import annotations

from .distribution import plot_distributions, plot_single_ap_likelihoods
from .heatmap import plot_confusion_matrix, plot_discriminability_heatmap, plot_roc_curve
from .peak_bars import plot_montecarlo_intervals, plot_peak_bars, plot_peak_ratios
from .style import PALETTE, curve_color, curve_marker, rc_params, style_context

__all__ = [
    "PALETTE",
    "curve_color",
    "curve_marker",
    "plot_confusion_matrix",
    "plot_discriminability_heatmap",
    "plot_distributions",
    "plot_montecarlo_intervals",
    "plot_peak_bars",
    "plot_peak_ratios",
    "plot_roc_curve",
    "plot_single_ap_likelihoods",
    "rc_params",
    "style_context",
]
