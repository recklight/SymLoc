"""One place that decides what every figure looks like.

The plotting routine behind the archived figures was a generated template: it
hard-coded the colour, marker and legend entry of each of exactly four lines,
fixed the y-axis ticks to a range that only suited one figure, and forced
x-axis tick *labels* without setting the tick *positions*, so the archived
axis showed numbers that had nothing to do with the data underneath.  Adding a
fifth curve broke it outright.

Everything here is derived instead: colours come from a palette long enough for
any curve count and are distinguishable for the common forms of colour vision
deficiency, markers cycle independently so the curves stay separable in
greyscale, and no axis limit or tick is fixed anywhere.  Axis labels are taken
from the grid the data was computed on, never typed in beside the plot.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib as mpl

_LOGGER = logging.getLogger(__name__)

#: Qualitative palette chosen for colour-vision-deficiency safety, in a fixed
#: order so the same curve keeps the same colour across every figure of a run.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#6E6E6E",  # grey
)

#: Marker cycle, so curves remain separable when a figure is printed in mono.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

#: Colour reserved for the unattacked control, which is a reference and not one
#: of the compared curves.
BASELINE_COLOR = "#000000"

#: Line style reserved for the control, for the same reason.
BASELINE_STYLE = (0, (4, 2))


def curve_color(index: int) -> str:
    """Colour for the ``index``-th curve, cycling through :data:`PALETTE`.

    Examples
    --------
    >>> curve_color(0), curve_color(len(PALETTE))
    ('#0072B2', '#0072B2')
    """
    return PALETTE[int(index) % len(PALETTE)]


def curve_marker(index: int) -> str:
    """Marker for the ``index``-th curve, cycling through :data:`MARKERS`.

    Examples
    --------
    >>> curve_marker(1)
    's'
    """
    return MARKERS[int(index) % len(MARKERS)]


def rc_params(base_font_size: float = 10.0) -> dict[str, Any]:
    """Matplotlib settings applied to every figure in this package.

    Parameters
    ----------
    base_font_size : float, optional
        Body font size in points; every other size is derived from it.

    Returns
    -------
    dict
        Mapping suitable for :func:`matplotlib.rc_context`.

    Examples
    --------
    >>> rc_params()["axes.grid"]
    True
    """
    size = float(base_font_size)
    return {
        "figure.figsize": (7.0, 4.5),
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": size,
        "axes.titlesize": size + 1.0,
        "axes.labelsize": size,
        "legend.fontsize": size - 1.0,
        "xtick.labelsize": size - 1.0,
        "ytick.labelsize": size - 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.autolayout": False,
    }


def style_context(base_font_size: float = 10.0) -> Any:
    """Context manager applying :func:`rc_params` to the figures made inside it.

    Used rather than a global ``rcParams`` update so importing this package
    never changes how a caller's own plots look.

    Examples
    --------
    >>> with style_context():
    ...     pass
    """
    return mpl.rc_context(rc_params(base_font_size))
