"""The candidate grid that every likelihood is evaluated on.

A fingerprint localizer scores a set of candidate positions and reports the
best one.  This module owns that set of candidates, in the two forms the
project needs:

* a one-dimensional sweep over a *mean RSSI* axis, which is what the original
  analysis did -- every access point was assumed to share one mean signal level
  at a given candidate, so the axis is a signal level in dBm and not a place on
  a floor;
* a two-dimensional set of reference points with real coordinates, which is
  what a fingerprint database actually looks like, and which becomes a usable
  radio map once :mod:`symloc.pathloss` gives each access point its own mean per
  point.

The distinction matters because the first form cannot support the phrase
"reference location" on an axis label: there is no geometry in it.  Keeping
both behind one :class:`Grid` type lets the same likelihood, combiner and
metric code serve the legacy reproduction and the geometric study, while the
axis label travels with the data instead of being typed into a plotting script.

The grid also carries its own ``step``.  Normalising a likelihood by its sum
turns it into a probability mass function whose peak height scales with the
grid resolution, so a reported peak is meaningless unless the step that
produced it is recorded alongside -- see :mod:`symloc.normalize`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

_LOGGER = logging.getLogger(__name__)


class GridError(ValueError):
    """Raised when a candidate grid is empty, malformed or inconsistent."""


@dataclass(frozen=True, slots=True, eq=False)
class Grid:
    """A set of candidate positions, plus the axis they should be plotted on.

    Attributes
    ----------
    values : numpy.ndarray
        Shape ``(n_points,)``.  For a dBm sweep these are the mean signal
        levels in dBm; for a set of reference points they are point indices.
    step : float
        Spacing between neighbouring candidates, in the unit of ``values``.
    axis_label : str
        Axis label matching ``values``.  Plotting code uses this verbatim, so
        the label can never drift away from the data it describes.
    unit : str
        Short unit string, e.g. ``"dBm"`` or ``"index"``.
    coordinates : numpy.ndarray or None
        Shape ``(n_points, 2)`` metre coordinates when the grid describes real
        reference points, otherwise ``None``.

    Examples
    --------
    >>> grid = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> grid.size, grid.step
    (51, 0.5)
    >>> float(grid.values[0]), float(grid.values[-1])
    (-85.0, -60.0)
    """

    values: np.ndarray
    step: float
    axis_label: str = "mean RSSI (dBm)"
    unit: str = "dBm"
    coordinates: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise GridError(f"grid values must be one-dimensional, got shape {values.shape}")
        if values.size < 2:
            raise GridError(f"a grid needs at least two candidates, got {values.size}")
        if not np.all(np.isfinite(values)):
            raise GridError("grid values must all be finite")
        object.__setattr__(self, "values", values)

        if not float(self.step) > 0:
            raise GridError(f"grid step must be positive, got {self.step!r}")
        object.__setattr__(self, "step", float(self.step))

        if self.coordinates is not None:
            coords = np.asarray(self.coordinates, dtype=np.float64)
            if coords.shape != (values.size, 2):
                raise GridError(
                    f"coordinates must have shape ({values.size}, 2), got {coords.shape}"
                )
            object.__setattr__(self, "coordinates", coords)

    @property
    def size(self) -> int:
        """Number of candidate positions."""
        return int(self.values.size)

    def __len__(self) -> int:
        return self.size

    @property
    def span(self) -> tuple[float, float]:
        """Closed interval covered by the grid, as ``(low, high)``."""
        return float(self.values.min()), float(self.values.max())

    @classmethod
    def from_dbm_sweep(cls, low_dbm: float, high_dbm: float, step_db: float) -> Grid:
        """Build the one-dimensional mean-RSSI sweep used by the legacy analysis.

        The endpoints are inclusive and the axis is returned in ascending dBm
        order, which is the natural direction for a plot.  The original script
        swept the same 51 levels in the opposite direction; because every
        statistic this package reports is either a peak, a sum or a divergence,
        the ordering does not affect any result.

        Parameters
        ----------
        low_dbm, high_dbm : float
            Inclusive endpoints of the sweep in dBm, with ``low_dbm`` the
            weaker (more negative) level.
        step_db : float
            Positive spacing in dB.

        Returns
        -------
        Grid
            An ascending sweep labelled as a mean RSSI axis.

        Raises
        ------
        GridError
            If the endpoints are out of order, the step is not positive, or the
            step does not divide the span to within a small tolerance.

        Examples
        --------
        >>> Grid.from_dbm_sweep(-85.0, -60.0, 0.5).size
        51
        """
        if not step_db > 0:
            raise GridError(f"step_db must be positive, got {step_db!r}")
        if not high_dbm > low_dbm:
            raise GridError(
                f"high_dbm ({high_dbm}) must be greater than low_dbm ({low_dbm}); "
                "give the endpoints as (weaker, stronger) in dBm"
            )
        span = high_dbm - low_dbm
        n_steps = span / step_db
        rounded = round(n_steps)
        if abs(n_steps - rounded) > 1e-9:
            raise GridError(
                f"step_db={step_db} does not divide the span {span} evenly "
                f"({n_steps:.6f} steps); adjust the endpoints or the step"
            )
        values = low_dbm + step_db * np.arange(rounded + 1, dtype=np.float64)
        _LOGGER.debug("built dBm sweep: %d points from %g to %g", values.size, low_dbm, high_dbm)
        return cls(values=values, step=float(step_db), axis_label="mean RSSI (dBm)", unit="dBm")

    @classmethod
    def from_points(cls, coordinates: np.ndarray) -> Grid:
        """Build a grid of real reference points from metre coordinates.

        Parameters
        ----------
        coordinates : array_like
            Shape ``(n_points, 2)``, in metres.

        Returns
        -------
        Grid
            ``values`` holds the point indices, ``coordinates`` the positions,
            and ``axis_label`` says so -- this is the only grid on which the
            phrase "reference location" is honest.

        Examples
        --------
        >>> import numpy as np
        >>> g = Grid.from_points(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
        >>> g.size, g.unit
        (3, 'index')
        """
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise GridError(f"coordinates must have shape (n_points, 2), got {coords.shape}")
        values = np.arange(coords.shape[0], dtype=np.float64)
        return cls(
            values=values,
            step=1.0,
            axis_label="reference point index",
            unit="index",
            coordinates=coords,
        )

    @classmethod
    def rectangular(cls, width_m: float, depth_m: float, spacing_m: float) -> Grid:
        """Build reference points on a regular rectangular floor plan.

        Parameters
        ----------
        width_m, depth_m : float
            Extent of the floor in metres along x and y.
        spacing_m : float
            Distance between neighbouring reference points, in metres.

        Returns
        -------
        Grid
            A grid whose ``coordinates`` tile the rectangle in row-major order.

        Examples
        --------
        >>> Grid.rectangular(4.0, 2.0, 2.0).size
        6
        """
        for name, value in (("width_m", width_m), ("depth_m", depth_m), ("spacing_m", spacing_m)):
            if not value > 0:
                raise GridError(f"{name} must be positive, got {value!r}")
        xs = np.arange(0.0, width_m + 1e-9, spacing_m, dtype=np.float64)
        ys = np.arange(0.0, depth_m + 1e-9, spacing_m, dtype=np.float64)
        mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="xy")
        coords = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        return cls.from_points(coords)

    def nearest_index(self, value: float) -> int:
        """Index of the candidate closest to ``value`` on this axis.

        Examples
        --------
        >>> Grid.from_dbm_sweep(-85.0, -60.0, 0.5).nearest_index(-70.2)
        30
        """
        return int(np.argmin(np.abs(self.values - float(value))))

    def contains(self, value: float, tolerance: float = 0.0) -> bool:
        """Whether ``value`` lies within the closed span of the grid.

        Examples
        --------
        >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
        >>> g.contains(-70.0), g.contains(-99.0)
        (True, False)
        """
        low, high = self.span
        return bool(low - tolerance <= float(value) <= high + tolerance)
