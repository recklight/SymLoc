"""Log-distance path loss with log-normal shadowing.

This module supplies the physical justification for modelling an RSSI reading
in dBm with a Gaussian, and it supplies the non-degenerate radio map the
original analysis lacked.

The log-distance model writes the path loss at distance :math:`d` as

.. math::

   PL(d) = PL(d_0) + 10\\,n \\log_{10}\\!\\left(\\frac{d}{d_0}\\right) + X_\\sigma,

where :math:`n` is the path loss exponent (about 2 in free space, 3 to 5 inside
a building) and :math:`X_\\sigma` is a zero-mean Gaussian in dB -- the
log-normal shadowing term.  Expressed as a received level, the mean reading of
an access point at distance :math:`d` is

.. math::

   \\mu(d) = P_{d_0} - 10\\,n \\log_{10}\\!\\left(\\frac{d}{d_0}\\right),

and a single observation is :math:`\\mu(d) + X_\\sigma`.  Because the shadowing
term is Gaussian *in dB*, a Gaussian likelihood over dBm readings is the
correct model rather than a convenient one, which is the assumption
:mod:`symloc.likelihood` rests on.

Two access points at different places therefore have different means at the
same reference point, and the candidate axis becomes a real position instead of
a shared signal level.  That is the whole difference between
:func:`pathloss_radio_map` and :func:`symloc.ap.uniform_radio_map`.

References
----------
.. [1] T. S. Rappaport, *Wireless Communications: Principles and Practice*,
       2nd ed., Prentice Hall, 2002.
"""

from __future__ import annotations

import logging

import numpy as np

from .ap import APError, RadioMap
from .grid import Grid

_LOGGER = logging.getLogger(__name__)

#: Typical indoor path loss exponents, for reference in configuration files.
TYPICAL_EXPONENTS: dict[str, float] = {
    "free_space": 2.0,
    "office_open": 2.6,
    "office_partitioned": 3.3,
    "cluttered_factory": 3.8,
}


class PathLossError(ValueError):
    """Raised when a path loss model is given impossible parameters."""


def mean_rssi_dbm(
    distance_m: np.ndarray | float,
    reference_rssi_dbm: float = -40.0,
    path_loss_exponent: float = 3.0,
    reference_distance_m: float = 1.0,
    min_distance_m: float = 0.5,
) -> np.ndarray:
    """Mean received level at a given distance under the log-distance model.

    Parameters
    ----------
    distance_m : array_like or float
        Distance from the transmitter, in metres.
    reference_rssi_dbm : float, optional
        Received level at ``reference_distance_m``, in dBm.
    path_loss_exponent : float, optional
        The exponent :math:`n`.  See :data:`TYPICAL_EXPONENTS`.
    reference_distance_m : float, optional
        The reference distance :math:`d_0`, in metres.
    min_distance_m : float, optional
        Distances are clipped up to this value before the logarithm, so a
        reference point sitting exactly under an access point gives a finite
        mean instead of ``+inf``.

    Returns
    -------
    numpy.ndarray
        Mean RSSI in dBm, same shape as ``distance_m``.

    Raises
    ------
    PathLossError
        If the exponent, the reference distance or the clip is not positive, or
        if any distance is negative.

    Examples
    --------
    >>> float(mean_rssi_dbm(1.0, reference_rssi_dbm=-40.0, path_loss_exponent=3.0))
    -40.0
    >>> float(mean_rssi_dbm(10.0, reference_rssi_dbm=-40.0, path_loss_exponent=3.0))
    -70.0
    """
    if not path_loss_exponent > 0:
        raise PathLossError(f"path_loss_exponent must be positive, got {path_loss_exponent!r}")
    if not reference_distance_m > 0:
        raise PathLossError(f"reference_distance_m must be positive, got {reference_distance_m!r}")
    if not min_distance_m > 0:
        raise PathLossError(f"min_distance_m must be positive, got {min_distance_m!r}")

    distance = np.asarray(distance_m, dtype=np.float64)
    if np.any(distance < 0):
        raise PathLossError("distances must be non-negative")

    clipped = np.maximum(distance, min_distance_m)
    return reference_rssi_dbm - 10.0 * path_loss_exponent * np.log10(clipped / reference_distance_m)


def pathloss_radio_map(
    grid: Grid,
    ap_positions_m: np.ndarray,
    ap_names: list[str] | tuple[str, ...],
    reference_rssi_dbm: float | np.ndarray = -40.0,
    path_loss_exponent: float | np.ndarray = 3.0,
    reference_distance_m: float = 1.0,
    min_distance_m: float = 0.5,
) -> RadioMap:
    """Build a radio map by placing access points on the floor plan.

    Each access point gets its own mean at each reference point, from its own
    distance to that point.  The resulting map is non-degenerate, so the
    candidate axis carries geometry and a localization error can be measured
    in metres.

    Parameters
    ----------
    grid : Grid
        A grid of reference points; ``grid.coordinates`` must be set.
    ap_positions_m : array_like
        Shape ``(n_ap, 2)`` access point positions in metres.
    ap_names : sequence of str
        Names, in the same order as ``ap_positions_m``.
    reference_rssi_dbm : float or array_like, optional
        Received level at the reference distance, shared or one per access
        point.
    path_loss_exponent : float or array_like, optional
        Path loss exponent, shared or one per access point.
    reference_distance_m, min_distance_m : float, optional
        Passed through to :func:`mean_rssi_dbm`.

    Returns
    -------
    RadioMap
        Shape ``(n_ap, grid.size)``.

    Raises
    ------
    PathLossError
        If the grid has no coordinates or the shapes disagree.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.grid import Grid
    >>> g = Grid.rectangular(4.0, 4.0, 2.0)
    >>> rm = pathloss_radio_map(g, np.array([[0.0, 0.0], [4.0, 4.0]]), ["A", "B"])
    >>> rm.means_dbm.shape, rm.is_degenerate
    ((2, 9), False)
    """
    if grid.coordinates is None:
        raise PathLossError(
            "pathloss_radio_map needs a grid of reference points with coordinates; "
            "build one with Grid.rectangular or Grid.from_points"
        )
    positions = np.asarray(ap_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise PathLossError(f"ap_positions_m must have shape (n_ap, 2), got {positions.shape}")
    names = tuple(str(n) for n in ap_names)
    if len(names) != positions.shape[0]:
        raise PathLossError(f"{positions.shape[0]} access point positions but {len(names)} names")

    n_ap = positions.shape[0]
    ref = np.broadcast_to(np.asarray(reference_rssi_dbm, dtype=np.float64), (n_ap,))
    exponent = np.broadcast_to(np.asarray(path_loss_exponent, dtype=np.float64), (n_ap,))

    # (n_ap, n_points) distances, fully vectorised over both axes.
    deltas = positions[:, None, :] - grid.coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)

    # The whole map in one expression: mean_rssi_dbm's own arithmetic, with the
    # per-access-point reference level and exponent broadcast down the rows,
    # rather than one call per access point.
    if not np.all(exponent > 0):
        raise PathLossError(f"every path_loss_exponent must be positive, got {exponent.tolist()}")
    if not reference_distance_m > 0:
        raise PathLossError(f"reference_distance_m must be positive, got {reference_distance_m!r}")
    if not min_distance_m > 0:
        raise PathLossError(f"min_distance_m must be positive, got {min_distance_m!r}")
    clipped = np.maximum(distances, min_distance_m)
    means = ref[:, None] - 10.0 * exponent[:, None] * np.log10(clipped / reference_distance_m)

    _LOGGER.debug(
        "built path loss radio map: %d access points over %d reference points",
        n_ap,
        grid.size,
    )
    try:
        return RadioMap(means_dbm=means, ap_names=names)
    except APError as exc:  # pragma: no cover - defensive, means are finite by construction
        raise PathLossError(str(exc)) from exc


def sample_shadowing(
    means_dbm: np.ndarray,
    sigma_db: np.ndarray | float,
    rng: np.random.Generator,
    size: int = 1,
) -> np.ndarray:
    """Draw observations from the log-normal shadowing model.

    Parameters
    ----------
    means_dbm : array_like
        Shape ``(n_ap,)`` mean levels at the true position.
    sigma_db : array_like or float
        Shape ``(n_ap,)`` or scalar shadowing standard deviations in dB.
    rng : numpy.random.Generator
        Seeded generator; the caller owns the seed so runs are reproducible.
    size : int, optional
        Number of independent observation vectors to draw.

    Returns
    -------
    numpy.ndarray
        Shape ``(size, n_ap)`` observations in dBm.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> sample_shadowing(np.array([-70.0, -75.0]), 4.0, rng, size=5).shape
    (5, 2)
    """
    means = np.asarray(means_dbm, dtype=np.float64)
    if means.ndim != 1:
        raise PathLossError(f"means_dbm must be one-dimensional, got shape {means.shape}")
    sigma = np.broadcast_to(np.asarray(sigma_db, dtype=np.float64), means.shape)
    if np.any(sigma <= 0):
        raise PathLossError("every sigma_db must be strictly positive")
    if size < 1:
        raise PathLossError(f"size must be at least 1, got {size}")
    return rng.normal(loc=means, scale=sigma, size=(int(size), means.size))
