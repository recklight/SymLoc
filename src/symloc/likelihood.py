"""Per-access-point Gaussian likelihood of an RSSI observation.

Under log-normal shadowing (:mod:`symloc.pathloss`) a reading in dBm from access
point :math:`k` at candidate position :math:`j` is Gaussian about the mean the
radio map predicts there, so its likelihood is

.. math::

   p_k(j) = \\frac{1}{\\sigma_k \\sqrt{2\\pi}}
            \\exp\\!\\left[-\\frac{(r_k - \\mu_{kj})^2}{2\\sigma_k^2}\\right],

with :math:`r_k` the observation and :math:`\\mu_{kj}` the expected mean.  This
is the standard per-access-point term of probabilistic WLAN fingerprinting; the
combiners in :mod:`symloc.combiners` decide how the :math:`N` terms are merged.

**Everything here is computed in the log domain.**  A product of :math:`N`
Gaussian densities underflows quickly -- with four access points the smallest
values are already around :math:`10^{-8}`, and a realistic deployment with
fifteen access points drives the product to exactly zero in float64, at which
point the normalised distribution becomes ``nan``.  Working with log densities
and combining them with ``logaddexp`` keeps the arithmetic exact over the whole
range, and the linear values are recovered once at the end.

A note on a piece of arithmetic that looks wrong and is not: the original code
evaluated ``normpdf(-r, -mu, sigma)``, negating both the observation and the
mean.  A Gaussian is symmetric about its mean, so negating both leaves the
density unchanged and the pair of minus signs is decorative.  This module
writes :math:`(r - \\mu)/\\sigma` directly.

References
----------
.. [1] T. Roos, P. Myllymaki, H. Tirri, P. Misikangas, and J. Sievanen,
       "A Probabilistic Approach to WLAN User Location Estimation",
       *International Journal of Wireless Information Networks*, vol. 9, no. 3,
       pp. 155-164, 2002.
.. [2] M. Youssef and A. Agrawala, "The Horus WLAN Location Determination
       System", *Proc. ACM MobiSys*, pp. 205-218, 2005.
"""

from __future__ import annotations

import logging

import numpy as np

from .ap import RadioMap, Scenario

_LOGGER = logging.getLogger(__name__)

#: ``log(sqrt(2*pi))``, the constant term of a Gaussian log density.
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


class LikelihoodError(ValueError):
    """Raised when observations, means and standard deviations do not agree."""


def log_likelihood(
    observed_rssi_dbm: np.ndarray,
    means_dbm: np.ndarray,
    sigma_db: np.ndarray,
) -> np.ndarray:
    """Log density of every observation at every candidate position.

    Written out directly rather than through ``scipy.stats.norm.logpdf`` so
    that the whole ``(n_ap, n_points)`` block is one vectorised expression with
    no Python-level loop over access points or candidates.

    Parameters
    ----------
    observed_rssi_dbm : array_like
        Shape ``(n_ap,)`` observations in dBm.
    means_dbm : array_like
        Shape ``(n_ap, n_points)`` expected means from the radio map.
    sigma_db : array_like
        Shape ``(n_ap,)`` shadowing standard deviations in dB.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_ap, n_points)`` log densities.

    Raises
    ------
    LikelihoodError
        If the shapes disagree, or any standard deviation is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> lp = log_likelihood(np.array([-70.0]), np.array([[-70.0, -80.0]]), np.array([5.0]))
    >>> bool(np.isclose(np.exp(lp[0, 0]), 1.0 / (5.0 * np.sqrt(2 * np.pi))))
    True
    """
    observations = np.asarray(observed_rssi_dbm, dtype=np.float64)
    means = np.asarray(means_dbm, dtype=np.float64)
    sigma = np.asarray(sigma_db, dtype=np.float64)

    if observations.ndim != 1:
        raise LikelihoodError(
            f"observed_rssi_dbm must be one-dimensional, got shape {observations.shape}"
        )
    if means.ndim != 2:
        raise LikelihoodError(f"means_dbm must be 2-D (n_ap, n_points), got {means.shape}")
    if sigma.ndim != 1:
        raise LikelihoodError(f"sigma_db must be one-dimensional, got shape {sigma.shape}")
    if means.shape[0] != observations.size:
        raise LikelihoodError(
            f"{observations.size} observations but the radio map has {means.shape[0]} rows"
        )
    if sigma.size != observations.size:
        raise LikelihoodError(
            f"{observations.size} observations but {sigma.size} standard deviations"
        )
    if np.any(sigma <= 0):
        bad = np.flatnonzero(sigma <= 0).tolist()
        raise LikelihoodError(f"every sigma_db must be strictly positive; offending indices {bad}")
    if not np.all(np.isfinite(observations)):
        raise LikelihoodError("observations must all be finite")

    scaled = (observations[:, None] - means) / sigma[:, None]
    return -0.5 * scaled * scaled - np.log(sigma)[:, None] - _LOG_SQRT_2PI


def likelihood(
    observed_rssi_dbm: np.ndarray,
    means_dbm: np.ndarray,
    sigma_db: np.ndarray,
) -> np.ndarray:
    """Linear-domain counterpart of :func:`log_likelihood`.

    Provided for inspection and for the legacy reproduction; the analysis
    pipeline itself stays in the log domain.

    Parameters
    ----------
    observed_rssi_dbm, means_dbm, sigma_db : array_like
        As in :func:`log_likelihood`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_ap, n_points)`` densities.

    Examples
    --------
    >>> import numpy as np
    >>> p = likelihood(np.array([-70.0]), np.array([[-70.0]]), np.array([5.0]))
    >>> bool(np.isclose(p[0, 0], 1.0 / (5.0 * np.sqrt(2 * np.pi))))
    True
    """
    return np.exp(log_likelihood(observed_rssi_dbm, means_dbm, sigma_db))


def scenario_log_likelihood(scenario: Scenario) -> np.ndarray:
    """Log likelihood block for a whole :class:`~symloc.ap.Scenario`.

    Parameters
    ----------
    scenario : Scenario
        A validated scenario.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_ap, n_points)`` log densities.

    Examples
    --------
    >>> from symloc.ap import APSpec, Scenario
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> sc = Scenario.build("demo", g, [APSpec("A", -70.0, 8.0), APSpec("B", -70.0, 10.0)])
    >>> scenario_log_likelihood(sc).shape
    (2, 51)
    """
    out_of_grid = scenario.out_of_grid_observations()
    if out_of_grid:
        _LOGGER.warning(
            "scenario %r: observations of %s fall outside the candidate grid %s; "
            "most of their probability mass lies beyond the grid edge and grid "
            "normalisation will rescale them",
            scenario.name,
            ", ".join(out_of_grid),
            scenario.grid.span,
        )
    return log_likelihood(
        scenario.observed_rssi_dbm,
        scenario.radio_map.means_dbm,
        scenario.sigma_db,
    )


def log_likelihood_for_observations(
    observations: np.ndarray,
    radio_map: RadioMap,
    sigma_db: np.ndarray,
) -> np.ndarray:
    """Log likelihood blocks for a batch of observation vectors.

    Used by the Monte-Carlo driver, which needs many trials scored against one
    radio map without a Python loop over trials.

    Parameters
    ----------
    observations : array_like
        Shape ``(n_trials, n_ap)`` observations in dBm.
    radio_map : RadioMap
        Expected means, shape ``(n_ap, n_points)``.
    sigma_db : array_like
        Shape ``(n_ap,)`` standard deviations.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_trials, n_ap, n_points)`` log densities.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.ap import RadioMap
    >>> rm = RadioMap(np.array([[-60.0, -70.0], [-65.0, -75.0]]), ("A", "B"))
    >>> obs = np.array([[-70.0, -75.0], [-60.0, -65.0]])
    >>> log_likelihood_for_observations(obs, rm, np.array([5.0, 5.0])).shape
    (2, 2, 2)
    """
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 2:
        raise LikelihoodError(f"observations must be 2-D (n_trials, n_ap), got shape {obs.shape}")
    if obs.shape[1] != radio_map.n_ap:
        raise LikelihoodError(
            f"observations carry {obs.shape[1]} access points but the radio map has "
            f"{radio_map.n_ap}"
        )
    sigma = np.asarray(sigma_db, dtype=np.float64)
    if sigma.size != radio_map.n_ap:
        raise LikelihoodError(
            f"{sigma.size} standard deviations but {radio_map.n_ap} access points"
        )
    if np.any(sigma <= 0):
        raise LikelihoodError("every sigma_db must be strictly positive")

    scaled = (obs[:, :, None] - radio_map.means_dbm[None, :, :]) / sigma[None, :, None]
    return -0.5 * scaled * scaled - np.log(sigma)[None, :, None] - _LOG_SQRT_2PI
