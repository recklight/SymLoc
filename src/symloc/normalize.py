"""Turning an unnormalised score into a distribution -- and keeping what that discards.

Dividing a curve by its own total over the candidate grid produces a
probability mass function, which is what makes two curves comparable *in
shape*.  It also destroys two things the argument actually needs, and this
module keeps both.

**The absolute magnitude.** The claim being tested is that tampering with an
access point makes the fingerprint match collapse.  That collapse is a change
in the absolute likelihood, and normalising each curve by its own total removes
it exactly: an attacked curve and an honest curve both end up summing to one.
:class:`Curve` therefore records ``log_evidence``, the log of the unnormalised
total, which is the quantity that shows the collapse.

**The mass that fell off the edge.** When an observation lies well outside the
swept range -- a spoofed reading of -99 dBm against a grid stopping at -85 --
most of that access point's likelihood mass is outside the grid, so the
normalisation rescales it by an amount that depends on how far outside it went.
Left unchecked this manufactures results: with the parameters of the original
script the naive-Bayes curve peaks *higher* under attack than the unattacked
control, which is an artefact of truncation and not a finding.
:func:`truncated_mass` measures it, and :func:`check_truncation` warns.

Finally, the mode matters.  Dividing by the sum gives a mass function whose
peak height scales with the grid step, so a peak of 0.0387 at 0.5 dB spacing
becomes 0.0194 at 0.25 dB spacing with nothing else changed.  Dividing by
``sum * step`` instead gives a density, whose peak is invariant.  Both are
offered; whichever is used travels with the result, together with the step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .ap import Scenario
from .grid import Grid

_LOGGER = logging.getLogger(__name__)

#: Normalisation modes accepted by :func:`normalize_log`.
NORMALIZE_MODES: tuple[str, ...] = ("sum", "density", "none")


class NormalizeError(ValueError):
    """Raised when a curve cannot be normalised."""


@dataclass(frozen=True, slots=True, eq=False)
class Curve:
    """One combined score over the grid, normalised and unnormalised.

    Attributes
    ----------
    label : str
        Identifier used in legends and result tables, e.g. ``"ESP (M=1)"``.
    grid : Grid
        The candidate grid the score was evaluated on.
    log_score : numpy.ndarray
        Shape ``(n_points,)`` unnormalised log score from the combiner.
    values : numpy.ndarray
        Shape ``(n_points,)`` normalised curve, in the units implied by ``mode``.
    mode : str
        One of :data:`NORMALIZE_MODES`.
    log_evidence : float
        ``logsumexp(log_score)``: the log of the unnormalised total over the
        grid.  Differences in this quantity between an attacked scenario and
        its honest control are the absolute evidence collapse, which
        normalisation would otherwise hide.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> curve = Curve.from_log_score("demo", g, np.zeros(g.size))
    >>> float(curve.values.sum()), curve.mode
    (1.0, 'sum')
    """

    label: str
    grid: Grid
    log_score: np.ndarray
    values: np.ndarray
    mode: str
    log_evidence: float
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_log_score(
        cls,
        label: str,
        grid: Grid,
        log_score: np.ndarray,
        mode: str = "sum",
        metadata: dict[str, object] | None = None,
    ) -> Curve:
        """Normalise a log score and wrap it up with its provenance.

        Parameters
        ----------
        label : str
            Curve identifier.
        grid : Grid
            Candidate grid; supplies the step used by ``"density"`` mode.
        log_score : array_like
            Shape ``(n_points,)`` unnormalised log score.
        mode : str, optional
            One of :data:`NORMALIZE_MODES`.
        metadata : dict, optional
            Extra fields carried into result tables.

        Returns
        -------
        Curve
            The normalised curve.
        """
        values, log_total = normalize_log(log_score, mode=mode, step=grid.step)
        return cls(
            label=str(label),
            grid=grid,
            log_score=np.asarray(log_score, dtype=np.float64),
            values=values,
            mode=mode,
            log_evidence=log_total,
            metadata=dict(metadata or {}),
        )

    @property
    def peak(self) -> float:
        """Height of the tallest point of the normalised curve."""
        return float(self.values.max())

    @property
    def argmax_index(self) -> int:
        """Index of the tallest point."""
        return int(np.argmax(self.values))

    @property
    def argmax_value(self) -> float:
        """Position of the tallest point on the grid axis."""
        return float(self.grid.values[self.argmax_index])


def normalize_log(
    log_score: np.ndarray,
    mode: str = "sum",
    step: float | None = None,
) -> tuple[np.ndarray, float]:
    """Normalise a log score, returning the curve and the log of its total.

    Parameters
    ----------
    log_score : array_like
        Shape ``(n_points,)`` unnormalised log score.
    mode : str, optional
        ``"sum"``
            Divide by the total, giving a probability **mass** function that
            sums to one.  Peak heights scale with the grid step.
        ``"density"``
            Divide by ``total * step``, giving a **density** that integrates to
            one.  Peak heights are invariant to the step, so curves computed at
            different resolutions can be compared.
        ``"none"``
            Exponentiate without normalising, preserving absolute magnitude.
    step : float, optional
        Grid spacing, required by ``"density"``.

    Returns
    -------
    values : numpy.ndarray
        The normalised curve.
    log_total : float
        ``logsumexp(log_score)``, the log unnormalised total.

    Raises
    ------
    NormalizeError
        If the mode is unknown, the step is missing or non-positive for
        ``"density"``, or the score is empty or entirely ``-inf``.

    Examples
    --------
    >>> import numpy as np
    >>> values, _ = normalize_log(np.zeros(4), mode="sum")
    >>> values.tolist()
    [0.25, 0.25, 0.25, 0.25]
    >>> values, _ = normalize_log(np.zeros(4), mode="density", step=0.5)
    >>> values.tolist()
    [0.5, 0.5, 0.5, 0.5]
    """
    if mode not in NORMALIZE_MODES:
        raise NormalizeError(
            f"unknown normalisation mode {mode!r}; choose one of {list(NORMALIZE_MODES)}"
        )
    array = np.asarray(log_score, dtype=np.float64)
    if array.ndim != 1:
        raise NormalizeError(f"log_score must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise NormalizeError("log_score is empty")
    if np.any(np.isnan(array)):
        raise NormalizeError("log_score contains nan")

    log_total = float(logsumexp(array))
    if not np.isfinite(log_total):
        raise NormalizeError(
            "log_score sums to zero probability over the whole grid; "
            "the observations are too far outside the candidate range to score"
        )

    if mode == "none":
        return np.exp(array), log_total
    if mode == "sum":
        return np.exp(array - log_total), log_total

    if step is None:
        raise NormalizeError("mode='density' needs the grid step")
    if not float(step) > 0:
        raise NormalizeError(f"step must be positive, got {step!r}")
    return np.exp(array - log_total) / float(step), log_total


def to_pmf(values: np.ndarray, step: float | None = None, mode: str = "sum") -> np.ndarray:
    """Normalise a linear-domain curve.

    A thin convenience over :func:`normalize_log` for callers holding linear
    values, such as the legacy reproduction.

    Parameters
    ----------
    values : array_like
        Shape ``(n_points,)`` non-negative curve.
    step : float, optional
        Grid spacing, required by ``mode="density"``.
    mode : str, optional
        As in :func:`normalize_log`.

    Returns
    -------
    numpy.ndarray
        The normalised curve.

    Raises
    ------
    NormalizeError
        If any value is negative.

    Examples
    --------
    >>> import numpy as np
    >>> to_pmf(np.array([1.0, 3.0])).round(6).tolist()
    [0.25, 0.75]
    """
    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0):
        raise NormalizeError("a likelihood curve cannot contain negative values")
    with np.errstate(divide="ignore"):
        log_values = np.log(array)
    normalized, _ = normalize_log(log_values, mode=mode, step=step)
    return normalized


def truncated_mass(scenario: Scenario) -> np.ndarray:
    """Fraction of each access point's likelihood mass lying outside the grid.

    Read as a function of the candidate mean, a single access point's Gaussian
    likelihood is itself a normalised density.  The share of it falling outside
    the range of means the radio map predicts for that access point is
    therefore exactly computable, and it is the factor by which grid
    normalisation silently rescales that access point.

    The bounds come from the radio map rather than from the grid axis, because
    the grid axis is in dBm only for a mean-RSSI sweep; on a floor plan it is a
    reference point index, and comparing a dBm reading against it would be a
    unit error.

    Parameters
    ----------
    scenario : Scenario
        The scenario to diagnose.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_ap,)`` fractions in ``[0, 1]``.

    Examples
    --------
    >>> from symloc.ap import APSpec, Scenario
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> sc = Scenario.build("demo", g, [APSpec("A", -70.0, 5.0), APSpec("B", -99.0, 5.0)])
    >>> round(float(truncated_mass(sc)[0]), 4), bool(truncated_mass(sc)[1] > 0.99)
    (0.0241, True)
    """
    spans = scenario.covered_mean_span()
    low = spans[:, 0]
    high = spans[:, 1]
    observations = scenario.observed_rssi_dbm
    sigma = scenario.sigma_db
    inside = norm.cdf((high - observations) / sigma) - norm.cdf((low - observations) / sigma)
    return np.clip(1.0 - inside, 0.0, 1.0)


def check_truncation(scenario: Scenario, warn_above: float = 0.5) -> dict[str, float]:
    """Log a warning for every access point losing too much mass off the grid.

    Parameters
    ----------
    scenario : Scenario
        The scenario to diagnose.
    warn_above : float, optional
        Fraction above which a warning is emitted.

    Returns
    -------
    dict of str to float
        Truncated fraction for every access point, keyed by name.

    Examples
    --------
    >>> from symloc.ap import APSpec, Scenario
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> sc = Scenario.build("demo", g, [APSpec("A", -70.0, 5.0)])
    >>> round(check_truncation(sc)["A"], 3)
    0.024
    """
    fractions = truncated_mass(scenario)
    spans = scenario.covered_mean_span()
    report: dict[str, float] = {}
    for index, (ap, fraction) in enumerate(zip(scenario.aps, fractions, strict=True)):
        report[ap.name] = float(fraction)
        if fraction > warn_above:
            low, high = spans[index]
            _LOGGER.warning(
                "scenario %r: %.1f%% of access point %s's likelihood mass lies outside "
                "the modelled means [%.1f, %.1f] dBm; grid normalisation rescales this "
                "curve by a factor that depends on the truncation, so peak comparisons "
                "against other curves are not on a common footing",
                scenario.name,
                100.0 * fraction,
                ap.name,
                low,
                high,
            )
    return report


def grid_resolution_note(grid: Grid, mode: str) -> str:
    """One-line description of how peak heights depend on this grid and mode.

    Written into every run manifest so a reported peak can always be read back
    with the resolution that produced it.

    Examples
    --------
    >>> from symloc.grid import Grid
    >>> grid_resolution_note(Grid.from_dbm_sweep(-85.0, -60.0, 0.5), "sum")
    'peaks are pmf masses on a 0.5 dBm grid (51 points); halving the step halves them'
    """
    if mode == "sum":
        return (
            f"peaks are pmf masses on a {grid.step:g} {grid.unit} grid ({grid.size} points); "
            "halving the step halves them"
        )
    if mode == "density":
        return (
            f"peaks are densities per {grid.unit} on a {grid.step:g} {grid.unit} grid "
            f"({grid.size} points); they are invariant to the step"
        )
    return f"values are unnormalised likelihoods on a {grid.step:g} {grid.unit} grid"
