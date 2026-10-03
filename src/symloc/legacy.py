"""Faithful reproduction of the original script, defects included.

This module exists so that the archived figures can be regenerated exactly,
and so that the difference between what they show and what the mathematics
gives can be stated as a number rather than as an assertion.  Nothing here is
used by the analysis pipeline; it is a regression target.

Two defects are reproduced on purpose, both switchable:

**The sign error.** In the three-access-point block the third pairwise product
of :math:`e_2` was written with the observation un-negated while its mean was
negated -- evaluating the density at *positive* 70 dBm against a mean near
:math:`-70`.  That term is about :math:`8 \\times 10^{-39}` instead of about
:math:`7 \\times 10^{-4}`, so one of the three pairwise products is silently
dropped.  The archived :math:`M = 1` curve carries it: its peak of 0.0305445
matches the buggy expression to six figures, while the correct :math:`e_2`
peaks at 0.0283886.  **The archived peak is 7.6% too high, and that peak is
the quantity the discriminability argument rests on.**

**Truncated normalisation.** Each curve is divided by its own total over the
truncated grid, so an access point whose observation lies far outside the swept
range is rescaled by an arbitrary amount.  With the surviving parameters this
makes the attacked naive-Bayes curve peak *above* the unattacked control, which
reads as a result and is an artefact.

The reference values below were recovered by solving for the parameters that
reproduce the archived figures' curve peaks, then recomputing from scratch.
The one-attack case matches to within :math:`1.2 \\times 10^{-7}`; the
two-attack case does not match any parameter set reachable from the surviving
code, which is what having no run manifest costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .combiners import log_elementary_symmetric_all
from .grid import Grid
from .likelihood import log_likelihood
from .normalize import to_pmf

_LOGGER = logging.getLogger(__name__)

#: Grid the original analysis swept: -60 down to -85 dBm in 0.5 dB steps.
LEGACY_GRID_LOW_DBM = -85.0
LEGACY_GRID_HIGH_DBM = -60.0
LEGACY_GRID_STEP_DB = 0.5

#: Parameters recovered from the archived one-attack figure by numerical
#: inversion of its four curve peaks.  The filename claimed the tampered access
#: point reported -99 dBm; the curves say -90 dBm.
LEGACY_ONE_ATTACK = {
    "observed_rssi_dbm": (-70.0, -70.0, -90.0),
    "sigma_db": (8.0, 10.0, 10.0),
    "honest_rssi_dbm": -70.0,
}

#: Peaks read off the archived one-attack figure, and what the corrected
#: mathematics gives for the same parameters.  Every value here is reproduced
#: by this module to at least six significant figures.
LEGACY_ONE_ATTACK_PEAKS = {
    "baseline": 0.0387524,
    "m0": 0.0390380,
    "m1_archived": 0.0305445,
    "m1_corrected": 0.0283886,
    "m2": 0.0242794,
}

#: Adjacent-order peak ratios, as archived (carrying the sign error) and as
#: corrected.  That pair is what the archived figure's legend reports.
LEGACY_ONE_ATTACK_RATIOS = {
    "archived": {"rho_0_1": 1.27807, "rho_1_2": 1.25805},
    "corrected": {"rho_0_1": 1.37513, "rho_1_2": 1.16925},
}

#: Peak heights taken from the archived bar figures.  The parameters that
#: produced these were never recorded and cannot be recovered from the
#: surviving code, so they are kept as provenance only and are not reproduced.
LEGACY_UNRECOVERED_BARS = {
    "one_attack": (0.048671, 0.035257, 0.027938),
    "two_attack": (0.096875, 0.087499, 0.026119),
}


@dataclass(frozen=True, slots=True, eq=False)
class LegacyResult:
    """Curves from the original formulation, normalised the way it normalised.

    Attributes
    ----------
    grid : Grid
        The swept candidate grid.
    baseline : numpy.ndarray
        The unattacked control curve.
    curves : dict of int to numpy.ndarray
        Curves keyed by :math:`M`.
    sign_bug : bool
        Whether the reproduced :math:`e_2` carries the original sign error.
    """

    grid: Grid
    baseline: np.ndarray
    curves: dict[int, np.ndarray]
    sign_bug: bool

    def peaks(self) -> dict[str, float]:
        """Peak of every curve, keyed ``"baseline"``, ``"m0"``, ``"m1"``, ...

        Examples
        --------
        >>> round(reproduce_one_attack(sign_bug=True).peaks()["m1"], 7)
        0.0305445
        """
        out = {"baseline": float(self.baseline.max())}
        for order, values in sorted(self.curves.items()):
            out[f"m{order}"] = float(values.max())
        return out

    def adjacent_peak_ratios(self) -> dict[str, float]:
        """Adjacent-order peak ratios, the original ``rho``.

        Examples
        --------
        >>> ratios = reproduce_one_attack(sign_bug=True).adjacent_peak_ratios()
        >>> round(ratios["rho_0_1"], 5)
        1.27807
        """
        orders = sorted(self.curves)
        return {
            f"rho_{low}_{high}": float(self.curves[low].max() / self.curves[high].max())
            for low, high in pairwise(orders)
        }

    def adjacent_peak_differences(self) -> dict[str, float]:
        """Adjacent-order peak differences, the original ``Delta``."""
        orders = sorted(self.curves)
        return {
            f"delta_{low}_{high}": float(self.curves[low].max() - self.curves[high].max())
            for low, high in pairwise(orders)
        }


def legacy_grid() -> Grid:
    """The 51-point sweep the original script used.

    Examples
    --------
    >>> legacy_grid().size
    51
    """
    return Grid.from_dbm_sweep(LEGACY_GRID_LOW_DBM, LEGACY_GRID_HIGH_DBM, LEGACY_GRID_STEP_DB)


def legacy_curves(
    observed_rssi_dbm: tuple[float, ...],
    sigma_db: tuple[float, ...],
    honest_rssi_dbm: float = -70.0,
    grid: Grid | None = None,
    sign_bug: bool = False,
) -> LegacyResult:
    """Reproduce the original curves for an arbitrary number of access points.

    The combination is computed with the same recurrence the rest of the
    package uses -- the original hand-expanded algebra is *equal* to it, which
    is the point -- except for the optional sign error, which is not equal to
    anything and so is applied explicitly.

    Parameters
    ----------
    observed_rssi_dbm : tuple of float
        What each access point reports, in dBm.
    sigma_db : tuple of float
        Shadowing standard deviations, in dB.
    honest_rssi_dbm : float, optional
        Reading every access point would give if none were tampered with; used
        for the control curve.
    grid : Grid, optional
        Candidate grid; defaults to :func:`legacy_grid`.
    sign_bug : bool, optional
        When true, reproduce the original sign error in :math:`e_2` for the
        three-access-point case.

    Returns
    -------
    LegacyResult
        Curves normalised by their own totals, as the original did.

    Raises
    ------
    ValueError
        If ``sign_bug`` is requested for anything other than three access
        points, where the defect actually occurred.

    Examples
    --------
    >>> result = legacy_curves((-70.0, -70.0, -90.0), (8.0, 10.0, 10.0), sign_bug=True)
    >>> round(result.peaks()["m1"], 7)
    0.0305445
    """
    observations = np.asarray(observed_rssi_dbm, dtype=np.float64)
    sigma = np.asarray(sigma_db, dtype=np.float64)
    if observations.size != sigma.size:
        raise ValueError(f"{observations.size} observations but {sigma.size} standard deviations")
    n_ap = int(observations.size)
    if n_ap < 2:
        raise ValueError(f"the original formulation needs at least two access points, got {n_ap}")
    if sign_bug and n_ap != 3:
        raise ValueError(
            "the sign error occurred in the three-access-point block; "
            f"it has no counterpart for {n_ap} access points"
        )

    working_grid = legacy_grid() if grid is None else grid
    means = np.tile(working_grid.values, (n_ap, 1))

    log_p = log_likelihood(observations, means, sigma)
    table = log_elementary_symmetric_all(log_p, n_ap)

    curves: dict[int, np.ndarray] = {}
    for order in range(n_ap):
        curves[order] = to_pmf(np.exp(table[n_ap - order]), step=working_grid.step, mode="sum")

    if sign_bug:
        curves[1] = to_pmf(
            _buggy_e2(observations, sigma, working_grid),
            step=working_grid.step,
            mode="sum",
        )
        _LOGGER.warning(
            "reproducing the original sign error in e_2: one of the three pairwise "
            "products is evaluated at +%.0f dBm instead of %.0f dBm and is therefore "
            "zero; the resulting peak is about 7.6%% too high",
            abs(observations[1]),
            observations[1],
        )

    honest = np.full(n_ap, float(honest_rssi_dbm), dtype=np.float64)
    honest_log_p = log_likelihood(honest, means, sigma)
    baseline = to_pmf(np.exp(honest_log_p.sum(axis=0)), step=working_grid.step, mode="sum")

    return LegacyResult(
        grid=working_grid, baseline=baseline, curves=curves, sign_bug=bool(sign_bug)
    )


def _buggy_e2(observations: np.ndarray, sigma: np.ndarray, grid: Grid) -> np.ndarray:
    """The three-access-point :math:`e_2` exactly as it was written, sign error included.

    The original expansion was ``pA*pB + pA*pC + pB*pC``, and in the third term
    ``pB`` was evaluated at ``+|r_B|`` rather than ``r_B``.  Because the mean
    stays negative, that density is around :math:`10^{-39}` and the term
    vanishes.
    """
    means = np.tile(grid.values, (3, 1))
    p = np.exp(log_likelihood(observations, means, sigma))
    # The defect: observation 1 has its sign flipped while its mean does not.
    flipped = observations.copy()
    flipped[1] = -flipped[1]
    p_flipped = np.exp(log_likelihood(flipped, means, sigma))
    return p[0] * p[1] + p[0] * p[2] + p_flipped[1] * p[2]


def reproduce_one_attack(sign_bug: bool = True) -> LegacyResult:
    """Reproduce the archived one-attack figure from the recovered parameters.

    Parameters
    ----------
    sign_bug : bool, optional
        ``True`` reproduces the figure as archived; ``False`` gives the same
        scenario with the arithmetic corrected.

    Returns
    -------
    LegacyResult
        The four curves of the archived figure.

    Examples
    --------
    >>> reproduced = reproduce_one_attack(sign_bug=True).peaks()
    >>> corrected = reproduce_one_attack(sign_bug=False).peaks()
    >>> round(reproduced["m1"], 7), round(corrected["m1"], 7)
    (0.0305445, 0.0283886)
    """
    return legacy_curves(
        observed_rssi_dbm=LEGACY_ONE_ATTACK["observed_rssi_dbm"],  # type: ignore[arg-type]
        sigma_db=LEGACY_ONE_ATTACK["sigma_db"],  # type: ignore[arg-type]
        honest_rssi_dbm=float(LEGACY_ONE_ATTACK["honest_rssi_dbm"]),  # type: ignore[arg-type]
        sign_bug=sign_bug,
    )


def sign_bug_impact(
    reproduced: LegacyResult | None = None,
    corrected: LegacyResult | None = None,
) -> dict[str, float]:
    """Quantify what the sign error did to the archived result.

    Parameters
    ----------
    reproduced, corrected : LegacyResult, optional
        Results from :func:`reproduce_one_attack` with and without the sign
        error.  A caller that already holds them passes them in rather than
        having them recomputed -- which also keeps the warning the buggy branch
        logs from being emitted twice for one report.

    Returns
    -------
    dict of str to float
        ``archived_peak``, ``corrected_peak``, ``relative_error`` (the
        fractional overstatement) and the two pairs of adjacent-order ratios.

    Examples
    --------
    >>> impact = sign_bug_impact()
    >>> round(100 * impact["relative_error"], 1)
    7.6
    """
    if reproduced is None:
        reproduced = reproduce_one_attack(sign_bug=True)
    if corrected is None:
        corrected = reproduce_one_attack(sign_bug=False)
    archived_peak = reproduced.peaks()["m1"]
    corrected_peak = corrected.peaks()["m1"]
    return {
        "archived_peak": archived_peak,
        "corrected_peak": corrected_peak,
        "relative_error": archived_peak / corrected_peak - 1.0,
        "archived_rho_0_1": reproduced.adjacent_peak_ratios()["rho_0_1"],
        "corrected_rho_0_1": corrected.adjacent_peak_ratios()["rho_0_1"],
        "archived_rho_1_2": reproduced.adjacent_peak_ratios()["rho_1_2"],
        "corrected_rho_1_2": corrected.adjacent_peak_ratios()["rho_1_2"],
    }


def truncation_artefact() -> dict[str, float]:
    """Show the attacked product curve peaking above the unattacked control.

    With the parameters left in the surviving script the :math:`M = 0` curve,
    which has a tampered access point in it, has a *higher* peak than the
    control with no tampering at all.  Nothing about the attack made the fix
    better; the attacked access point simply lost more mass off the edge of the
    grid, so dividing by a smaller total raised what remained.

    Returns
    -------
    dict of str to float
        The two peaks and their ratio.

    Examples
    --------
    >>> artefact = truncation_artefact()
    >>> artefact["m0_peak"] > artefact["baseline_peak"]
    True
    """
    result = legacy_curves(
        observed_rssi_dbm=(-70.0, -70.0, -88.0),
        sigma_db=(8.0, 10.0, 10.0),
        honest_rssi_dbm=-70.0,
        sign_bug=False,
    )
    peaks = result.peaks()
    return {
        "baseline_peak": peaks["baseline"],
        "m0_peak": peaks["m0"],
        "ratio": peaks["m0"] / peaks["baseline"],
    }
