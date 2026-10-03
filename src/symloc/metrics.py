"""How sharply a curve picks out a position, and how far two curves differ.

The original analysis judged discriminability with two numbers: the ratio and
the difference between the peaks of curves at neighbouring :math:`M`.  Both are
kept here for comparison, and both are weak -- a peak is one sample of a
51-point curve, so two distributions with completely different spreads and
completely different argmaxes can share a peak height.

This module adds the measures that actually describe the two properties the
argument needs:

**Divergence between two distributions** -- how far the attacked posterior has
moved from the honest one.  Kullback-Leibler in both directions (it is not
symmetric, and reporting one direction alone hides that), its symmetric bounded
relative Jensen-Shannon, and the Bhattacharyya and Hellinger overlap measures.

**Sharpness of one distribution** -- how confidently it picks a single
candidate.  Entropy (Shannon and Renyi), full width at half maximum, excess
kurtosis, the ratio of the peak to the next-highest local maximum, and the
effective support width.

**Localization error** -- how far the argmax sits from the truth, which is the
only measure here that answers the question a positioning system is actually
built to answer, and which the original work could not report at all because
its candidate axis had no geometry.

Conventions: every function takes distributions already normalised over the
grid, and every one is invariant to the grid step except where the docstring
says otherwise.  Where a formula needs bin widths, the grid is passed in.

References
----------
.. [1] S. Kullback and R. A. Leibler, "On Information and Sufficiency",
       *Annals of Mathematical Statistics*, vol. 22, no. 1, pp. 79-86, 1951.
.. [2] A. Bhattacharyya, "On a Measure of Divergence Between Two Statistical
       Populations Defined by Their Probability Distributions", *Bulletin of
       the Calcutta Mathematical Society*, vol. 35, pp. 99-109, 1943.
.. [3] T. Kailath, "The Divergence and Bhattacharyya Distance Measures in
       Signal Selection", *IEEE Transactions on Communication Technology*,
       vol. 15, no. 1, pp. 52-60, 1967.
.. [4] J. Lin, "Divergence Measures Based on the Shannon Entropy", *IEEE
       Transactions on Information Theory*, vol. 37, no. 1, pp. 145-151, 1991.
"""

from __future__ import annotations

import logging

import numpy as np

from .grid import Grid
from .normalize import Curve

_LOGGER = logging.getLogger(__name__)


class MetricError(ValueError):
    """Raised when a metric is given curves it cannot compare."""


def _as_pmf(values: np.ndarray, name: str = "p") -> np.ndarray:
    """Validate a discrete distribution and renormalise it to sum to one."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise MetricError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise MetricError(f"{name} is empty")
    if np.any(np.isnan(array)):
        raise MetricError(f"{name} contains nan")
    if np.any(array < 0):
        raise MetricError(f"{name} contains negative values; a distribution cannot")
    total = array.sum()
    if not total > 0:
        raise MetricError(f"{name} sums to zero and cannot be normalised")
    return array / total


def _pair(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate two distributions of matching length."""
    left = _as_pmf(p, "p")
    right = _as_pmf(q, "q")
    if left.size != right.size:
        raise MetricError(
            f"distributions must share a grid: got {left.size} and {right.size} points"
        )
    return left, right


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence :math:`D(p \\| q)` in nats.

    Terms where ``p`` is zero contribute nothing, following the convention
    :math:`0 \\log 0 = 0`.  Where ``p`` is positive and ``q`` is zero the
    divergence is genuinely infinite and ``inf`` is returned rather than a
    smoothed stand-in, because silently smoothing turns "this candidate is
    impossible" into a finite penalty.

    Parameters
    ----------
    p, q : array_like
        Distributions over the same grid.

    Returns
    -------
    float
        Divergence in nats, non-negative.

    Examples
    --------
    >>> round(kl_divergence([0.5, 0.5], [0.5, 0.5]), 12)
    0.0
    >>> round(kl_divergence([1.0, 0.0], [0.5, 0.5]), 6)
    0.693147
    """
    left, right = _pair(p, q)
    support = left > 0
    if np.any(right[support] <= 0):
        return float("inf")
    return float(np.sum(left[support] * np.log(left[support] / right[support])))


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Jeffreys divergence :math:`D(p\\|q) + D(q\\|p)`.

    Examples
    --------
    >>> round(symmetric_kl([0.5, 0.5], [0.5, 0.5]), 12)
    0.0
    """
    return kl_divergence(p, q) + kl_divergence(q, p)


def js_divergence(p: np.ndarray, q: np.ndarray, base: float | None = None) -> float:
    """Jensen-Shannon divergence: symmetric, always finite, bounded.

    Unlike KL this is defined even where the two distributions have disjoint
    support, and it is bounded above by :math:`\\log 2` in nats, so values from
    different scenarios are directly comparable.

    Parameters
    ----------
    p, q : array_like
        Distributions over the same grid.
    base : float, optional
        Logarithm base.  ``2`` puts the result in bits and on ``[0, 1]``.

    Returns
    -------
    float
        Divergence, in nats unless ``base`` is given.

    Examples
    --------
    >>> round(js_divergence([1.0, 0.0], [0.0, 1.0], base=2), 6)
    1.0
    >>> round(js_divergence([0.5, 0.5], [0.5, 0.5]), 12)
    0.0
    """
    left, right = _pair(p, q)
    mixture = 0.5 * (left + right)
    value = 0.5 * kl_divergence(left, mixture) + 0.5 * kl_divergence(right, mixture)
    if base is not None:
        if not base > 1:
            raise MetricError(f"base must be greater than 1, got {base!r}")
        value = value / float(np.log(base))
    return float(value)


def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    """Overlap :math:`BC = \\sum_i \\sqrt{p_i q_i}`, in ``[0, 1]``.

    One means the distributions are identical, zero that their supports are
    disjoint.

    Examples
    --------
    >>> round(bhattacharyya_coefficient([0.5, 0.5], [0.5, 0.5]), 12)
    1.0
    >>> round(bhattacharyya_coefficient([1.0, 0.0], [0.0, 1.0]), 12)
    0.0
    """
    left, right = _pair(p, q)
    return float(np.sum(np.sqrt(left * right)))


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """:math:`-\\log BC`, zero for identical distributions and ``inf`` for disjoint ones.

    Examples
    --------
    >>> round(bhattacharyya_distance([0.5, 0.5], [0.5, 0.5]), 12)
    0.0
    """
    coefficient = bhattacharyya_coefficient(p, q)
    if coefficient <= 0:
        return float("inf")
    # BC <= 1 makes -log(BC) non-negative; clamping removes the -0.0 that
    # floating-point rounding produces for identical distributions.
    return float(max(0.0, -np.log(coefficient)))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance :math:`\\sqrt{1 - BC}`, a true metric on ``[0, 1]``.

    Examples
    --------
    >>> round(hellinger_distance([1.0, 0.0], [0.0, 1.0]), 12)
    1.0
    """
    coefficient = bhattacharyya_coefficient(p, q)
    return float(np.sqrt(max(0.0, 1.0 - coefficient)))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance :math:`\\tfrac12 \\sum_i |p_i - q_i|`.

    Examples
    --------
    >>> round(total_variation([1.0, 0.0], [0.0, 1.0]), 12)
    1.0
    """
    left, right = _pair(p, q)
    return float(0.5 * np.sum(np.abs(left - right)))


def shannon_entropy(p: np.ndarray, base: float | None = None) -> float:
    """Shannon entropy of a discrete distribution.

    Depends on the grid resolution: a finer grid spreads the same shape over
    more bins and raises the entropy by roughly :math:`\\log` of the refinement
    factor.  Compare entropies only across curves on one grid.

    Parameters
    ----------
    p : array_like
        Distribution.
    base : float, optional
        Logarithm base; ``2`` gives bits.

    Returns
    -------
    float
        Entropy, in nats unless ``base`` is given.

    Examples
    --------
    >>> round(shannon_entropy([0.5, 0.5], base=2), 12)
    1.0
    >>> round(shannon_entropy([1.0, 0.0]), 12)
    0.0
    """
    array = _as_pmf(p)
    support = array > 0
    # Entropy is non-negative; clamping removes the -0.0 a degenerate
    # distribution produces, which would otherwise print with a minus sign.
    value = max(0.0, float(-np.sum(array[support] * np.log(array[support]))))
    if base is not None:
        if not base > 1:
            raise MetricError(f"base must be greater than 1, got {base!r}")
        value = value / float(np.log(base))
    return float(value)


def renyi_entropy(p: np.ndarray, alpha: float = 2.0, base: float | None = None) -> float:
    """Renyi entropy of order ``alpha``.

    ``alpha`` towards 1 recovers the Shannon entropy; ``alpha = 2`` (the
    collision entropy) weights the peak more heavily and is the more sensitive
    sharpness measure for these curves.

    Parameters
    ----------
    p : array_like
        Distribution.
    alpha : float, optional
        Order; must be positive and not exactly 1.
    base : float, optional
        Logarithm base.

    Returns
    -------
    float
        Entropy, in nats unless ``base`` is given.

    Examples
    --------
    >>> round(renyi_entropy([0.5, 0.5], alpha=2.0, base=2), 12)
    1.0
    """
    array = _as_pmf(p)
    order = float(alpha)
    if not order > 0:
        raise MetricError(f"alpha must be positive, got {alpha!r}")
    if abs(order - 1.0) < 1e-12:
        return shannon_entropy(array, base=base)
    value = float(np.log(np.sum(array**order)) / (1.0 - order))
    if base is not None:
        if not base > 1:
            raise MetricError(f"base must be greater than 1, got {base!r}")
        value = value / float(np.log(base))
    return float(value)


def peak_ratio(p: np.ndarray, q: np.ndarray) -> float:
    """Ratio of peak heights, the original discriminability measure ``rho``.

    Kept so that results can be checked against the archived figures.  It
    compares one sample of each curve and says nothing about their spread.

    Examples
    --------
    >>> round(peak_ratio([0.4, 0.6], [0.5, 0.5]), 6)
    1.2
    """
    left, right = _pair(p, q)
    denominator = float(right.max())
    if denominator <= 0:
        return float("inf")
    return float(left.max() / denominator)


def peak_difference(p: np.ndarray, q: np.ndarray) -> float:
    """Difference of peak heights, the original measure ``Delta``.

    Examples
    --------
    >>> round(peak_difference([0.4, 0.6], [0.5, 0.5]), 6)
    0.1
    """
    left, right = _pair(p, q)
    return float(left.max() - right.max())


def full_width_half_maximum(values: np.ndarray, grid: Grid) -> float:
    """Width of the curve at half its peak height, in grid units.

    The crossings are located by linear interpolation between the two samples
    that straddle the half-maximum level, so the answer is not quantised to the
    grid step.  Returns ``nan`` when the curve does not fall back below half
    its maximum inside the grid, which is itself a useful signal: the curve is
    truncated.

    Parameters
    ----------
    values : array_like
        Curve over the grid; need not be normalised.
    grid : Grid
        The grid ``values`` was evaluated on.

    Returns
    -------
    float
        Width in the unit of ``grid``, or ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-10.0, 10.0, 0.1)
    >>> bell = np.exp(-0.5 * (g.values / 2.0) ** 2)
    >>> round(full_width_half_maximum(bell, g), 3)   # 2*sqrt(2 ln 2)*sigma
    4.71
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape != grid.values.shape:
        raise MetricError(f"curve has {array.shape} points but the grid has {grid.values.shape}")
    peak = float(array.max())
    if peak <= 0:
        return float("nan")
    half = 0.5 * peak
    index = int(np.argmax(array))

    def _crossing(start: int, step: int) -> float | None:
        """Walk outwards from the peak and interpolate the half-maximum crossing."""
        position = start
        while 0 <= position + step < array.size:
            neighbour = position + step
            if array[position] >= half > array[neighbour]:
                span = array[position] - array[neighbour]
                weight = 0.0 if span == 0 else (array[position] - half) / span
                return float(
                    grid.values[position]
                    + weight * (grid.values[neighbour] - grid.values[position])
                )
            position = neighbour
        return None

    right = _crossing(index, +1)
    left = _crossing(index, -1)

    if left is None or right is None:
        _LOGGER.debug("curve does not return below half maximum inside the grid")
        return float("nan")
    return abs(right - left)


def excess_kurtosis(values: np.ndarray, grid: Grid) -> float:
    """Excess kurtosis of the distribution over the grid axis.

    Zero for a Gaussian, positive for a sharper peak with heavier tails.

    Parameters
    ----------
    values : array_like
        Curve over the grid.
    grid : Grid
        The grid ``values`` was evaluated on.

    Returns
    -------
    float
        Excess kurtosis, or ``nan`` for a degenerate distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-30.0, 30.0, 0.05)
    >>> bell = np.exp(-0.5 * (g.values / 3.0) ** 2)
    >>> abs(excess_kurtosis(bell, g)) < 0.01
    True
    """
    array = _as_pmf(values, "values")
    if array.size != grid.size:
        raise MetricError(f"curve has {array.size} points but the grid has {grid.size}")
    axis = grid.values
    mean = float(np.sum(array * axis))
    centred = axis - mean
    variance = float(np.sum(array * centred**2))
    if variance <= 0:
        return float("nan")
    fourth = float(np.sum(array * centred**4))
    return float(fourth / variance**2 - 3.0)


def _local_maxima(array: np.ndarray) -> np.ndarray:
    """Heights of every local maximum, endpoints and flat tops included.

    A mode is a maximal run of equal values that is higher than whatever lies
    immediately outside it.  Defining it that way, rather than as a point
    strictly greater than both neighbours, matters twice here:

    * the two ends of the grid are modes too.  A spoofed reading pushes the
      posterior's peak against the edge of the swept range, so a rule that only
      looks at interior points can miss the tallest peak altogether -- and then
      report the ratio between two *lesser* modes as though it were the answer;
    * a flat top, which two neighbouring candidates produce whenever they tie,
      is one mode, not none.

    Returns the heights in descending order.
    """
    # One representative height per maximal run of equal values.
    heights = array[np.r_[True, array[1:] != array[:-1]]]
    # A run is a mode when it is higher than the run before and the run after;
    # a missing neighbour (at either end of the grid) never vetoes it.
    higher_than_left = np.r_[True, heights[1:] > heights[:-1]]
    higher_than_right = np.r_[heights[:-1] > heights[1:], True]
    modes = heights[higher_than_left & higher_than_right]
    return np.sort(modes)[::-1]


def peak_to_secondary_ratio(values: np.ndarray) -> float:
    """Ratio of the tallest peak to the next-highest local maximum.

    A localizer is ambiguous when a second candidate scores nearly as well, and
    this ratio is what says so.  Returns ``inf`` when the curve is unimodal.

    Local maxima are found by :func:`_local_maxima`, which counts the grid
    endpoints and flat tops as modes -- a curve that peaks against the edge of
    the swept range is the normal outcome of a spoofed reading, and it is
    exactly the case where ambiguity has to be reported rather than hidden.

    Parameters
    ----------
    values : array_like
        Curve over the grid.

    Returns
    -------
    float
        Ratio, at least 1, or ``inf``.

    Examples
    --------
    >>> peak_to_secondary_ratio([0.0, 1.0, 0.0, 0.5, 0.0])
    2.0
    >>> peak_to_secondary_ratio([0.0, 1.0, 0.0])
    inf
    >>> peak_to_secondary_ratio([0.5, 0.2, 1.0, 0.2, 0.45])   # a mode at the edge
    2.0
    >>> peak_to_secondary_ratio([0.0, 1.0, 1.0, 0.0, 0.5, 0.0])   # a flat top
    2.0
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 3:
        raise MetricError("peak_to_secondary_ratio needs at least three points")
    ordered = _local_maxima(array)
    if ordered.size < 2:
        return float("inf")
    if ordered[1] <= 0:
        return float("inf")
    return float(ordered[0] / ordered[1])


def effective_support_width(values: np.ndarray, grid: Grid, fraction: float = 0.9) -> float:
    """Width of the smallest set of candidates holding ``fraction`` of the mass.

    Unlike the full width at half maximum this needs no assumption that the
    curve is unimodal, and unlike the entropy it is reported in the unit of the
    axis, so it reads directly as "the answer is somewhere inside this many dB".

    Parameters
    ----------
    values : array_like
        Curve over the grid.
    grid : Grid
        The grid ``values`` was evaluated on.
    fraction : float, optional
        Mass to cover, in ``(0, 1]``.

    Returns
    -------
    float
        Width in the unit of ``grid``.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-10.0, 10.0, 0.5)
    >>> flat = np.ones(g.size)
    >>> round(effective_support_width(flat, g, fraction=1.0), 3)
    20.5
    """
    array = _as_pmf(values, "values")
    if array.size != grid.size:
        raise MetricError(f"curve has {array.size} points but the grid has {grid.size}")
    if not 0 < fraction <= 1:
        raise MetricError(f"fraction must lie in (0, 1], got {fraction!r}")
    ordered = np.sort(array)[::-1]
    cumulative = np.cumsum(ordered)
    count = int(np.searchsorted(cumulative, fraction * (1.0 - 1e-12)) + 1)
    count = min(count, array.size)
    return float(count * grid.step)


def argmax_offset(values: np.ndarray, grid: Grid, true_index: int) -> float:
    """Signed distance from the curve's argmax to the true position.

    On a dBm sweep the answer is in dB; on a grid of reference points with
    coordinates it is the Euclidean distance in metres, which is the honest
    localization error.

    Parameters
    ----------
    values : array_like
        Curve over the grid.
    grid : Grid
        The grid ``values`` was evaluated on.
    true_index : int
        Index of the true position.

    Returns
    -------
    float
        Offset in the unit of the grid, or metres when coordinates exist.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> curve = np.zeros(g.size); curve[30] = 1.0
    >>> argmax_offset(curve, g, true_index=28)
    1.0
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size != grid.size:
        raise MetricError(f"curve has {array.size} points but the grid has {grid.size}")
    index = int(true_index)
    if not 0 <= index < grid.size:
        raise MetricError(f"true_index {index} is outside the grid (0..{grid.size - 1})")
    estimated = int(np.argmax(array))
    if grid.coordinates is not None:
        delta = grid.coordinates[estimated] - grid.coordinates[index]
        return float(np.linalg.norm(delta))
    return float(grid.values[estimated] - grid.values[index])


def curve_metrics(curve: Curve, reference: Curve | None = None) -> dict[str, float]:
    """Every sharpness metric for one curve, plus divergences against a reference.

    Parameters
    ----------
    curve : Curve
        The curve to describe.
    reference : Curve, optional
        Control curve, normally the unattacked baseline.  When given, the
        divergence measures and the evidence gap are included.

    Returns
    -------
    dict of str to float
        Metric name to value.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.grid import Grid
    >>> from symloc.normalize import Curve
    >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> flat = Curve.from_log_score("flat", g, np.zeros(g.size))
    >>> round(curve_metrics(flat)["peak"], 6)
    0.019608
    """
    values = curve.values
    grid = curve.grid
    metrics: dict[str, float] = {
        "peak": float(values.max()),
        "argmax_value": curve.argmax_value,
        "shannon_entropy": shannon_entropy(values),
        "renyi_entropy_2": renyi_entropy(values, alpha=2.0),
        "fwhm": full_width_half_maximum(values, grid),
        "excess_kurtosis": excess_kurtosis(values, grid),
        "peak_to_secondary_ratio": peak_to_secondary_ratio(values),
        "effective_support_90": effective_support_width(values, grid, fraction=0.9),
        "log_evidence": float(curve.log_evidence),
    }
    if reference is not None:
        metrics.update(
            {
                "kl_to_reference": kl_divergence(values, reference.values),
                "kl_from_reference": kl_divergence(reference.values, values),
                "js_divergence": js_divergence(values, reference.values),
                "bhattacharyya_distance": bhattacharyya_distance(values, reference.values),
                "hellinger_distance": hellinger_distance(values, reference.values),
                "total_variation": total_variation(values, reference.values),
                "peak_ratio_to_reference": peak_ratio(values, reference.values),
                "peak_difference_to_reference": peak_difference(values, reference.values),
                # The quantity grid normalisation destroys: how much absolute
                # evidence the tampering removed, in nats.
                "log_evidence_gap": float(curve.log_evidence - reference.log_evidence),
            }
        )
    return metrics
