"""How the per-access-point likelihoods are merged into one score per candidate.

The family this project is about, written ``ESP(M)``, takes every subset of
:math:`N-M` access points out of :math:`N`, multiplies the likelihoods inside
each subset, and adds the subsets up:

.. math::

   \\mathrm{ESP}(M) \\;=\\;
   \\sum_{\\substack{S \\subseteq \\{1..N\\} \\\\ |S| = N-M}} \\; \\prod_{k \\in S} p_k .

That expression is the **elementary symmetric polynomial** (ESP) of order
:math:`N-M` in the likelihoods, :math:`e_{N-M}(p_1, \\ldots, p_N)`.  Naming it
is what makes the family tractable: the original code expanded each order by
hand, one algebraic expression per access point count, so the analysis could
never leave the three- and four-access-point cases it was typed out for.  The
recurrence below computes every order for any :math:`N` in :math:`O(N k)`.

Reading off the two ends:

* :math:`M = 0` gives :math:`e_N = \\prod_k p_k`, the ordinary naive-Bayes
  fingerprint likelihood -- one tampered access point drives it to nearly zero;
* :math:`M = N-1` gives :math:`e_1 = \\sum_k p_k`, a plain sum, which no single
  access point can veto but which is also barely peaked.

So :math:`M` is the number of tampered access points the score is designed to
tolerate, and the parameter trades sharpness against robustness.

The recurrence is the coefficient extraction of

.. math::

   \\prod_{k=1}^{N} (x + p_k) = \\sum_{j=0}^{N} e_j(p)\\, x^{N-j},

computed here in the **log domain**, where multiplication becomes addition and
addition becomes :func:`numpy.logaddexp`.  That matters: in the linear domain a
product of fifteen densities underflows to exactly zero and the normalised
distribution becomes ``nan``, while the log recurrence stays exact.

Alongside the symmetric-polynomial family this module registers three robust
combiners from the statistics literature, so that the ESP idea can be
compared against the standard tools for the same problem rather than only
against itself.

References
----------
.. [1] I. G. Macdonald, *Symmetric Functions and Hall Polynomials*, 2nd ed.,
       Oxford University Press, 1995, Chapter I.2.
.. [2] P. J. Rousseeuw, "Least Median of Squares Regression", *Journal of the
       American Statistical Association*, vol. 79, no. 388, pp. 871-880, 1984.
.. [3] P. J. Huber, *Robust Statistics*, John Wiley & Sons, 1981.
.. [4] D. Liu, P. Ning, and W. Du, "Attack-Resistant Location Estimation in
       Sensor Networks", *Proc. ACM/IEEE IPSN*, pp. 99-106, 2005.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from itertools import combinations

import numpy as np
from scipy.special import logsumexp

_LOGGER = logging.getLogger(__name__)

#: Signature every registered combiner obeys.
CombinerFn = Callable[..., np.ndarray]

_REGISTRY: dict[str, CombinerFn] = {}


class CombinerError(ValueError):
    """Raised when a combiner is given an unusable order, strategy or array."""


def register(name: str) -> Callable[[CombinerFn], CombinerFn]:
    """Register a combining strategy under ``name``.

    Parameters
    ----------
    name : str
        Key used in configuration files and on the command line.

    Returns
    -------
    callable
        A decorator that stores the function and returns it unchanged.
    """

    def decorator(func: CombinerFn) -> CombinerFn:
        if name in _REGISTRY:
            raise CombinerError(f"combiner {name!r} is already registered")
        _REGISTRY[name] = func
        return func

    return decorator


def available_combiners() -> tuple[str, ...]:
    """Names of every registered strategy, sorted.

    Examples
    --------
    >>> available_combiners()
    ('esp', 'median', 'product', 'softmin', 'sum', 'trimmed')
    """
    return tuple(sorted(_REGISTRY))


def _validate_log_p(log_p: np.ndarray) -> np.ndarray:
    array = np.asarray(log_p, dtype=np.float64)
    if array.ndim != 2:
        raise CombinerError(
            f"log-likelihoods must be 2-D (n_ap, n_points), got shape {array.shape}"
        )
    if array.shape[0] < 1:
        raise CombinerError("at least one access point is required")
    if np.any(np.isnan(array)):
        raise CombinerError("log-likelihoods contain nan")
    if np.any(array == np.inf):
        raise CombinerError("log-likelihoods contain +inf")
    return array


def log_elementary_symmetric_all(log_p: np.ndarray, k_max: int | None = None) -> np.ndarray:
    """Every elementary symmetric polynomial up to order ``k_max``, in logs.

    One pass of the recurrence produces all orders, so a study that sweeps
    :math:`M` over its whole range costs the same as computing a single order.

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    k_max : int, optional
        Highest order to compute; defaults to ``n_ap``.

    Returns
    -------
    numpy.ndarray
        Shape ``(k_max + 1, n_points)``, row :math:`j` holding
        :math:`\\log e_j`.  Row 0 is zero, since :math:`e_0 = 1`.

    Raises
    ------
    CombinerError
        If ``k_max`` is negative or exceeds the number of access points.

    Examples
    --------
    >>> import numpy as np
    >>> log_p = np.log(np.array([[2.0], [3.0], [5.0]]))
    >>> np.exp(log_elementary_symmetric_all(log_p))[:, 0].round(6)
    array([ 1., 10., 31., 30.])
    """
    array = _validate_log_p(log_p)
    n_ap, n_points = array.shape
    order = n_ap if k_max is None else int(k_max)
    if order < 0:
        raise CombinerError(f"k_max must be non-negative, got {order}")
    if order > n_ap:
        raise CombinerError(
            f"k_max={order} exceeds the number of access points ({n_ap}); e_k vanishes for k > N"
        )

    table = np.full((order + 1, n_points), -np.inf, dtype=np.float64)
    table[0] = 0.0
    for index in range(n_ap):
        # Descending j keeps each update reading the previous row's value.
        for j in range(min(order, index + 1), 0, -1):
            table[j] = np.logaddexp(table[j], array[index] + table[j - 1])
    return table


def log_elementary_symmetric(log_p: np.ndarray, k: int) -> np.ndarray:
    """Log of the elementary symmetric polynomial of order ``k``.

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    k : int
        Order, between 0 and ``n_ap`` inclusive.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)``.

    Examples
    --------
    >>> import numpy as np
    >>> log_p = np.log(np.array([[2.0], [3.0], [5.0]]))
    >>> float(np.exp(log_elementary_symmetric(log_p, 2))[0])
    31.0
    """
    order = int(k)
    return log_elementary_symmetric_all(log_p, order)[order]


def elementary_symmetric_bruteforce(p: np.ndarray, k: int) -> np.ndarray:
    """Reference implementation of :math:`e_k` by explicit enumeration.

    Sums the product over every :math:`\\binom{N}{k}` subset.  This is the
    definition rather than the recurrence, so it is the right thing to test the
    recurrence against; it is exponential in :math:`N` and is not used by the
    pipeline.

    Parameters
    ----------
    p : array_like
        Shape ``(n_ap, n_points)`` linear-domain likelihoods.
    k : int
        Order.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)``.

    Examples
    --------
    >>> import numpy as np
    >>> float(elementary_symmetric_bruteforce(np.array([[2.0], [3.0], [5.0]]), 2)[0])
    31.0
    """
    array = np.asarray(p, dtype=np.float64)
    if array.ndim != 2:
        raise CombinerError(f"p must be 2-D (n_ap, n_points), got shape {array.shape}")
    n_ap, n_points = array.shape
    order = int(k)
    if not 0 <= order <= n_ap:
        raise CombinerError(f"order k={order} must lie in 0..{n_ap}")
    if order == 0:
        return np.ones(n_points, dtype=np.float64)
    total = np.zeros(n_points, dtype=np.float64)
    for subset in combinations(range(n_ap), order):
        total += np.prod(array[list(subset)], axis=0)
    return total


@register("esp")
def combine_esp(log_p: np.ndarray, m: int = 0, **_: object) -> np.ndarray:
    """ESP(``m``): the elementary symmetric polynomial of order :math:`N-m`.

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    m : int, optional
        Number of tampered access points to tolerate, from 0 to ``n_ap - 1``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)`` log scores.

    Raises
    ------
    CombinerError
        If ``m`` is outside ``0..n_ap - 1``.

    Examples
    --------
    >>> import numpy as np
    >>> log_p = np.log(np.array([[2.0], [3.0], [5.0]]))
    >>> round(float(np.exp(combine_esp(log_p, m=0))[0]), 6)
    30.0
    >>> round(float(np.exp(combine_esp(log_p, m=2))[0]), 6)
    10.0
    """
    array = _validate_log_p(log_p)
    n_ap = array.shape[0]
    order = int(m)
    if not 0 <= order <= n_ap - 1:
        raise CombinerError(
            f"M={order} is out of range for {n_ap} access points; "
            f"M must lie in 0..{n_ap - 1} so that at least one access point is kept"
        )
    return log_elementary_symmetric(array, n_ap - order)


@register("product")
def combine_product(log_p: np.ndarray, m: int = 0, **_: object) -> np.ndarray:
    """Naive-Bayes product of every likelihood, i.e. ESP(0).

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    m : int, optional
        Ignored; accepted so every strategy shares one signature.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)`` log scores.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(np.exp(combine_product(np.log(np.array([[2.0], [3.0]]))))[0]), 6)
    6.0
    """
    return _validate_log_p(log_p).sum(axis=0)


@register("sum")
def combine_sum(log_p: np.ndarray, m: int = 0, **_: object) -> np.ndarray:
    """Plain sum of the likelihoods, i.e. ESP(:math:`N-1`).

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    m : int, optional
        Ignored; accepted so every strategy shares one signature.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)`` log scores.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(np.exp(combine_sum(np.log(np.array([[2.0], [3.0]]))))[0]), 6)
    5.0
    """
    return logsumexp(_validate_log_p(log_p), axis=0)


@register("trimmed")
def combine_trimmed(log_p: np.ndarray, m: int = 0, **_: object) -> np.ndarray:
    """Least-trimmed score: drop the ``m`` worst access points, then multiply.

    At each candidate the ``m`` smallest log likelihoods are discarded and the
    remaining :math:`N-m` are summed.  Where ESP averages over *all* subsets
    of size :math:`N-m`, this keeps only the single best-fitting one, which is
    the classical high-breakdown construction.  The two answer the same
    question differently, and comparing them is the point of registering both.

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    m : int, optional
        Number of access points to discard, from 0 to ``n_ap - 1``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)`` log scores.

    Examples
    --------
    >>> import numpy as np
    >>> log_p = np.log(np.array([[2.0], [3.0], [100.0]]))
    >>> round(float(np.exp(combine_trimmed(log_p, m=1))[0]), 6)
    300.0
    """
    array = _validate_log_p(log_p)
    n_ap = array.shape[0]
    order = int(m)
    if not 0 <= order <= n_ap - 1:
        raise CombinerError(
            f"M={order} is out of range for {n_ap} access points; M must lie in 0..{n_ap - 1}"
        )
    if order == 0:
        return array.sum(axis=0)
    # Ascending sort puts the worst-fitting access points first; drop them.
    ordered = np.sort(array, axis=0)
    return ordered[order:].sum(axis=0)


@register("median")
def combine_median(log_p: np.ndarray, m: int = 0, **_: object) -> np.ndarray:
    """Median log likelihood, rescaled to :math:`N` terms.

    The median ignores up to half the access points in either direction, so its
    breakdown point is the highest of any combiner here, at the cost of
    discarding the information in the remaining readings.

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    m : int, optional
        Ignored; the median has no order parameter.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)`` log scores.

    Examples
    --------
    >>> import numpy as np
    >>> log_p = np.log(np.array([[2.0], [4.0], [1000.0]]))
    >>> round(float(np.exp(combine_median(log_p))[0]), 6)
    64.0
    """
    array = _validate_log_p(log_p)
    return array.shape[0] * np.median(array, axis=0)


@register("softmin")
def combine_softmin(log_p: np.ndarray, m: int = 0, beta: float = 1.0, **_: object) -> np.ndarray:
    """Soft minimum of the log likelihoods, rescaled to :math:`N` terms.

    Interpolates between the mean (``beta`` towards 0) and the worst single
    access point (``beta`` large), giving a continuous dial where ESP offers
    integer steps.

    Registered as a deliberate counter-example: weighting the *worst*-fitting
    access point most heavily is the opposite of robustness, so under a spoofing
    attack this combiner follows the tampered reading rather than rejecting it.
    ``examples/combiner_robustness.py`` measures the cost, and it is large.  It
    is kept because it is the natural conservative aggregator, and showing that
    the natural choice fails here is part of the argument for trimming.

    .. math::

       \\mathrm{softmin}_\\beta(\\ell)
         = -\\frac{1}{\\beta}\\left[
             \\log \\sum_k e^{-\\beta \\ell_k} - \\log N \\right].

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    m : int, optional
        Ignored; ``beta`` is this combiner's parameter.
    beta : float, optional
        Sharpness; must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)`` log scores.

    Examples
    --------
    >>> import numpy as np
    >>> log_p = np.array([[0.0], [-10.0]])
    >>> round(float(combine_softmin(log_p, beta=50.0)[0]), 1)
    -20.0
    """
    array = _validate_log_p(log_p)
    sharpness = float(beta)
    if not sharpness > 0:
        raise CombinerError(f"beta must be strictly positive, got {beta!r}")
    n_ap = array.shape[0]
    soft = -(logsumexp(-sharpness * array, axis=0) - np.log(n_ap)) / sharpness
    return n_ap * soft


def combine(
    log_p: np.ndarray,
    strategy: str = "esp",
    m: int = 0,
    **kwargs: object,
) -> np.ndarray:
    """Apply a registered combining strategy by name.

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    strategy : str, optional
        One of :func:`available_combiners`.
    m : int, optional
        Order parameter, interpreted by the strategy.
    **kwargs
        Extra strategy-specific arguments, e.g. ``beta`` for ``"softmin"``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)`` log scores.

    Raises
    ------
    CombinerError
        If ``strategy`` is not registered.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(np.exp(combine(np.log(np.array([[2.0], [3.0]])), "sum"))[0]), 6)
    5.0
    """
    try:
        func = _REGISTRY[strategy]
    except KeyError:
        raise CombinerError(
            f"unknown combiner {strategy!r}; choose one of {list(available_combiners())}"
        ) from None
    return func(log_p, m=m, **kwargs)


def combine_many(
    log_p: np.ndarray,
    m_values: Iterable[int],
    strategy: str = "esp",
    **kwargs: object,
) -> dict[int, np.ndarray]:
    """Apply one strategy at several orders, returning ``{m: log_score}``.

    For the ``"esp"`` strategy every order comes from a single pass of the
    recurrence rather than one pass per order.

    Parameters
    ----------
    log_p : array_like
        Shape ``(n_ap, n_points)`` log likelihoods.
    m_values : iterable of int
        Orders to evaluate.
    strategy : str, optional
        Strategy name.
    **kwargs
        Extra strategy-specific arguments.

    Returns
    -------
    dict of int to numpy.ndarray
        Log scores keyed by order.

    Examples
    --------
    >>> import numpy as np
    >>> scores = combine_many(np.log(np.array([[2.0], [3.0], [5.0]])), [0, 1, 2])
    >>> [round(float(np.exp(scores[m])[0]), 6) for m in (0, 1, 2)]
    [30.0, 31.0, 10.0]
    """
    array = _validate_log_p(log_p)
    n_ap = array.shape[0]
    orders = [int(value) for value in m_values]
    if not orders:
        raise CombinerError("m_values is empty; give at least one order")

    if strategy == "esp":
        for order in orders:
            if not 0 <= order <= n_ap - 1:
                raise CombinerError(
                    f"M={order} is out of range for {n_ap} access points; "
                    f"M must lie in 0..{n_ap - 1}"
                )
        table = log_elementary_symmetric_all(array, n_ap)
        return {order: table[n_ap - order] for order in orders}

    return {order: combine(array, strategy=strategy, m=order, **kwargs) for order in orders}
