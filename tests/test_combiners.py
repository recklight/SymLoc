"""The elementary-symmetric recurrence, checked against the definition.

The recurrence is the load-bearing piece of the whole package: it replaces the
hand-expanded algebra the original analysis used, and every curve depends on it
being right.  So it is tested against explicit enumeration over subsets, which
is the definition rather than a second implementation of the same trick.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from symloc.combiners import (
    CombinerError,
    available_combiners,
    combine,
    combine_many,
    combine_median,
    combine_softmin,
    combine_trimmed,
    elementary_symmetric_bruteforce,
    log_elementary_symmetric,
    log_elementary_symmetric_all,
)


@pytest.mark.parametrize("n_ap", [1, 2, 3, 4, 5, 6])
def test_recurrence_matches_brute_force_enumeration(n_ap, random_log_likelihoods):
    """log e_k from the recurrence equals the sum over all subsets of size k."""
    log_p = random_log_likelihoods(n_ap=n_ap, n_points=7)
    p = np.exp(log_p)
    for k in range(n_ap + 1):
        recurrence = np.exp(log_elementary_symmetric(log_p, k))
        definition = elementary_symmetric_bruteforce(p, k)
        assert np.allclose(recurrence, definition, rtol=1e-10)


def test_all_orders_in_one_pass_match_individual_orders(random_log_likelihoods):
    """The single-pass table agrees with computing each order on its own."""
    log_p = random_log_likelihoods(n_ap=5, n_points=11)
    table = log_elementary_symmetric_all(log_p)
    for k in range(6):
        assert np.allclose(table[k], log_elementary_symmetric(log_p, k), rtol=1e-12)


def test_e0_is_one_and_eN_is_the_product(random_log_likelihoods):
    """The two endpoints of the family are the constant 1 and the full product."""
    log_p = random_log_likelihoods(n_ap=4, n_points=6)
    table = log_elementary_symmetric_all(log_p)
    assert np.allclose(np.exp(table[0]), 1.0)
    assert np.allclose(table[4], log_p.sum(axis=0), rtol=1e-12)


def test_esp_orders_map_onto_symmetric_polynomials(random_log_likelihoods):
    """ESP(M) is e_{N-M}: M=0 is the product and M=N-1 is the plain sum."""
    log_p = random_log_likelihoods(n_ap=4, n_points=6)
    p = np.exp(log_p)
    scores = combine_many(log_p, [0, 1, 2, 3])
    assert np.allclose(np.exp(scores[0]), p.prod(axis=0), rtol=1e-10)
    assert np.allclose(np.exp(scores[3]), p.sum(axis=0), rtol=1e-10)
    assert np.allclose(np.exp(scores[1]), elementary_symmetric_bruteforce(p, 3), rtol=1e-10)
    assert np.allclose(np.exp(scores[2]), elementary_symmetric_bruteforce(p, 2), rtol=1e-10)


def test_hand_expansions_from_the_original_formulation_are_the_same_polynomials():
    """The algebra the original wrote out by hand equals e_3 and e_2.

    These are the exact expressions from the four-access-point block.  They look
    nothing like the recurrence, which is precisely why this identity is worth
    pinning down rather than assuming.
    """
    rng = np.random.default_rng(0)
    pa, pb, pc, pd = rng.uniform(0.1, 1.5, size=(4, 13))
    log_p = np.log(np.vstack([pa, pb, pc, pd]))

    hand_e3 = pa * pb * (pc + pd) + pc * pd * (pa + pb)
    hand_e2 = pa * (pb + pc + pd) + pb * (pc + pd) + pc * pd

    assert np.allclose(np.exp(log_elementary_symmetric(log_p, 3)), hand_e3, rtol=1e-12)
    assert np.allclose(np.exp(log_elementary_symmetric(log_p, 2)), hand_e2, rtol=1e-12)


def test_log_domain_survives_underflow_that_kills_the_linear_domain():
    """With many access points the linear product underflows and the log one does not.

    This is the numerical failure the original formulation walks into: at
    fifteen access points the product of the densities is exactly zero in
    float64, so the normalised curve becomes nan.
    """
    n_ap = 15
    log_p = np.full((n_ap, 5), np.log(1e-25))

    linear_product = np.prod(np.exp(log_p), axis=0)
    assert np.all(linear_product == 0.0), "expected the linear product to underflow"

    log_score = log_elementary_symmetric(log_p, n_ap)
    assert np.all(np.isfinite(log_score))
    assert np.allclose(log_score, n_ap * np.log(1e-25), rtol=1e-12)


def test_log_domain_esp_matches_high_precision_reference():
    """Every order stays exact where the linear domain has already collapsed."""
    n_ap = 12
    rng = np.random.default_rng(7)
    log_p = rng.uniform(-80.0, -40.0, size=(n_ap, 3))
    table = log_elementary_symmetric_all(log_p)

    # Reference: enumerate subsets in the log domain with logaddexp, which is a
    # different algorithm from the recurrence even though both avoid underflow.
    for k in (1, 2, 11, 12):
        accumulator = np.full(3, -np.inf)
        for subset in combinations(range(n_ap), k):
            accumulator = np.logaddexp(accumulator, log_p[list(subset)].sum(axis=0))
        assert np.allclose(table[k], accumulator, rtol=1e-12)


def test_trimmed_discards_the_worst_fitting_access_points():
    """The trimmed combiner keeps the best N-M terms and drops the rest."""
    log_p = np.log(np.array([[2.0], [3.0], [100.0], [0.001]]))
    assert np.isclose(np.exp(combine_trimmed(log_p, m=0))[0], 2.0 * 3.0 * 100.0 * 0.001)
    assert np.isclose(np.exp(combine_trimmed(log_p, m=1))[0], 2.0 * 3.0 * 100.0)
    assert np.isclose(np.exp(combine_trimmed(log_p, m=2))[0], 3.0 * 100.0)


def test_median_is_insensitive_to_a_single_extreme_value():
    """One tampered access point cannot move the median combiner."""
    clean = np.log(np.array([[1.0], [1.0], [1.0]]))
    spoofed = np.log(np.array([[1.0], [1.0], [1e-30]]))
    assert np.isclose(combine_median(clean)[0], combine_median(spoofed)[0])


def test_softmin_interpolates_between_mean_and_minimum():
    """Large beta approaches N*min; small beta approaches N*mean."""
    log_p = np.array([[0.0], [-10.0], [-2.0]])
    n_ap = 3
    sharp = combine_softmin(log_p, beta=200.0)[0]
    smooth = combine_softmin(log_p, beta=1e-4)[0]
    # The residual at large beta is N*log(N)/beta, which is about 0.016 here.
    assert np.isclose(sharp, n_ap * log_p.min(), atol=n_ap * np.log(n_ap) / 200.0 + 1e-6)
    assert np.isclose(smooth, n_ap * log_p.mean(), atol=1e-2)


def test_every_registered_combiner_returns_one_score_per_candidate(random_log_likelihoods):
    """The registry's contract: (n_ap, n_points) in, (n_points,) out."""
    log_p = random_log_likelihoods(n_ap=4, n_points=17)
    for name in available_combiners():
        score = combine(log_p, strategy=name, m=1)
        assert score.shape == (17,), name
        assert np.all(np.isfinite(score)), name


@pytest.mark.parametrize("bad_m", [-1, 4, 99])
def test_out_of_range_order_is_rejected_with_a_useful_message(bad_m, random_log_likelihoods):
    """M must leave at least one access point in play."""
    log_p = random_log_likelihoods(n_ap=4, n_points=3)
    with pytest.raises(CombinerError, match="out of range"):
        combine(log_p, strategy="esp", m=bad_m)


def test_unknown_combiner_lists_the_available_ones(random_log_likelihoods):
    log_p = random_log_likelihoods(n_ap=3, n_points=3)
    with pytest.raises(CombinerError, match="esp"):
        combine(log_p, strategy="not_a_combiner")


def test_malformed_input_is_rejected():
    """A one-dimensional array is a common slip and must not be silently reshaped."""
    with pytest.raises(CombinerError, match="2-D"):
        log_elementary_symmetric(np.zeros(5), 1)
    with pytest.raises(CombinerError, match="nan"):
        log_elementary_symmetric(np.array([[np.nan, 0.0]]), 1)
    with pytest.raises(CombinerError, match="exceeds"):
        log_elementary_symmetric_all(np.zeros((2, 3)), k_max=5)


def test_combine_many_rejects_an_empty_order_list(random_log_likelihoods):
    with pytest.raises(CombinerError, match="empty"):
        combine_many(random_log_likelihoods(n_ap=3, n_points=3), [])
