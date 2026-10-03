"""Parity with the archived results, and the size of the defects in them.

These tests pin two things at once: that this package reproduces the archived
figures exactly from the recovered parameters, and that the difference between
what they show and what the mathematics gives is the stated size.  If either
drifts, something in the pipeline has changed meaning.

The reference values live in ``data/legacy_reference/peaks.json`` and are
derived numbers only -- peak heights recovered by numerical inversion.  No
original file is copied into this repository.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from symloc.combiners import log_elementary_symmetric
from symloc.config import Config
from symloc.legacy import (
    LEGACY_ONE_ATTACK,
    LEGACY_ONE_ATTACK_PEAKS,
    LEGACY_ONE_ATTACK_RATIOS,
    legacy_curves,
    legacy_grid,
    reproduce_one_attack,
    sign_bug_impact,
    truncation_artefact,
)
from symloc.likelihood import log_likelihood
from symloc.normalize import to_pmf
from symloc.pipeline import analyse_scenario


@pytest.fixture
def reference(project_root) -> dict:
    """Read-only reference values recovered from the archived figures."""
    path = project_root / "data" / "legacy_reference" / "peaks.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_grid_has_the_fifty_one_points_the_original_swept():
    grid = legacy_grid()
    assert grid.size == 51
    assert grid.step == 0.5
    assert grid.span == (-85.0, -60.0)


def test_reproduces_every_archived_peak(reference):
    """All four archived curve peaks come back from the recovered parameters."""
    archived = reference["one_attack"]["archived_peaks"]
    peaks = reproduce_one_attack(sign_bug=True).peaks()
    for key, expected in archived.items():
        assert peaks[key] == pytest.approx(expected, abs=5e-7), key


def test_reproduces_the_archived_peak_ratios(reference):
    """The discriminability numbers in the archived legend come back too."""
    archived = reference["one_attack"]["archived_ratios"]
    ratios = reproduce_one_attack(sign_bug=True).adjacent_peak_ratios()
    for key, expected in archived.items():
        assert ratios[key] == pytest.approx(expected, abs=5e-6), key
    assert LEGACY_ONE_ATTACK_RATIOS["archived"] == pytest.approx(archived, abs=5e-6)


def test_the_corrected_curve_differs_from_the_archived_one_by_the_stated_amount(reference):
    """The sign error overstates the archived M=1 peak by about 7.6%."""
    entry = reference["one_attack"]
    impact = sign_bug_impact()

    assert impact["archived_peak"] == pytest.approx(entry["archived_peaks"]["m1"], abs=5e-7)
    assert impact["corrected_peak"] == pytest.approx(entry["corrected_peaks"]["m1"], abs=5e-7)
    assert impact["relative_error"] == pytest.approx(
        entry["sign_error"]["relative_overstatement"], abs=5e-4
    )
    # The defect inflates the peak, so the archived value is the larger one.
    assert impact["archived_peak"] > impact["corrected_peak"]


def test_the_sign_error_reverses_which_pair_of_orders_looks_more_separable(reference):
    """The defect does not merely shift a number; it changes the conclusion.

    As archived, rho(0,1) > rho(1,2) only slightly.  Corrected, the gap widens
    considerably in the same direction -- and the corrected rho(1,2) drops
    below the archived one, so any claim resting on the archived pair is
    resting on the defect.
    """
    impact = sign_bug_impact()
    assert impact["archived_rho_1_2"] > impact["corrected_rho_1_2"]
    assert impact["corrected_rho_0_1"] > impact["archived_rho_0_1"]


def test_the_corrected_pipeline_agrees_with_the_corrected_legacy_path(project_root):
    """The analysis pipeline and the legacy reproduction meet once the bug is off.

    They are separate code paths -- one configuration-driven and vectorised,
    the other written to mirror the original -- so their agreement is a real
    cross-check rather than a tautology.
    """
    config = Config.from_yaml(project_root / "configs" / "legacy_exact.yaml")
    modern = analyse_scenario(config)
    legacy = reproduce_one_attack(sign_bug=False)

    for order in (0, 1, 2):
        assert modern.curves[order].peak == pytest.approx(legacy.peaks()[f"m{order}"], rel=1e-12)
        assert np.allclose(modern.curves[order].values, legacy.curves[order], rtol=1e-12)
    assert modern.baseline.peak == pytest.approx(legacy.peaks()["baseline"], rel=1e-12)


def test_the_buggy_expansion_drops_exactly_one_pairwise_term():
    """The defective e_2 equals the correct one minus its third product."""
    observations = np.array(LEGACY_ONE_ATTACK["observed_rssi_dbm"], dtype=float)
    sigma = np.array(LEGACY_ONE_ATTACK["sigma_db"], dtype=float)
    grid = legacy_grid()
    means = np.tile(grid.values, (3, 1))
    p = np.exp(log_likelihood(observations, means, sigma))

    correct_e2 = p[0] * p[1] + p[0] * p[2] + p[1] * p[2]
    assert np.allclose(np.exp(log_elementary_symmetric(np.log(p), 2)), correct_e2, rtol=1e-12)

    buggy = legacy_curves(observations, sigma, sign_bug=True).curves[1]
    missing_term = to_pmf(p[0] * p[1] + p[0] * p[2], step=grid.step)
    assert np.allclose(buggy, missing_term, rtol=1e-10)


def test_truncated_normalisation_lifts_the_attacked_curve_above_the_control(reference):
    """The artefact that looks like a result: attacked peaks higher than honest."""
    entry = reference["truncation_artefact"]
    artefact = truncation_artefact()
    assert artefact["baseline_peak"] == pytest.approx(entry["baseline_peak"], abs=5e-7)
    assert artefact["m0_peak"] == pytest.approx(entry["m0_peak"], abs=5e-7)
    assert artefact["m0_peak"] > artefact["baseline_peak"]


def test_the_unrecovered_figures_are_recorded_but_never_asserted(reference):
    """Values whose parameters were lost are provenance, not a reproduction target.

    Keeping them in the reference file without a parity assertion is the honest
    position: they were measured from the archive, and nothing in the surviving
    code regenerates them.
    """
    entry = reference["unrecovered"]
    assert "not recoverable" in entry["status"]
    assert len(entry["one_attack_bars"]) == 3
    assert len(entry["two_attack_bars"]) == 3


def test_module_constants_agree_with_the_reference_file(reference):
    """The constants shipped in the package match the archived record."""
    archived = reference["one_attack"]["archived_peaks"]
    assert LEGACY_ONE_ATTACK_PEAKS["baseline"] == pytest.approx(archived["baseline"])
    assert LEGACY_ONE_ATTACK_PEAKS["m0"] == pytest.approx(archived["m0"])
    assert LEGACY_ONE_ATTACK_PEAKS["m1_archived"] == pytest.approx(archived["m1"])
    assert LEGACY_ONE_ATTACK_PEAKS["m2"] == pytest.approx(archived["m2"])
    assert LEGACY_ONE_ATTACK_PEAKS["m1_corrected"] == pytest.approx(
        reference["one_attack"]["corrected_peaks"]["m1"]
    )


def test_the_sign_error_has_no_counterpart_outside_three_access_points():
    """Requesting the defect where it never happened is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="three-access-point"):
        legacy_curves((-70.0, -70.0, -70.0, -90.0), (5.0, 5.0, 5.0, 5.0), sign_bug=True)


def test_peaks_scale_with_the_grid_step_under_sum_normalisation():
    """Halving the step halves every reported peak, which is why it is recorded."""
    from symloc.grid import Grid

    coarse = legacy_curves(LEGACY_ONE_ATTACK["observed_rssi_dbm"], LEGACY_ONE_ATTACK["sigma_db"])
    fine = legacy_curves(
        LEGACY_ONE_ATTACK["observed_rssi_dbm"],
        LEGACY_ONE_ATTACK["sigma_db"],
        grid=Grid.from_dbm_sweep(-85.0, -60.0, 0.25),
    )
    ratio = coarse.peaks()["m0"] / fine.peaks()["m0"]
    assert ratio == pytest.approx(2.0, rel=0.02)
