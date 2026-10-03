"""Reproduce the archived result, then show what the corrected version gives.

Three defects are demonstrated side by side, each with a number attached:

1. **A sign error** in one of the three pairwise products of :math:`e_2`, which
   silently dropped that term. The archived curve's peak is 7.6% too high, and
   that peak is the quantity the discriminability argument rests on.
2. **Truncated normalisation**, which lets an attacked curve peak *above* the
   unattacked control -- an artefact of dividing by a smaller total, not a
   result.
3. **Grid-dependent peaks**, where halving the candidate spacing halves every
   reported peak, with nothing in the archive recording which spacing was used.

Everything is computed from parameters recovered by solving for the settings
that reproduce the archived curve peaks; no archived file is read or needed.

Run it with::

    python examples/reproduce_legacy_and_fix.py
    python examples/reproduce_legacy_and_fix.py --output results/legacy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from symloc import configure_logging
from symloc.grid import Grid
from symloc.legacy import (
    LEGACY_ONE_ATTACK,
    LEGACY_ONE_ATTACK_PEAKS,
    LEGACY_UNRECOVERED_BARS,
    legacy_curves,
    reproduce_one_attack,
    sign_bug_impact,
    truncation_artefact,
)
from symloc.plotting.style import BASELINE_COLOR, curve_color, style_context


def report_sign_error() -> None:
    """Print the archived and corrected values side by side."""
    reproduced = reproduce_one_attack(sign_bug=True)
    corrected = reproduce_one_attack(sign_bug=False)
    impact = sign_bug_impact()

    print("=" * 72)
    print("1. THE SIGN ERROR")
    print("=" * 72)
    print(
        "\nRecovered parameters: observations "
        f"{list(LEGACY_ONE_ATTACK['observed_rssi_dbm'])} dBm, "
        f"sigma {list(LEGACY_ONE_ATTACK['sigma_db'])} dB."
    )
    print("The archived filename claimed the tampered AP reported -99 dBm;")
    print("solving for the curve peaks gives -90 dBm instead.\n")

    print(f"{'curve':<12}{'archived':>12}{'reproduced':>14}{'corrected':>14}")
    print("-" * 52)
    reproduced_peaks = reproduced.peaks()
    corrected_peaks = corrected.peaks()
    for key, archived_key in (
        ("baseline", "baseline"),
        ("m0", "m0"),
        ("m1", "m1_archived"),
        ("m2", "m2"),
    ):
        archived = LEGACY_ONE_ATTACK_PEAKS[archived_key]
        print(
            f"{key:<12}{archived:>12.7f}{reproduced_peaks[key]:>14.7f}{corrected_peaks[key]:>14.7f}"
        )

    print(
        f"\nThe M=1 curve as archived peaks at {impact['archived_peak']:.7f}; "
        f"the mathematics gives {impact['corrected_peak']:.7f}."
    )
    print(f"The archived value is {100 * impact['relative_error']:.2f}% too high.\n")
    print(f"{'ratio':<12}{'as archived':>16}{'corrected':>14}")
    print("-" * 42)
    print(
        f"{'rho(0,1)':<12}{impact['archived_rho_0_1']:>16.5f}{impact['corrected_rho_0_1']:>14.5f}"
    )
    print(
        f"{'rho(1,2)':<12}{impact['archived_rho_1_2']:>16.5f}{impact['corrected_rho_1_2']:>14.5f}"
    )
    print("\nBoth discriminability numbers move, so any claim resting on them")
    print("is resting on the defect rather than on the model.")


def report_truncation() -> None:
    """Print the artefact that a truncated normalisation manufactures."""
    artefact = truncation_artefact()
    print("\n" + "=" * 72)
    print("2. TRUNCATED NORMALISATION")
    print("=" * 72)
    print("\nWith the parameters left in the surviving script:")
    print(f"  unattacked control peak : {artefact['baseline_peak']:.7f}")
    print(f"  attacked M=0 peak       : {artefact['m0_peak']:.7f}")
    print(
        f"\nThe attacked curve peaks {100 * (artefact['ratio'] - 1):.2f}% HIGHER than the control."
    )
    print("Nothing about the attack improved the fix. The tampered access point")
    print("lost more of its likelihood mass off the edge of the grid, so dividing")
    print("by a smaller total raised whatever was left.")


def report_grid_dependence() -> None:
    """Print how the reported peaks move with the candidate spacing."""
    print("\n" + "=" * 72)
    print("3. PEAKS DEPEND ON THE GRID STEP")
    print("=" * 72)
    print("\nThe same scenario, evaluated at three candidate spacings:\n")
    print(f"{'step (dB)':>12}{'points':>10}{'M=0 peak':>14}")
    print("-" * 36)
    for step in (1.0, 0.5, 0.25):
        curves = legacy_curves(
            LEGACY_ONE_ATTACK["observed_rssi_dbm"],
            LEGACY_ONE_ATTACK["sigma_db"],
            grid=Grid.from_dbm_sweep(-85.0, -60.0, step),
        )
        grid = curves.grid
        print(f"{step:>12.2f}{grid.size:>10}{curves.peaks()['m0']:>14.7f}")
    print("\nHalving the step halves every peak, because dividing by the sum")
    print("gives a mass function rather than a density. The archive records no")
    print("step anywhere, so its peak values cannot be interpreted on their own.")


def report_unrecoverable() -> None:
    """Print the values whose parameters were never recorded."""
    print("\n" + "=" * 72)
    print("4. WHAT COULD NOT BE RECOVERED")
    print("=" * 72)
    print("\nThese peak heights were measured from archived bar figures:")
    print(f"  one attack : {list(LEGACY_UNRECOVERED_BARS['one_attack'])}")
    print(f"  two attacks: {list(LEGACY_UNRECOVERED_BARS['two_attack'])}")
    print("\nA search over access point counts, observations and standard")
    print("deviations gets no closer than about 7e-3 to them, against 1.2e-7 for")
    print("the one-attack figure. Without a run manifest they cannot be")
    print("regenerated at all -- which is why every run of this package writes one.")


def write_figure(output: Path) -> Path:
    """Draw the archived and corrected M=1 curves on one pair of axes."""
    reproduced = reproduce_one_attack(sign_bug=True)
    corrected = reproduce_one_attack(sign_bug=False)
    grid = reproduced.grid

    output.mkdir(parents=True, exist_ok=True)
    with style_context():
        figure, axes = plt.subplots()
        axes.plot(
            grid.values,
            reproduced.curves[1],
            color=curve_color(1),
            marker="o",
            markevery=6,
            label=f"M=1 as archived ({reproduced.peaks()['m1']:.5f})",
        )
        axes.plot(
            grid.values,
            corrected.curves[1],
            color=curve_color(2),
            linestyle="--",
            marker="s",
            markevery=6,
            label=f"M=1 corrected ({corrected.peaks()['m1']:.5f})",
        )
        axes.plot(
            grid.values,
            reproduced.baseline,
            color=BASELINE_COLOR,
            linestyle=(0, (4, 2)),
            linewidth=1.2,
            label="no attacked AP",
        )
        difference = float(np.max(reproduced.curves[1]) / np.max(corrected.curves[1]) - 1.0)
        axes.set_xlabel(grid.axis_label)
        axes.set_ylabel("probability mass per grid point")
        axes.set_title(f"The sign error inflates the archived peak by {100 * difference:.2f}%")
        axes.legend(loc="best")
        figure.text(
            0.01,
            -0.04,
            f"peaks are pmf masses on a {grid.step:g} dBm grid ({grid.size} points)",
            fontsize=7,
            color="#555555",
        )
    target = output / "sign_error_comparison.png"
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=None, help="Folder for the comparison figure."
    )
    arguments = parser.parse_args()

    # The legacy path warns loudly when it reproduces the defect on purpose.
    configure_logging("ERROR")

    report_sign_error()
    report_truncation()
    report_grid_dependence()
    report_unrecoverable()

    if arguments.output is not None:
        target = write_figure(arguments.output)
        print(f"\nWrote {target}")


if __name__ == "__main__":
    main()
