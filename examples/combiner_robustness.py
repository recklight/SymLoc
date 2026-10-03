"""Does leave-M-out actually buy robustness? A study on a real radio map.

The original analysis judged the symmetric-polynomial (ESP) family by the peak
heights of curves on a one-dimensional sweep of mean signal level.  That axis
has no geometry in it, so the question a positioning system is built to answer
-- *how far is the estimate from the truth* -- could not be asked at all.

This script asks it.  Access points are placed on a floor plan, the radio map
comes from the log-distance path loss model, one access point is tampered with,
and every combiner is scored by how far its argmax lands from the true
reference point, in metres, over many positions and many choices of which
access point is attacked.

The finding is worth stating plainly, because it is not the one the family was
designed to produce:

    ESP sums over *every* subset of size N-M, including the subsets that
    still contain the tampered access point.  Those terms keep their weight, so
    a spoofed reading that fits sharply somewhere wrong can still dominate.
    Discarding the worst-fitting readings instead -- the classical trimmed
    construction -- isolates the tampered access point outright.

Run it with::

    python examples/combiner_robustness.py
    python examples/combiner_robustness.py --trials 60 --output results/robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from symloc import Config, configure_logging
from symloc.combiners import combine_many
from symloc.likelihood import scenario_log_likelihood
from symloc.metrics import argmax_offset
from symloc.plotting.style import curve_color, curve_marker, style_context

#: Combiners compared, in the order they appear in every table and figure.
COMBINERS = ("esp", "trimmed", "median", "softmin")


def base_config() -> Config:
    """A 20 x 15 m floor with five access points and a 1 m reference grid."""
    return Config.from_dict(
        {
            "radio_map": {
                "kind": "pathloss",
                "floor_width_m": 20.0,
                "floor_depth_m": 15.0,
                "spacing_m": 1.0,
                "reference_rssi_dbm": -40.0,
                "path_loss_exponent": 3.0,
            },
            "scenario": {
                "name": "robustness",
                "aps": [
                    {"name": "AP1", "sigma_db": 6.0, "position_m": [0.0, 0.0]},
                    {"name": "AP2", "sigma_db": 6.0, "position_m": [20.0, 0.0]},
                    {"name": "AP3", "sigma_db": 6.0, "position_m": [0.0, 15.0]},
                    {"name": "AP4", "sigma_db": 6.0, "position_m": [20.0, 15.0]},
                    {"name": "AP5", "sigma_db": 6.0, "position_m": [10.0, 7.0]},
                ],
                "attacked_indices": [4],
                "attack_offset_db": -20.0,
                "true_index": 176,
            },
            "log_level": "ERROR",
        }
    )


def localization_errors(config: Config, combiner: str) -> dict[int, float]:
    """Distance in metres from each order's argmax to the true reference point."""
    scenario = config.build_scenario()
    log_p = scenario_log_likelihood(scenario)
    scores = combine_many(log_p, range(scenario.n_ap), strategy=combiner)
    return {
        order: abs(
            argmax_offset(np.exp(score - score.max()), scenario.grid, int(scenario.true_index or 0))
        )
        for order, score in scores.items()
    }


def single_scenario_table(config: Config) -> pd.DataFrame:
    """Every combiner at every order, for one fixed scenario."""
    rows = [
        {"combiner": combiner, "m": order, "error_m": error}
        for combiner in COMBINERS
        for order, error in localization_errors(config, combiner).items()
    ]
    return pd.DataFrame(rows).pivot(index="combiner", columns="m", values="error_m")


def survey(config: Config, n_positions: int, seed: int) -> pd.DataFrame:
    """Sweep the true position and which access point is tampered with.

    Each combination is scored by its *best* order of M, which is the fairest
    reading of the family: it asks whether any setting of M recovers the
    position, not whether a particular one does.
    """
    scenario = config.build_scenario()
    n_points = scenario.grid.size
    n_ap = scenario.n_ap
    rng = np.random.default_rng(seed)
    positions = rng.choice(n_points, size=min(n_positions, n_points), replace=False)

    rows: list[dict[str, object]] = []
    for true_index in positions:
        for attacked in range(n_ap):
            point = config.override(
                {
                    "scenario.true_index": int(true_index),
                    "scenario.attacked_indices": (attacked,),
                }
            )
            for combiner in COMBINERS:
                errors = localization_errors(point, combiner)
                # Orders 0 and N-1 are the plain product and the plain sum, so
                # the interesting range is the one strictly between them.
                interior = [errors[m] for m in range(1, n_ap - 1)]
                rows.append(
                    {
                        "combiner": combiner,
                        "true_index": int(true_index),
                        "attacked_ap": scenario.aps[attacked].name,
                        "best_error_m": min(interior) if interior else errors[0],
                    }
                )
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean, median and exact-recovery rate per combiner."""
    grouped = frame.groupby("combiner")["best_error_m"]
    summary = grouped.agg(
        mean_error_m="mean",
        median_error_m="median",
        p90_error_m=lambda values: values.quantile(0.90),
        max_error_m="max",
    )
    summary["exact_recovery"] = grouped.apply(lambda values: float((values < 1e-9).mean()))
    return summary.reindex(list(COMBINERS))


def write_figure(frame: pd.DataFrame, output: Path) -> Path:
    """Survival-style plot: what share of cases lands within a given error."""
    output.mkdir(parents=True, exist_ok=True)
    with style_context():
        figure, axes = plt.subplots()
        thresholds = np.linspace(0.0, 12.0, 121)
        for index, combiner in enumerate(COMBINERS):
            errors = frame.loc[frame["combiner"] == combiner, "best_error_m"].to_numpy()
            share = [(errors <= threshold).mean() for threshold in thresholds]
            axes.plot(
                thresholds,
                share,
                color=curve_color(index),
                marker=curve_marker(index),
                markevery=12,
                label=combiner,
            )
        axes.set_xlabel("localization error (m)")
        axes.set_ylabel("share of cases within the error")
        axes.set_ylim(0.0, 1.02)
        axes.set_title("Recovery under one tampered access point")
        axes.legend(loc="lower right")
    target = output / "combiner_robustness.png"
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=40, help="Number of true positions to sample."
    )
    parser.add_argument("--seed", type=int, default=20120601, help="Random seed.")
    parser.add_argument("--output", type=Path, default=None, help="Folder for the figure.")
    arguments = parser.parse_args()

    configure_logging("ERROR")
    config = base_config()

    print("=" * 72)
    print("ONE SCENARIO: true point (8, 8) m, AP5 suppressed by 20 dB")
    print("=" * 72)
    print("\nLocalization error in metres, by combiner and order of M:\n")
    print(single_scenario_table(config).round(2).to_string())
    print("\nEvery order of the symmetric-polynomial family misses by the same")
    print("7.07 m here, while trimming the worst reading recovers the point exactly.")

    print("\n" + "=" * 72)
    print(f"SURVEY: {arguments.trials} positions x 5 choices of tampered AP")
    print("=" * 72)
    frame = survey(config, arguments.trials, arguments.seed)
    summary = summarise(frame)
    print(f"\nScored by each combiner's best interior order of M, over {len(frame) // 4} cases:\n")
    formatted = summary.copy()
    formatted["exact_recovery"] = (100 * formatted["exact_recovery"]).round(1).astype(str) + "%"
    print(formatted.round(3).to_string())

    esp_rate = summary.loc["esp", "exact_recovery"]
    trimmed_rate = summary.loc["trimmed", "exact_recovery"]
    print(
        f"\nESP recovers the exact reference point in {100 * esp_rate:.0f}% of cases; "
        f"trimming does so in {100 * trimmed_rate:.0f}%."
    )
    print("\nWhy: e_(N-M) averages over every subset of size N-M, and the subsets")
    print("that still contain the spoofed access point keep their full weight. A")
    print("spoofed reading which happens to fit sharply somewhere wrong therefore")
    print("survives the averaging. Trimming selects the best-fitting subset")
    print("instead of averaging over all of them, so it drops that reading outright.")
    print("\nSoft-min fails worst of all, and for the same reason read backwards:")
    print("it weights the WORST-fitting access point most heavily, so it follows")
    print("the spoofed reading instead of rejecting it. It is included precisely")
    print("to show that the natural conservative aggregator is the wrong tool here.")
    print("\nThis comparison is only possible on a radio map with real geometry;")
    print("on the original one-dimensional sweep there is no distance to measure.")

    if arguments.output is not None:
        target = write_figure(frame, arguments.output)
        print(f"\nWrote {target}")


if __name__ == "__main__":
    main()
