"""End-to-end demonstration on entirely synthetic data.

Runs the whole project on numbers it generates itself -- no data file, no radio
map and no download -- and writes its figures and tables to a temporary folder:

1. build a floor plan and place five access points on it;
2. derive a radio map from the log-distance path loss model;
3. score one tampered observation vector at every order of M;
4. resample the observation noise and report metrics with confidence intervals;
5. generate a labelled spoofing-detection set and cross-validate a classifier;
6. write every figure, table and manifest.

Run it with::

    python examples/end_to_end_synthetic.py
    python examples/end_to_end_synthetic.py --output results/demo
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from symloc import (
    Config,
    analyse_scenario,
    build_dataset,
    configure_logging,
    cross_validate,
    curves_to_frame,
    evidence_collapse,
    run_montecarlo,
    run_sweep,
)
from symloc.io_ import prepare_run, save_figure, write_manifest, write_table
from symloc.plotting import (
    plot_confusion_matrix,
    plot_discriminability_heatmap,
    plot_distributions,
    plot_montecarlo_intervals,
    plot_peak_bars,
    plot_single_ap_likelihoods,
)
from symloc.sweep import discriminability_pivot

_LOGGER = logging.getLogger("example")


def build_config(output_dir: Path) -> Config:
    """A five-access-point floor plan with one tampered access point.

    Every value is written out here rather than read from a file, so this
    script has no inputs at all.
    """
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
                "name": "synthetic_floor",
                "aps": [
                    {"name": "AP1", "sigma_db": 6.0, "position_m": [0.0, 0.0]},
                    {"name": "AP2", "sigma_db": 6.0, "position_m": [20.0, 0.0]},
                    {"name": "AP3", "sigma_db": 6.0, "position_m": [0.0, 15.0]},
                    {"name": "AP4", "sigma_db": 6.0, "position_m": [20.0, 15.0]},
                    {"name": "AP5", "sigma_db": 6.0, "position_m": [10.0, 7.0]},
                ],
                "attacked_indices": [4],
                "attack_offset_db": -20.0,
                "combiner": "esp",
                "true_index": 176,
            },
            "montecarlo": {"n_trials": 120, "seed": 20120601},
            "detection": {
                "model": "logistic",
                "n_splits": 5,
                "n_trials_per_class": 300,
                "seed": 20120601,
                "attack_offsets_db": [-30.0, -25.0, -20.0, -15.0, -10.0],
            },
            "sweep": {
                "attack_offsets_db": [-5.0, -10.0, -20.0, -30.0, -40.0],
                "combiners": ["esp"],
            },
            "output": {"results_dir": str(output_dir), "figure_formats": ["png"], "dpi": 150},
            "log_level": "INFO",
        }
    )


def main() -> None:
    """Run every stage and report what it found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder for the figures and tables (default: a temporary folder).",
    )
    arguments = parser.parse_args()

    holder: tempfile.TemporaryDirectory | None = None
    if arguments.output is None:
        holder = tempfile.TemporaryDirectory(prefix="symloc-example-")
        output_dir = Path(holder.name)
    else:
        output_dir = arguments.output
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = build_config(output_dir)
        configure_logging(config.log_level)
        paths = prepare_run(config, "example")

        # -- 1. one scenario ------------------------------------------------
        result = analyse_scenario(config)
        print(f"\nScenario: {result.scenario.name}")
        print(f"  {result.scenario.n_ap} access points, {result.scenario.n_attacked} tampered")
        print(f"  radio map degenerate: {result.scenario.radio_map.is_degenerate}")
        print("\n  peak of each curve:")
        for order, peak in sorted(result.peaks().items()):
            print(f"    M={order}: {peak:.6f}")

        collapse = evidence_collapse(result)
        print("\n  change in absolute evidence vs the honest control (nats):")
        for order in result.m_values:
            gap = collapse[f"log_evidence_gap_m{order}"]
            print(f"    M={order}: {gap:+.3f}")
        print("  Negative means the tampering made the readings fit the map worse")
        print("  overall. On a geometric map the sign can go either way at high M,")
        print("  because a spoofed reading still fits well somewhere else on the")
        print("  floor. Either way, per-curve normalisation erases this entirely.")

        localization = result.metrics.query("metric == 'localization_error'")
        if not localization.empty:
            print("\n  localization error under attack (metres):")
            for _, row in localization.iterrows():
                print(f"    M={int(row['m'])}: {row['value']:.2f}")

        # -- 2. repeated trials ---------------------------------------------
        montecarlo = run_montecarlo(config)
        peaks = montecarlo.summary.query("metric == 'peak'")
        print(f"\nMonte-Carlo over {montecarlo.n_trials} trials (seed {montecarlo.seed}):")
        for _, row in peaks.iterrows():
            print(
                f"    M={int(row['m'])}: peak {row['mean']:.6f} "
                f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]"
            )
        print("  The original analysis was a single deterministic evaluation,")
        print("  so it could not state any interval at all.")

        # -- 3. spoofing detection ------------------------------------------
        dataset = build_dataset(config)
        cross_validated = cross_validate(dataset, config)
        print(f"\nSpoofing detection on {dataset.n_samples} synthetic trials:")
        print(f"    {cross_validated.report()}")
        print("  confusion (rows true, columns predicted):")
        print(f"    {cross_validated.confusion.tolist()}")

        # -- 4. parameter sweep ---------------------------------------------
        swept = run_sweep(config)
        pivot = discriminability_pivot(swept, "js_divergence")
        print("\nJensen-Shannon divergence from the honest control:")
        print(pivot.round(4).to_string())

        # -- 5. write everything --------------------------------------------
        write_table(result.metrics, paths.tables / "metrics", config.output.write_parquet)
        write_table(curves_to_frame(result), paths.tables / "curves", config.output.write_parquet)
        write_table(montecarlo.summary, paths.tables / "montecarlo", config.output.write_parquet)
        write_table(
            cross_validated.per_fold, paths.tables / "cv_folds", config.output.write_parquet
        )
        write_table(swept, paths.tables / "sweep", config.output.write_parquet)
        config.save(paths.root / "config.yaml")
        write_manifest(
            paths.root / "manifest.json",
            config,
            extra={
                **result.notes,
                "evidence_collapse": collapse,
                "detection_summary": cross_validated.summary,
            },
        )

        figures = {
            "distributions": plot_distributions(result),
            "single_ap_likelihoods": plot_single_ap_likelihoods(result),
            "peak_bars": plot_peak_bars(result),
            "montecarlo_peak": plot_montecarlo_intervals(montecarlo.summary, metric="peak"),
            "confusion_matrix": plot_confusion_matrix(cross_validated.confusion),
            "heatmap_js": plot_discriminability_heatmap(pivot, "Jensen-Shannon divergence"),
        }
        for name, figure in figures.items():
            save_figure(figure, paths.figures / name, config)
            plt.close(figure)

        print(f"\nWrote {len(figures)} figures and 5 tables to:\n  {paths.root}")
        if holder is not None:
            print("  (a temporary folder; pass --output to keep the results)")
    finally:
        if holder is not None:
            holder.cleanup()


if __name__ == "__main__":
    main()
