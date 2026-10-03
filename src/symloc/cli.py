"""Command-line interface.

The original analysis had no interface at all: the access point count, the
observations, the standard deviations and the attack scenario were bare
constants at the top of a script, so changing any of them meant editing source.
Every one of them is a configuration entry here, and every command accepts
``--set key=value`` for one-off overrides.

Commands
--------
``run``
    Analyse one scenario and write its curves, metrics, figures and manifest.
``sweep``
    Evaluate a grid of parameters and write the long-format table and heatmap.
``figure``
    Regenerate the figures of a configuration without recomputing the tables.
``reproduce-legacy``
    Reproduce the archived result, with and without the original sign error,
    and report the difference.
``inspect-fig``
    Read a MATLAB figure the user supplies and print the curves inside it.
``prepare``
    Generate the labelled spoofing-detection dataset.
``train``
    Fit the detector and save it.
``evaluate``
    Cross-validate the detector and report UAR, sensitivity, specificity, the
    confusion matrix and ROC-AUC.
``predict``
    Score one observation vector with a saved detector.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

from .config import Config, ConfigError
from .logging_utils import configure_logging

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Probability-distribution analysis of attack-resistant fingerprint likelihoods.",
)


def _load_config(config_path: Path | None, overrides: list[str] | None) -> Config:
    """Load a configuration and apply ``key=value`` overrides from the command line."""
    config = Config.from_yaml(config_path) if config_path is not None else Config()
    if overrides:
        parsed: dict[str, Any] = {}
        for item in overrides:
            if "=" not in item:
                raise typer.BadParameter(
                    f"--set expects key=value, got {item!r}", param_hint="--set"
                )
            key, _, raw = item.partition("=")
            parsed[key.strip()] = _coerce(raw.strip())
        config = config.override(parsed)
    configure_logging(config.log_level)
    return config


def _coerce(raw: str) -> Any:
    """Turn a command-line string into the value it evidently means."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


ConfigOption = typer.Option(None, "--config", "-c", help="YAML configuration file.")
SetOption = typer.Option(None, "--set", "-s", help="Override a setting: key=value.")


@app.command()
def run(
    config: Path = ConfigOption,
    set_: list[str] = SetOption,
    label: str = typer.Option("run", "--label", help="Prefix of the run folder name."),
    figures: bool = typer.Option(True, "--figures/--no-figures", help="Write figures."),
) -> None:
    """Analyse one scenario end to end and write every artefact."""
    from .io_ import prepare_run, save_figure, write_manifest, write_table
    from .pipeline import analyse_scenario, curves_to_frame, evidence_collapse

    settings = _load_config(config, set_)
    result = analyse_scenario(settings)
    paths = prepare_run(settings, label)

    write_table(result.metrics, paths.tables / "metrics", settings.output.write_parquet)
    write_table(curves_to_frame(result), paths.tables / "curves", settings.output.write_parquet)
    settings.save(paths.root / "config.yaml")

    collapse = evidence_collapse(result)
    write_manifest(
        paths.root / "manifest.json",
        settings,
        extra={
            **result.notes,
            "adjacent_peak_ratios": result.adjacent_peak_ratios(),
            "adjacent_peak_differences": result.adjacent_peak_differences(),
            "evidence_collapse": collapse,
        },
    )

    if figures:
        import matplotlib.pyplot as plt

        from .plotting import (
            plot_distributions,
            plot_peak_bars,
            plot_peak_ratios,
            plot_single_ap_likelihoods,
        )

        for name, builder in (
            ("distributions", plot_distributions),
            ("peak_bars", plot_peak_bars),
            ("peak_ratios", plot_peak_ratios),
            ("single_ap_likelihoods", plot_single_ap_likelihoods),
        ):
            figure = builder(result)
            save_figure(figure, paths.figures / name, settings)
            plt.close(figure)

    typer.echo(f"scenario   : {result.scenario.name}")
    typer.echo(f"peaks      : {_format_mapping(result.peaks())}")
    typer.echo(f"peak ratios: {_format_mapping(result.adjacent_peak_ratios())}")
    typer.echo(
        "evidence   : "
        + _format_mapping({k: v for k, v in collapse.items() if k.startswith("log_evidence")})
        + "  (nats below the honest control)"
    )
    typer.echo(f"written to : {paths.root}")


@app.command()
def sweep(
    config: Path = ConfigOption,
    set_: list[str] = SetOption,
    metric: str = typer.Option("js_divergence", "--metric", help="Metric for the heatmap."),
    label: str = typer.Option("sweep", "--label", help="Prefix of the run folder name."),
) -> None:
    """Evaluate a grid of parameters and write the long-format table."""
    from .io_ import prepare_run, save_figure, write_manifest, write_table
    from .sweep import discriminability_pivot, run_sweep

    settings = _load_config(config, set_)
    frame = run_sweep(settings)
    paths = prepare_run(settings, label)

    write_table(frame, paths.tables / "sweep", settings.output.write_parquet)
    settings.save(paths.root / "config.yaml")
    write_manifest(paths.root / "manifest.json", settings, extra={"n_rows": int(frame.shape[0])})

    try:
        table = discriminability_pivot(frame, metric)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--metric") from exc

    write_table(
        table.reset_index(), paths.tables / f"pivot_{metric}", settings.output.write_parquet
    )

    import matplotlib.pyplot as plt

    from .plotting import plot_discriminability_heatmap

    figure = plot_discriminability_heatmap(table, metric_name=metric.replace("_", " "))
    save_figure(figure, paths.figures / f"heatmap_{metric}", settings)
    plt.close(figure)

    typer.echo(f"rows       : {frame.shape[0]}")
    typer.echo(f"metric     : {metric}")
    typer.echo(f"written to : {paths.root}")


@app.command()
def figure(
    config: Path = ConfigOption,
    set_: list[str] = SetOption,
    label: str = typer.Option("figure", "--label", help="Prefix of the run folder name."),
) -> None:
    """Regenerate the figures of a configuration, without the tables."""
    import matplotlib.pyplot as plt

    from .io_ import prepare_run, save_figure
    from .pipeline import analyse_scenario
    from .plotting import (
        plot_distributions,
        plot_peak_bars,
        plot_peak_ratios,
        plot_single_ap_likelihoods,
    )

    settings = _load_config(config, set_)
    result = analyse_scenario(settings)
    paths = prepare_run(settings, label)
    settings.save(paths.root / "config.yaml")

    written: list[Path] = []
    for name, builder in (
        ("distributions", plot_distributions),
        ("peak_bars", plot_peak_bars),
        ("peak_ratios", plot_peak_ratios),
        ("single_ap_likelihoods", plot_single_ap_likelihoods),
    ):
        fig = builder(result)
        written.extend(save_figure(fig, paths.figures / name, settings))
        plt.close(fig)

    for path in written:
        typer.echo(str(path))


@app.command("reproduce-legacy")
def reproduce_legacy(
    output: Path = typer.Option(None, "--output", "-o", help="Folder for the comparison figure."),
) -> None:
    """Reproduce the archived result and quantify the original sign error."""
    from .legacy import (
        LEGACY_ONE_ATTACK_PEAKS,
        reproduce_one_attack,
        sign_bug_impact,
        truncation_artefact,
    )

    configure_logging("WARNING")
    reproduced = reproduce_one_attack(sign_bug=True)
    corrected = reproduce_one_attack(sign_bug=False)
    impact = sign_bug_impact(reproduced, corrected)
    artefact = truncation_artefact()

    typer.echo("Archived figure, reproduced from the recovered parameters:")
    for key, value in reproduced.peaks().items():
        expected = LEGACY_ONE_ATTACK_PEAKS.get(
            "m1_archived" if key == "m1" else key.replace("m1_corrected", "m1")
        )
        suffix = f"   (archived {expected:.7g})" if expected is not None else ""
        typer.echo(f"  {key:<10} {value:.7g}{suffix}")

    typer.echo("")
    typer.echo("Effect of the sign error on the archived M=1 curve:")
    typer.echo(f"  as archived  : {impact['archived_peak']:.7g}")
    typer.echo(f"  corrected    : {impact['corrected_peak']:.7g}")
    typer.echo(f"  overstated by: {100 * impact['relative_error']:.2f}%")
    typer.echo(
        f"  rho(0,1)     : {impact['archived_rho_0_1']:.5f} as archived, "
        f"{impact['corrected_rho_0_1']:.5f} corrected"
    )
    typer.echo(
        f"  rho(1,2)     : {impact['archived_rho_1_2']:.5f} as archived, "
        f"{impact['corrected_rho_1_2']:.5f} corrected"
    )

    typer.echo("")
    typer.echo("Truncated-normalisation artefact (parameters left in the original script):")
    typer.echo(f"  unattacked control peak : {artefact['baseline_peak']:.6g}")
    typer.echo(f"  attacked M=0 peak       : {artefact['m0_peak']:.6g}")
    typer.echo(
        f"  the attacked curve peaks {100 * (artefact['ratio'] - 1):.2f}% HIGHER than the "
        "control, purely because more of its mass fell off the grid edge"
    )

    if output is not None:
        import matplotlib.pyplot as plt

        from .plotting.style import curve_color, style_context

        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        with style_context():
            fig, axes = plt.subplots()
            grid = reproduced.grid
            axes.plot(
                grid.values,
                reproduced.curves[1],
                color=curve_color(1),
                label="M=1 as archived (sign error)",
            )
            axes.plot(
                grid.values,
                corrected.curves[1],
                color=curve_color(2),
                linestyle="--",
                label="M=1 corrected",
            )
            axes.set_xlabel(grid.axis_label)
            axes.set_ylabel("probability mass per grid point")
            axes.set_title("Effect of the sign error on the archived curve")
            axes.legend(loc="best")
        target = folder / "sign_bug_comparison.png"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        plt.close(fig)
        typer.echo(f"\nwrote {target}")


@app.command("inspect-fig")
def inspect_fig(
    path: Path = typer.Argument(..., help="A MATLAB .fig file you supply."),
) -> None:
    """Print the curves stored inside an archived MATLAB figure."""
    from .figio import FigFormatError, read_fig

    configure_logging("WARNING")
    try:
        document = read_fig(path)
    except FigFormatError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(document.summary())


@app.command()
def prepare(
    config: Path = ConfigOption,
    set_: list[str] = SetOption,
    label: str = typer.Option("dataset", "--label", help="Prefix of the run folder name."),
) -> None:
    """Generate the labelled spoofing-detection dataset from synthetic trials."""
    from .detection import build_dataset
    from .io_ import prepare_run, write_manifest, write_table

    settings = _load_config(config, set_)
    dataset = build_dataset(settings)
    paths = prepare_run(settings, label)

    write_table(dataset.to_frame(), paths.tables / "dataset", settings.output.write_parquet)
    settings.save(paths.root / "config.yaml")
    write_manifest(
        paths.root / "manifest.json",
        settings,
        extra={
            "n_samples": dataset.n_samples,
            "n_features": len(dataset.feature_names),
            "feature_names": list(dataset.feature_names),
        },
    )

    typer.echo(f"samples    : {dataset.n_samples} ({int(dataset.labels.sum())} attacked)")
    typer.echo(f"features   : {len(dataset.feature_names)}")
    typer.echo(f"written to : {paths.root}")


@app.command()
def train(
    config: Path = ConfigOption,
    set_: list[str] = SetOption,
    output: Path = typer.Option(
        Path("results/detector.joblib"), "--output", "-o", help="Where to save the model."
    ),
) -> None:
    """Fit the spoofing detector on freshly generated trials and save it."""
    import joblib

    from .detection import build_dataset, fit_detector

    settings = _load_config(config, set_)
    dataset = build_dataset(settings)
    model = fit_detector(dataset, settings)

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": dataset.feature_names,
            "config": settings.to_dict(),
        },
        destination,
    )
    typer.echo(f"model      : {settings.detection.model}")
    typer.echo(f"samples    : {dataset.n_samples}")
    typer.echo(f"written to : {destination}")


@app.command()
def evaluate(
    config: Path = ConfigOption,
    set_: list[str] = SetOption,
    label: str = typer.Option("evaluate", "--label", help="Prefix of the run folder name."),
    figures: bool = typer.Option(True, "--figures/--no-figures", help="Write figures."),
) -> None:
    """Cross-validate the detector and report the full classification metrics."""
    import numpy as np

    from .detection import build_dataset, cross_validate
    from .io_ import prepare_run, save_figure, write_manifest, write_table

    settings = _load_config(config, set_)
    dataset = build_dataset(settings)
    result = cross_validate(dataset, settings)
    paths = prepare_run(settings, label)

    write_table(result.per_fold, paths.tables / "cv_folds", settings.output.write_parquet)
    settings.save(paths.root / "config.yaml")
    write_manifest(
        paths.root / "manifest.json",
        settings,
        extra={
            "summary": result.summary,
            "confusion": result.confusion.tolist(),
            "feature_names": list(result.feature_names),
        },
    )

    if figures:
        import matplotlib.pyplot as plt
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        from .detection import build_classifier
        from .plotting import plot_confusion_matrix, plot_roc_curve

        fig = plot_confusion_matrix(result.confusion)
        save_figure(fig, paths.figures / "confusion_matrix", settings)
        plt.close(fig)

        splitter = StratifiedKFold(
            n_splits=settings.detection.n_splits, shuffle=True, random_state=settings.detection.seed
        )
        scores = cross_val_predict(
            build_classifier(settings.detection.model, settings.detection.seed),
            dataset.features,
            dataset.labels,
            cv=splitter,
            method="predict_proba",
        )[:, 1]
        fig = plot_roc_curve(dataset.labels, scores, label=settings.detection.model)
        save_figure(fig, paths.figures / "roc_curve", settings)
        plt.close(fig)

    typer.echo(result.report())
    typer.echo("confusion (rows true, columns predicted):")
    typer.echo(f"  {np.array2string(result.confusion)}")
    typer.echo(f"written to : {paths.root}")


@app.command()
def predict(
    rssi: str = typer.Argument(
        ...,
        help=(
            "Comma-separated observed RSSI, one per AP. A received Wi-Fi level is "
            "always negative, so magnitudes are accepted and negated: '70,70,90' "
            "means -70, -70, -90 dBm. To pass the signs explicitly, put the "
            "readings after a '--' separator, with every option before it, since "
            "'--' makes the rest positional: "
            "symloc predict --model results/detector.joblib -- -70,-70,-90"
        ),
    ),
    model_path: Path = typer.Option(
        Path("results/detector.joblib"), "--model", "-m", help="A model saved by 'train'."
    ),
    config: Path = ConfigOption,
    set_: list[str] = SetOption,
) -> None:
    """Score one observation vector with a saved detector."""
    import joblib
    import numpy as np

    from .detection import predict_observation

    settings = _load_config(config, set_)
    try:
        values = np.array([float(part) for part in rssi.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise typer.BadParameter(
            f"could not read {rssi!r} as comma-separated numbers", param_hint="RSSI"
        ) from exc
    if values.size and np.all(values > 0):
        # A positive received level in dBm would be a kilowatt at the antenna, so
        # the user has given magnitudes; a leading minus makes the shell treat
        # the argument as an option, which is why magnitudes are the friendly form.
        _LOGGER.info("reading %s as negative dBm", values.tolist())
        values = -values
    positive = np.flatnonzero(values > 0).tolist()
    if positive:
        # Mixed signs are neither convention, and taking them at face value would
        # score a physically impossible reading without ever saying so.
        raise typer.BadParameter(
            f"reading(s) at position(s) {positive} are positive dBm, which no received "
            "Wi-Fi level can be; give every reading with the same convention -- all "
            "magnitudes ('70,70,90') or all signed (after a '--' separator)",
            param_hint="RSSI",
        )

    source = Path(model_path)
    if not source.is_file():
        typer.echo(
            f"error: no model at {source}; run 'symloc train --output {source}' first", err=True
        )
        raise typer.Exit(code=1)
    payload = joblib.load(source)

    outcome = predict_observation(payload["model"], values, settings)
    verdict = "ATTACKED" if outcome["attacked"] >= 0.5 else "honest"
    typer.echo(f"readings   : {values.tolist()} dBm")
    typer.echo(f"verdict    : {verdict} (p={outcome['probability']:.3f})")
    typer.echo(
        f"position   : {outcome['estimated_position']:.2f} "
        f"(grid index {int(outcome['estimated_index'])})"
    )


def _format_mapping(values: dict[str, float]) -> str:
    """Render a small mapping compactly for terminal output."""
    return ", ".join(f"{key}={value:.6g}" for key, value in values.items())


def main() -> None:
    """Entry point for the ``symloc`` console script."""
    try:
        app()
    except ConfigError as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
