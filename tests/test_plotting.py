"""Figures: that they build, and that their axes agree with their data.

The axis contract is the point of these tests.  The archived figures carried
an abscissa labelled "Reference location" and ticked 0 to 30 over data that ran
from -60 to -85 dBm, because the tick labels were forced without setting the
tick positions.  Nothing here may be able to drift that way.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from symloc.config import Config
from symloc.detection import build_dataset, cross_validate
from symloc.montecarlo import run_montecarlo
from symloc.pipeline import analyse_scenario
from symloc.plotting import (
    PALETTE,
    curve_color,
    curve_marker,
    plot_confusion_matrix,
    plot_discriminability_heatmap,
    plot_distributions,
    plot_montecarlo_intervals,
    plot_peak_bars,
    plot_peak_ratios,
    plot_roc_curve,
    plot_single_ap_likelihoods,
    rc_params,
)
from symloc.sweep import discriminability_pivot, run_sweep


@pytest.fixture
def result(config):
    return analyse_scenario(config)


def test_the_palette_cycles_and_never_runs_out():
    """A fifth curve broke the original plotting routine outright."""
    assert curve_color(0) == PALETTE[0]
    assert curve_color(len(PALETTE)) == PALETTE[0]
    assert len({curve_color(i) for i in range(len(PALETTE))}) == len(PALETTE)
    for index in range(40):
        assert curve_color(index) in PALETTE
        assert curve_marker(index)


def test_the_style_fixes_no_axis_limits_or_ticks():
    """Hard-coded ticks are what made the archived axis meaningless."""
    settings = rc_params()
    for key in settings:
        assert "xtick.major" not in key
        assert not key.endswith(("xlim", "ylim"))


def test_the_distribution_plot_uses_the_grid_axis_label(result):
    figure = plot_distributions(result)
    try:
        axes = figure.axes[0]
        assert axes.get_xlabel() == result.scenario.grid.axis_label
        assert axes.get_ylabel() == "probability mass per grid point"
    finally:
        plt.close(figure)


def test_the_plotted_x_data_is_the_grid_itself(result):
    """No rescaling, no relabelling: the line carries the grid values."""
    figure = plot_distributions(result, show_baseline=False, annotate_peaks=False)
    try:
        axes = figure.axes[0]
        for line in axes.get_lines():
            assert np.allclose(line.get_xdata(), result.scenario.grid.values)
    finally:
        plt.close(figure)


def test_the_y_label_follows_the_normalisation_mode(config):
    density = analyse_scenario(config.override({"normalize.mode": "density"}))
    figure = plot_distributions(density)
    try:
        assert "density" in figure.axes[0].get_ylabel()
    finally:
        plt.close(figure)


def test_one_line_per_curve_plus_the_control(result):
    figure = plot_distributions(result, show_baseline=True, annotate_peaks=False)
    try:
        assert len(figure.axes[0].get_lines()) == len(result.m_values) + 1
    finally:
        plt.close(figure)


def test_the_legend_names_every_curve(result):
    figure = plot_distributions(result)
    try:
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert "ESP (M=0)" in labels
        # Spelled correctly, unlike the archived legends.
        assert "Bayesian (no attacked AP)" in labels
        assert not any("Byaesian" in label for label in labels)
    finally:
        plt.close(figure)


def test_a_figure_with_many_curves_still_builds(config):
    """Eight access points means eight orders of M; nothing may be hard-coded to four."""
    aps = [{"name": f"AP{i}", "true_rssi_dbm": -70.0, "sigma_db": 8.0} for i in range(1, 9)]
    wide = analyse_scenario(
        config.override(
            {
                "scenario.aps": aps,
                "scenario.attacked_indices": (7,),
                # Empty means every valid order, so M runs 0..7 here.
                "scenario.m_values": (),
            }
        )
    )
    assert wide.m_values == tuple(range(8))
    figure = plot_distributions(wide, annotate_peaks=False)
    try:
        assert len(figure.axes[0].get_lines()) == 8 + 1
    finally:
        plt.close(figure)


def test_the_per_access_point_plot_marks_the_attacked_one(result):
    figure = plot_single_ap_likelihoods(result)
    try:
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert any("[attacked]" in label for label in labels)
    finally:
        plt.close(figure)


def test_peak_bars_print_their_values(result):
    figure = plot_peak_bars(result)
    try:
        annotations = [child.get_text() for child in figure.axes[0].texts]
        assert len(annotations) == len(result.m_values) + 1
        # The numbers are on the figure, not left in a workspace variable.
        assert any(text.startswith("0.0") for text in annotations)
    finally:
        plt.close(figure)


def test_peak_ratio_panels_show_both_measures(result):
    figure = plot_peak_ratios(result)
    try:
        assert len(figure.axes) == 2
        assert "ratio" in figure.axes[0].get_title()
        assert "difference" in figure.axes[1].get_title()
    finally:
        plt.close(figure)


def test_montecarlo_intervals_plot_every_order(config):
    summary = run_montecarlo(config).summary
    figure = plot_montecarlo_intervals(summary, metric="peak")
    try:
        axes = figure.axes[0]
        assert list(axes.get_xticks()) == list(config.scenario.m_values)
        assert "M" in axes.get_xlabel()
    finally:
        plt.close(figure)


def test_montecarlo_intervals_reject_a_missing_metric(config):
    summary = run_montecarlo(config).summary
    with pytest.raises(ValueError, match="available"):
        plot_montecarlo_intervals(summary, metric="not_a_metric")


def test_the_heatmap_labels_both_axes_from_the_table():
    config = Config().override({"sweep.attack_offsets_db": (-10.0, -20.0, -30.0)})
    table = discriminability_pivot(run_sweep(config), "js_divergence")
    figure = plot_discriminability_heatmap(table)
    try:
        axes = figure.axes[0]
        assert len(axes.get_xticks()) == table.shape[1]
        assert len(axes.get_yticks()) == table.shape[0]
        assert [t.get_text() for t in axes.get_xticklabels()] == [f"M={c}" for c in table.columns]
    finally:
        plt.close(figure)


def test_an_empty_heatmap_is_rejected():
    import pandas as pd

    with pytest.raises(ValueError, match="empty"):
        plot_discriminability_heatmap(pd.DataFrame())


def test_a_heatmap_of_an_all_nan_metric_draws_without_warning():
    """fwhm is nan wherever a curve never falls back below half its peak."""
    import warnings

    import pandas as pd

    table = pd.DataFrame(np.full((2, 3), np.nan), index=[-10.0, -20.0], columns=[0, 1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        figure = plot_discriminability_heatmap(table, metric_name="fwhm")
    try:
        assert len(figure.axes) == 2
    finally:
        plt.close(figure)


def test_the_confusion_matrix_shows_counts_and_row_shares():
    figure = plot_confusion_matrix(np.array([[8, 2], [1, 9]]))
    try:
        texts = [child.get_text() for child in figure.axes[0].texts]
        assert any("8" in text and "80.0%" in text for text in texts)
    finally:
        plt.close(figure)


def test_a_confusion_matrix_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="2x2"):
        plot_confusion_matrix(np.zeros((3, 3)))


def test_the_roc_curve_reports_its_area_and_a_chance_line(config):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from symloc.detection import build_classifier

    dataset = build_dataset(config)
    scores = cross_val_predict(
        build_classifier(config.detection.model, config.detection.seed),
        dataset.features,
        dataset.labels,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        method="predict_proba",
    )[:, 1]
    figure = plot_roc_curve(dataset.labels, scores)
    try:
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert any("AUC=" in label for label in labels)
        assert "chance" in labels
    finally:
        plt.close(figure)


def test_the_whole_figure_set_builds_for_a_pathloss_scenario(project_root):
    """A geometric grid labels its axis by reference point, not by dBm."""
    config = Config.from_yaml(project_root / "configs" / "pathloss_floor.yaml")
    outcome = analyse_scenario(config)
    figure = plot_distributions(outcome)
    try:
        assert figure.axes[0].get_xlabel() == "reference point index"
    finally:
        plt.close(figure)


def test_cross_validation_results_can_be_drawn(config):
    outcome = cross_validate(build_dataset(config), config)
    figure = plot_confusion_matrix(outcome.confusion)
    try:
        assert figure.axes[0].get_xlabel() == "predicted"
    finally:
        plt.close(figure)
