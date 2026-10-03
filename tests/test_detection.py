"""The spoofing detector: features, cross-validation and the metrics reported.

Everything here runs on synthetic trials drawn from the model in
:mod:`symloc.pathloss`; no measurement is read from disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from symloc.detection import (
    DetectionError,
    build_classifier,
    build_dataset,
    classification_metrics,
    cross_validate,
    extract_features,
    feature_names,
    fit_detector,
    predict_observation,
)


def test_classification_metrics_match_a_hand_computed_confusion_matrix():
    """Worked by hand: 2 honest and 2 attacked, one honest misclassified."""
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["tn"] == 1 and metrics["fp"] == 1
    assert metrics["fn"] == 0 and metrics["tp"] == 2
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["uar"] == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["ppv"] == pytest.approx(2 / 3)
    assert metrics["npv"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)


def test_uar_is_unmoved_by_class_imbalance():
    """The reason UAR is the headline: a constant answer scores 0.5, not 0.9."""
    y_true = [0] * 90 + [1] * 10
    always_honest = [0] * 100
    metrics = classification_metrics(y_true, always_honest)
    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["uar"] == pytest.approx(0.5)


def test_uar_is_undefined_when_a_class_is_missing_from_the_truth():
    """Averaging over the one class that exists would score the degenerate case 1.0."""
    metrics = classification_metrics([0, 0, 0], [0, 0, 0])
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert np.isnan(metrics["uar"])
    assert np.isnan(metrics["sensitivity"])
    assert metrics["specificity"] == pytest.approx(1.0)


def test_roc_auc_is_reported_when_scores_are_given():
    metrics = classification_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["roc_auc"] == pytest.approx(1.0)
    without = classification_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert np.isnan(without["roc_auc"])


def test_mismatched_label_arrays_are_rejected():
    with pytest.raises(DetectionError, match="shape"):
        classification_metrics([0, 1], [0, 1, 1])


def test_feature_names_match_the_extracted_width(grid, rng):
    for n_ap, orders in ((3, (0, 1, 2)), (5, (0, 2, 4))):
        block = rng.normal(size=(4, n_ap, grid.size))
        features = extract_features(block, grid, orders)
        assert features.shape == (4, len(feature_names(orders)))


def test_extract_features_rejects_the_wrong_rank(grid):
    with pytest.raises(DetectionError, match="3-D"):
        extract_features(np.zeros((3, 4)), grid, (0,))


def test_the_dataset_is_balanced_and_labelled(config):
    data = build_dataset(config)
    assert data.n_samples == 2 * config.detection.n_trials_per_class
    assert int(data.labels.sum()) == config.detection.n_trials_per_class
    assert data.features.shape[1] == len(data.feature_names)
    assert np.all(np.isfinite(data.features))


def test_honest_samples_carry_a_zero_attack_offset(config):
    data = build_dataset(config)
    assert np.all(data.offsets_db[data.labels == 0] == 0.0)
    assert np.all(data.offsets_db[data.labels == 1] != 0.0)


def test_the_feature_map_never_sees_the_label(config):
    """A detector that could read the label would report a meaningless score.

    Only the readings and the radio map go into the features, so this checks
    the obvious way that could go wrong: identical readings must give identical
    features regardless of how they were labelled.
    """
    data = build_dataset(config)
    scenario = config.build_scenario()
    from symloc.likelihood import log_likelihood_for_observations

    log_p = log_likelihood_for_observations(
        data.observations, scenario.radio_map, scenario.sigma_db
    )
    recomputed = extract_features(
        log_p,
        scenario.grid,
        config.scenario.m_values,
        combiner=config.scenario.combiner,
        mode=config.normalize.mode,
    )
    assert np.allclose(recomputed, data.features, rtol=1e-12)


def test_the_dataset_is_reproducible_from_the_seed(config):
    first = build_dataset(config)
    second = build_dataset(config)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)

    different = build_dataset(config.override({"detection.seed": 4242}))
    assert not np.array_equal(first.features, different.features)


def test_a_scenario_with_no_attacked_access_point_cannot_produce_the_class(config):
    with pytest.raises(DetectionError, match="no attacked access point"):
        build_dataset(config.override({"scenario.attacked_indices": ()}))


def test_cross_validation_uses_every_fold_and_is_stratified(config):
    result = cross_validate(build_dataset(config), config)
    assert result.per_fold.shape[0] == config.detection.n_splits
    # Stratification: each fold's test set holds both classes, so every fold
    # reports a finite sensitivity and specificity.
    assert result.per_fold["sensitivity"].notna().all()
    assert result.per_fold["specificity"].notna().all()
    assert result.confusion.sum() == build_dataset(config).n_samples


def test_cross_validation_is_reproducible(config):
    first = cross_validate(build_dataset(config), config)
    second = cross_validate(build_dataset(config), config)
    assert first.summary["uar_mean"] == pytest.approx(second.summary["uar_mean"])
    assert np.array_equal(first.confusion, second.confusion)


def test_the_detector_beats_chance_on_a_clearly_separable_attack(config):
    """A 45 dB attack is far outside the shadowing noise and must be detectable."""
    settings = config.override(
        {"detection.attack_offsets_db": (-45.0,), "detection.n_trials_per_class": 120}
    )
    result = cross_validate(build_dataset(settings), settings)
    assert result.summary["uar_mean"] > 0.9
    assert result.summary["roc_auc_mean"] > 0.95


def test_detection_gets_harder_as_the_attack_shrinks(config):
    """An attack well inside the noise is undetectable, and the metrics say so."""
    strong = config.override(
        {"detection.attack_offsets_db": (-45.0,), "detection.n_trials_per_class": 120}
    )
    weak = config.override(
        {"detection.attack_offsets_db": (-1.0,), "detection.n_trials_per_class": 120}
    )
    strong_uar = cross_validate(build_dataset(strong), strong).summary["uar_mean"]
    weak_uar = cross_validate(build_dataset(weak), weak).summary["uar_mean"]
    assert strong_uar > weak_uar
    assert weak_uar < 0.7


def test_too_few_samples_for_the_fold_count_is_reported(config):
    data = build_dataset(config)
    settings = config.override({"detection.n_splits": 2})
    trimmed = type(data)(
        features=data.features[:2],
        labels=data.labels[:2],
        feature_names=data.feature_names,
        observations=data.observations[:2],
        offsets_db=data.offsets_db[:2],
    )
    with pytest.raises(DetectionError, match="folds were requested"):
        cross_validate(trimmed, settings.override({"detection.n_splits": 5}))


@pytest.mark.parametrize("model", ["logistic", "svm", "random_forest", "mlp"])
def test_every_model_trains_and_scores(config, model):
    settings = config.override({"detection.model": model})
    result = cross_validate(build_dataset(settings), settings)
    assert 0.0 <= result.summary["uar_mean"] <= 1.0
    assert "UAR" in result.report()


def test_an_unknown_model_is_rejected():
    with pytest.raises(DetectionError, match="unknown detection model"):
        build_classifier("nonsense", 0)


def test_scaling_lives_inside_the_pipeline_so_it_cannot_leak(config):
    """Fitting the scaler on the whole set before splitting is the classic leak."""
    pipeline = build_classifier(config.detection.model, config.detection.seed)
    assert [name for name, _ in pipeline.steps] == ["scaler", "classifier"]


def test_inference_returns_a_probability_and_a_position(config):
    model = fit_detector(build_dataset(config), config)
    outcome = predict_observation(model, np.array([-70.0, -70.0, -70.0]), config)
    assert 0.0 <= outcome["probability"] <= 1.0
    assert outcome["attacked"] in (0.0, 1.0)
    scenario = config.build_scenario()
    assert scenario.grid.contains(outcome["estimated_position"])


def test_inference_rejects_a_reading_of_the_wrong_length(config):
    model = fit_detector(build_dataset(config), config)
    with pytest.raises(DetectionError, match="access points"):
        predict_observation(model, np.array([-70.0, -70.0]), config)


def test_a_blatantly_spoofed_reading_is_flagged(config):
    """End to end: train, then score a vector with one wildly wrong access point."""
    settings = config.override(
        {"detection.attack_offsets_db": (-45.0,), "detection.n_trials_per_class": 150}
    )
    model = fit_detector(build_dataset(settings), settings)
    spoofed = predict_observation(model, np.array([-70.0, -70.0, -115.0]), settings)
    honest = predict_observation(model, np.array([-70.0, -70.0, -70.0]), settings)
    assert spoofed["probability"] > honest["probability"]
    assert spoofed["attacked"] == 1.0


def test_the_dataset_frame_is_tidy(config):
    frame = build_dataset(config).to_frame()
    assert frame.columns[0] == "attacked"
    assert frame.columns[1] == "attack_offset_db"
    assert len(frame) == 2 * config.detection.n_trials_per_class
