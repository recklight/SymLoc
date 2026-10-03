"""Deciding whether an observation vector has been tampered with.

The original work argued for a robust likelihood family but never posed the
decision problem that family implies: given one set of readings, *has* an
attack happened?  Without that question there is no classifier, so there are no
classification metrics, no cross-validation and no way to say whether the
shapes it plotted are separable in practice or merely look different.

This module closes that gap.  Monte-Carlo trials generate a balanced, labelled
set -- honest vectors and tampered vectors drawn from the same shadowing model
-- a fixed feature map turns each vector into a descriptor of how well it fits
the radio map, and a stratified :math:`k`-fold cross-validation with a fixed
seed reports unweighted average recall, sensitivity, specificity, the confusion
matrix and the area under the ROC curve.

The features are all quantities a deployed system could actually compute: no
label, no attack offset and no knowledge of which access point was tampered
with enters the feature map.  The most informative of them is the spread of the
per-access-point log likelihoods evaluated at the position the whole set
agrees on -- a spoofed access point fits badly exactly where the honest ones
fit well, which is the signal a robust combiner is built to survive and a
detector is built to notice.

References
----------
.. [1] Y. Chen, W. Trappe, and R. P. Martin, "Detecting and Localizing Wireless
       Spoofing Attacks", *Proc. IEEE SECON*, pp. 193-202, 2007.
.. [2] Z. Li, W. Trappe, Y. Zhang, and B. Nath, "Robust Statistical Methods for
       Securing Wireless Localization in Sensor Networks", *Proc. ACM/IEEE
       IPSN*, pp. 91-98, 2005.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .ap import Scenario
from .config import Config
from .grid import Grid
from .likelihood import log_likelihood_for_observations
from .metrics import effective_support_width, shannon_entropy
from .montecarlo import batched_combine, resolve_true_index, sample_observations
from .normalize import normalize_log

_LOGGER = logging.getLogger(__name__)


class DetectionError(ValueError):
    """Raised when a detection experiment is misconfigured."""


@dataclass(frozen=True, slots=True, eq=False)
class Dataset:
    """A labelled set of observation vectors and their features.

    Attributes
    ----------
    features : numpy.ndarray
        Shape ``(n_samples, n_features)``.
    labels : numpy.ndarray
        Shape ``(n_samples,)``, 0 for honest and 1 for tampered.
    feature_names : tuple of str
        Column names of ``features``.
    observations : numpy.ndarray
        Shape ``(n_samples, n_ap)`` raw readings, kept for inspection.
    offsets_db : numpy.ndarray
        Shape ``(n_samples,)`` attack strength of each sample, 0 for honest.
    """

    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    observations: np.ndarray
    offsets_db: np.ndarray

    @property
    def n_samples(self) -> int:
        """Number of labelled vectors."""
        return int(self.features.shape[0])

    def to_frame(self) -> pd.DataFrame:
        """Features and labels as a tidy table."""
        frame = pd.DataFrame(self.features, columns=list(self.feature_names))
        frame.insert(0, "attacked", self.labels)
        frame.insert(1, "attack_offset_db", self.offsets_db)
        return frame


@dataclass(frozen=True, slots=True, eq=False)
class CrossValidationResult:
    """Cross-validated classification performance.

    Attributes
    ----------
    per_fold : pandas.DataFrame
        One row per fold, one column per metric.
    summary : dict of str to float
        Mean and standard deviation of every metric across folds.
    confusion : numpy.ndarray
        Shape ``(2, 2)`` confusion matrix summed over folds, rows true and
        columns predicted.
    model : str
        Classifier name.
    n_splits : int
        Number of folds.
    seed : int
        Seed used for the fold assignment.
    feature_names : tuple of str
        Features the classifier saw.
    """

    per_fold: pd.DataFrame
    summary: dict[str, float]
    confusion: np.ndarray
    model: str
    n_splits: int
    seed: int
    feature_names: tuple[str, ...] = field(default=())

    def report(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.model}: UAR {self.summary['uar_mean']:.3f} "
            f"+/- {self.summary['uar_std']:.3f}, "
            f"sensitivity {self.summary['sensitivity_mean']:.3f}, "
            f"specificity {self.summary['specificity_mean']:.3f}, "
            f"ROC-AUC {self.summary['roc_auc_mean']:.3f} "
            f"({self.n_splits}-fold, seed {self.seed})"
        )


def feature_names(m_values: tuple[int, ...]) -> tuple[str, ...]:
    """Names of the features :func:`extract_features` produces.

    Examples
    --------
    >>> feature_names((0, 1))[:3]
    ('best_fit_logp_min', 'best_fit_logp_max', 'best_fit_logp_range')
    """
    names = [
        "best_fit_logp_min",
        "best_fit_logp_max",
        "best_fit_logp_range",
        "best_fit_logp_std",
        "best_fit_logp_mean",
        "consensus_logp_min",
        "consensus_logp_range",
        "consensus_logp_std",
    ]
    for order in m_values:
        names.extend(
            [
                f"log_evidence_m{order}",
                f"peak_m{order}",
                f"entropy_m{order}",
                f"support90_m{order}",
            ]
        )
    return tuple(names)


def extract_features(
    log_p: np.ndarray,
    grid: Grid,
    m_values: tuple[int, ...],
    combiner: str = "esp",
    mode: str = "sum",
) -> np.ndarray:
    """Turn a batch of log-likelihood blocks into fixed-length descriptors.

    Every feature is computable from the readings and the radio map alone.

    Parameters
    ----------
    log_p : numpy.ndarray
        Shape ``(n_trials, n_ap, n_points)`` log likelihoods.
    grid : Grid
        The candidate grid.
    m_values : tuple of int
        Orders whose combined curves contribute features.
    combiner : str, optional
        Combining strategy.
    mode : str, optional
        Normalisation mode for the peak feature.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_trials, len(feature_names(m_values)))``.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> block = np.zeros((5, 3, g.size))
    >>> extract_features(block, g, (0, 1)).shape
    (5, 16)
    """
    array = np.asarray(log_p, dtype=np.float64)
    if array.ndim != 3:
        raise DetectionError(
            f"log_p must be 3-D (n_trials, n_ap, n_points), got shape {array.shape}"
        )
    n_trials = array.shape[0]

    # How well each access point fits at its own best candidate: a spoofed
    # reading can still fit somewhere, so this alone is weak.
    best_fit = array.max(axis=2)

    scores = batched_combine(array, m_values, strategy=combiner)
    consensus_order = max(m_values)
    consensus_index = np.argmax(scores[consensus_order], axis=1)
    # How well each access point fits at the position the whole set agrees on.
    # A tampered access point fits badly precisely where the honest ones fit
    # well, which is what separates the two classes.
    consensus = array[np.arange(n_trials), :, consensus_index]

    columns: list[np.ndarray] = [
        best_fit.min(axis=1),
        best_fit.max(axis=1),
        best_fit.max(axis=1) - best_fit.min(axis=1),
        best_fit.std(axis=1),
        best_fit.mean(axis=1),
        consensus.min(axis=1),
        consensus.max(axis=1) - consensus.min(axis=1),
        consensus.std(axis=1),
    ]

    for order in m_values:
        score = scores[order]
        log_evidence = logsumexp(score, axis=1)
        peaks = np.empty(n_trials, dtype=np.float64)
        entropies = np.empty(n_trials, dtype=np.float64)
        supports = np.empty(n_trials, dtype=np.float64)
        for trial in range(n_trials):
            values, _ = normalize_log(score[trial], mode=mode, step=grid.step)
            peaks[trial] = values.max()
            entropies[trial] = shannon_entropy(values)
            supports[trial] = effective_support_width(values, grid, fraction=0.9)
        columns.extend([log_evidence, peaks, entropies, supports])

    return np.column_stack(columns)


def build_dataset(config: Config, scenario: Scenario | None = None) -> Dataset:
    """Generate a balanced labelled set of honest and tampered observations.

    Honest and attacked trials are drawn from the same shadowing model at the
    same true position, so the only systematic difference between the classes
    is the tampering itself.  Attack strengths are drawn uniformly from
    ``config.detection.attack_offsets_db``, which keeps the task from
    collapsing into "detect one specific offset".

    Parameters
    ----------
    config : Config
        Settings; ``config.detection`` supplies the counts, seed and offsets.
    scenario : Scenario, optional
        Scenario defining the access points and the radio map.

    Returns
    -------
    Dataset
        Features, labels and the raw readings behind them.

    Examples
    --------
    >>> from symloc.config import Config
    >>> cfg = Config().override({"detection.n_trials_per_class": 20})
    >>> data = build_dataset(cfg)
    >>> data.n_samples, int(data.labels.sum())
    (40, 20)
    """
    if scenario is None:
        scenario = config.build_scenario()
    settings = config.detection

    if not scenario.attacked_mask.any():
        raise DetectionError(
            f"scenario {scenario.name!r} has no attacked access point, so the "
            "attacked class cannot be generated; set scenario.attacked_indices"
        )

    rng = np.random.default_rng(settings.seed)
    true_index = resolve_true_index(scenario)
    n_per_class = settings.n_trials_per_class

    honest_scenario = scenario.honest()
    honest_obs = sample_observations(honest_scenario, n_per_class, rng, true_index)

    attacked_mask = scenario.attacked_mask
    attacked_obs = sample_observations(honest_scenario, n_per_class, rng, true_index)
    offsets = rng.choice(np.array(settings.attack_offsets_db), size=n_per_class)
    attacked_obs = attacked_obs + np.outer(offsets, attacked_mask.astype(np.float64))

    observations = np.vstack([honest_obs, attacked_obs])
    labels = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    offsets_db = np.concatenate([np.zeros(n_per_class), offsets])

    orders = config.scenario.m_values or tuple(range(scenario.n_ap))
    orders = tuple(order for order in orders if 0 <= order <= scenario.n_ap - 1)

    log_p = log_likelihood_for_observations(observations, scenario.radio_map, scenario.sigma_db)
    features = extract_features(
        log_p,
        scenario.grid,
        orders,
        combiner=config.scenario.combiner,
        mode=config.normalize.mode,
    )

    _LOGGER.info(
        "built detection dataset: %d samples, %d features, seed %d",
        observations.shape[0],
        features.shape[1],
        settings.seed,
    )
    return Dataset(
        features=features,
        labels=labels,
        feature_names=feature_names(orders),
        observations=observations,
        offsets_db=offsets_db,
    )


def build_classifier(model: str, seed: int) -> Pipeline:
    """Build a scaled classifier pipeline by name.

    Scaling is inside the pipeline, so it is fitted on the training fold only
    and never sees the held-out fold -- the standard way this leaks, and the
    reason it is not done once up front.

    Parameters
    ----------
    model : str
        One of :data:`~symloc.config.DETECTOR_MODELS`.
    seed : int
        Seed for models with randomness.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Scaler plus classifier.

    Examples
    --------
    >>> build_classifier("logistic", 0).steps[-1][0]
    'classifier'
    """
    if model == "logistic":
        estimator = LogisticRegression(max_iter=2000, random_state=seed)
    elif model == "svm":
        estimator = SVC(kernel="rbf", probability=True, random_state=seed)
    elif model == "random_forest":
        estimator = RandomForestClassifier(n_estimators=300, random_state=seed)
    elif model == "mlp":
        estimator = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=seed)
    else:
        raise DetectionError(f"unknown detection model {model!r}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Unweighted average recall and the rest of the two-class report.

    Unweighted average recall -- the mean of the per-class recalls -- is the
    headline because it is unmoved by class balance, so a detector that simply
    answers "no attack" every time scores 0.5 rather than whatever the honest
    share of the data happens to be.

    That guarantee only holds while both classes are actually present.  If
    ``y_true`` carries one class only, one of the two recalls does not exist and
    ``uar`` is ``nan`` rather than the recall that does -- averaging over the
    single available class would report a perfect 1.0 for exactly the
    always-answer-the-same-thing detector the measure is chosen to catch.

    Parameters
    ----------
    y_true, y_pred : array_like
        True and predicted labels, 0 for honest and 1 for tampered.
    y_score : array_like, optional
        Scores for the positive class, enabling ROC-AUC.

    Returns
    -------
    dict of str to float
        ``uar``, ``accuracy``, ``sensitivity``, ``specificity``, ``ppv``,
        ``npv``, ``f1``, ``roc_auc`` and the four confusion counts.

    Examples
    --------
    >>> metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    >>> round(metrics["uar"], 3), round(metrics["sensitivity"], 3)
    (0.75, 1.0)
    >>> import math
    >>> math.isnan(classification_metrics([0, 0], [0, 0])["uar"])
    True
    """
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape:
        raise DetectionError(f"y_true has shape {true.shape} but y_pred has {pred.shape}")
    matrix = confusion_matrix(true, pred, labels=[0, 1])
    tn, fp, fn, tp = (float(v) for v in matrix.ravel())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else float("nan")
    f1 = (
        2 * ppv * sensitivity / (ppv + sensitivity)
        if np.isfinite(ppv) and np.isfinite(sensitivity) and (ppv + sensitivity) > 0
        else float("nan")
    )

    # Both recalls must exist, or the average of "the ones that do" is a
    # flattering number for a degenerate evaluation set rather than a score.
    uar = (
        0.5 * (sensitivity + specificity)
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )

    metrics = {
        "uar": float(uar),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if y_score is not None and len(np.unique(true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(true, np.asarray(y_score, dtype=float)))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def cross_validate(
    dataset: Dataset,
    config: Config,
) -> CrossValidationResult:
    """Stratified :math:`k`-fold cross-validation of the spoofing detector.

    Stratified because the fold assignment must preserve the class balance;
    seeded because a reported number that cannot be reproduced is not a result.

    Parameters
    ----------
    dataset : Dataset
        Features and labels.
    config : Config
        Settings; ``config.detection`` supplies the model, folds and seed.

    Returns
    -------
    CrossValidationResult
        Per-fold metrics, their summary and the pooled confusion matrix.

    Examples
    --------
    >>> from symloc.config import Config
    >>> cfg = Config().override({"detection.n_trials_per_class": 40, "detection.n_splits": 3})
    >>> result = cross_validate(build_dataset(cfg), cfg)
    >>> result.per_fold.shape[0]
    3
    >>> bool(result.summary["uar_mean"] > 0.5)
    True
    """
    settings = config.detection
    features = dataset.features
    labels = dataset.labels

    if features.shape[0] != labels.shape[0]:
        raise DetectionError(f"{features.shape[0]} feature rows but {labels.shape[0]} labels")
    counts = np.bincount(labels, minlength=2)
    if counts.min() < settings.n_splits:
        raise DetectionError(
            f"the smaller class has {counts.min()} samples but "
            f"{settings.n_splits} folds were requested; reduce detection.n_splits "
            "or raise detection.n_trials_per_class"
        )

    splitter = StratifiedKFold(n_splits=settings.n_splits, shuffle=True, random_state=settings.seed)
    rows: list[dict[str, float]] = []
    pooled = np.zeros((2, 2), dtype=int)

    for fold, (train_index, test_index) in enumerate(splitter.split(features, labels)):
        classifier = build_classifier(settings.model, settings.seed)
        classifier.fit(features[train_index], labels[train_index])
        predicted = classifier.predict(features[test_index])
        try:
            scores = classifier.predict_proba(features[test_index])[:, 1]
        except AttributeError:  # pragma: no cover - every model here has it
            scores = classifier.decision_function(features[test_index])

        metrics = classification_metrics(labels[test_index], predicted, scores)
        metrics["fold"] = float(fold)
        rows.append(metrics)
        pooled += confusion_matrix(labels[test_index], predicted, labels=[0, 1])

    per_fold = pd.DataFrame(rows)
    ordered = ["fold"] + [c for c in per_fold.columns if c != "fold"]
    per_fold = per_fold[ordered]

    summary: dict[str, float] = {}
    for column in per_fold.columns:
        if column == "fold":
            continue
        values = per_fold[column].to_numpy(dtype=float)
        summary[f"{column}_mean"] = float(np.nanmean(values))
        summary[f"{column}_std"] = float(np.nanstd(values, ddof=0))

    result = CrossValidationResult(
        per_fold=per_fold,
        summary=summary,
        confusion=pooled,
        model=settings.model,
        n_splits=settings.n_splits,
        seed=settings.seed,
        feature_names=dataset.feature_names,
    )
    _LOGGER.info("%s", result.report())
    return result


def fit_detector(dataset: Dataset, config: Config) -> Pipeline:
    """Fit the configured classifier on the whole dataset, for later inference.

    Parameters
    ----------
    dataset : Dataset
        Features and labels.
    config : Config
        Settings.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted pipeline.

    Examples
    --------
    >>> from symloc.config import Config
    >>> cfg = Config().override({"detection.n_trials_per_class": 20})
    >>> model = fit_detector(build_dataset(cfg), cfg)
    >>> model.predict(build_dataset(cfg).features).shape
    (40,)
    """
    classifier = build_classifier(config.detection.model, config.detection.seed)
    classifier.fit(dataset.features, dataset.labels)
    return classifier


def predict_observation(
    model: Pipeline,
    observation_dbm: np.ndarray,
    config: Config,
    scenario: Scenario | None = None,
) -> dict[str, float]:
    """Score one observation vector with a fitted detector.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        A pipeline from :func:`fit_detector`.
    observation_dbm : array_like
        Shape ``(n_ap,)`` readings in dBm.
    config : Config
        Settings used to build the features; must match the training run.
    scenario : Scenario, optional
        Scenario supplying the radio map.

    Returns
    -------
    dict of str to float
        ``attacked`` (0 or 1), ``probability`` and the estimated position.

    Examples
    --------
    >>> import numpy as np
    >>> from symloc.config import Config
    >>> cfg = Config().override({"detection.n_trials_per_class": 30})
    >>> model = fit_detector(build_dataset(cfg), cfg)
    >>> out = predict_observation(model, np.array([-70.0, -70.0, -70.0]), cfg)
    >>> 0.0 <= out["probability"] <= 1.0
    True
    """
    if scenario is None:
        scenario = config.build_scenario()
    observation = np.asarray(observation_dbm, dtype=np.float64)
    if observation.ndim != 1:
        # Reshaping blindly would flatten a batch of vectors into one long
        # reading and score the wrong thing without complaining.
        raise DetectionError(
            f"observation_dbm must be one-dimensional (n_ap,), got shape {observation.shape}; "
            "score a batch with extract_features instead"
        )
    observation = observation[None, :]
    if observation.shape[1] != scenario.n_ap:
        raise DetectionError(
            f"observation has {observation.shape[1]} readings but the scenario has "
            f"{scenario.n_ap} access points"
        )

    orders = config.scenario.m_values or tuple(range(scenario.n_ap))
    orders = tuple(order for order in orders if 0 <= order <= scenario.n_ap - 1)

    log_p = log_likelihood_for_observations(observation, scenario.radio_map, scenario.sigma_db)
    features = extract_features(
        log_p,
        scenario.grid,
        orders,
        combiner=config.scenario.combiner,
        mode=config.normalize.mode,
    )
    label = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])

    scores = batched_combine(log_p, orders, strategy=config.scenario.combiner)
    robust_order = max(orders)
    estimated_index = int(np.argmax(scores[robust_order][0]))

    return {
        "attacked": float(label),
        "probability": probability,
        "estimated_index": float(estimated_index),
        "estimated_position": float(scenario.grid.values[estimated_index]),
    }
