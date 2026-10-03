"""Probability distributions of attack-resistant fingerprint likelihoods.

A Wi-Fi fingerprint localizer scores each candidate position by how well the
observed signal strengths match what a radio map says should be seen there.
The usual score multiplies one Gaussian likelihood per access point, which is
optimal when every reading is honest and catastrophic when one is not: a single
tampered access point drives the product to nearly zero everywhere and the
estimate collapses.

The family studied here, ``ESP(M)`` for *elementary symmetric polynomial*,
replaces the product with a sum over every subset of :math:`N-M` access points,

.. math::

   \\mathrm{ESP}(M) = e_{N-M}(p_1, \\ldots, p_N),

which is the elementary symmetric polynomial of order :math:`N-M` in the
per-access-point likelihoods.  :math:`M` is the number of tampered readings the
score is built to tolerate: :math:`M = 0` is the ordinary product and
:math:`M = N-1` is a plain sum.  Naming the family this way is what lets it be
computed for any :math:`N` by an :math:`O(Nk)` recurrence, in the log domain,
instead of being expanded by hand for one access point count at a time.

The route from a configuration to a result:

1. **Grid** the candidate positions (:mod:`symloc.grid`), as a mean-RSSI sweep or
   as real reference points;
2. **Map** the expected level of each access point at each candidate
   (:mod:`symloc.pathloss`), from a log-distance path loss model;
3. **Score** every access point at every candidate (:mod:`symloc.likelihood`);
4. **Combine** the scores at each order of :math:`M` (:mod:`symloc.combiners`);
5. **Normalise** into distributions while keeping what that discards
   (:mod:`symloc.normalize`);
6. **Measure** sharpness, divergence and localization error
   (:mod:`symloc.metrics`), with confidence intervals over resampled noise
   (:mod:`symloc.montecarlo`);
7. **Decide** whether an observation was tampered with at all
   (:mod:`symloc.detection`), cross-validated;
8. **Record** every figure and table beside the configuration that produced it
   (:mod:`symloc.io_`).

:mod:`symloc.legacy` reproduces the original formulation exactly, defects
included, as a regression target.

Examples
--------
>>> from symloc import Config, analyse_scenario
>>> result = analyse_scenario(Config())
>>> sorted(result.peaks())
[0, 1, 2]
>>> result.peaks()[0] > result.peaks()[2]
True
"""

from __future__ import annotations

__version__ = "0.1.0"

from .ap import APError, APSpec, RadioMap, Scenario, uniform_radio_map
from .combiners import (
    CombinerError,
    available_combiners,
    combine,
    combine_many,
    elementary_symmetric_bruteforce,
    log_elementary_symmetric,
    log_elementary_symmetric_all,
)
from .config import (
    APConfig,
    Config,
    ConfigError,
    DetectionConfig,
    GridConfig,
    MonteCarloConfig,
    NormalizeConfig,
    OutputConfig,
    RadioMapConfig,
    ScenarioConfig,
    SweepConfig,
)
from .detection import (
    CrossValidationResult,
    Dataset,
    build_dataset,
    classification_metrics,
    cross_validate,
    fit_detector,
    predict_observation,
)
from .figio import FigFormatError, read_fig
from .grid import Grid, GridError
from .legacy import legacy_curves, reproduce_one_attack, sign_bug_impact, truncation_artefact
from .likelihood import LikelihoodError, likelihood, log_likelihood, scenario_log_likelihood
from .logging_utils import configure_logging
from .metrics import (
    MetricError,
    argmax_offset,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    curve_metrics,
    effective_support_width,
    excess_kurtosis,
    full_width_half_maximum,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    peak_difference,
    peak_ratio,
    peak_to_secondary_ratio,
    renyi_entropy,
    shannon_entropy,
    total_variation,
)
from .montecarlo import MonteCarloResult, run_montecarlo
from .normalize import Curve, NormalizeError, normalize_log, to_pmf, truncated_mass
from .pathloss import PathLossError, mean_rssi_dbm, pathloss_radio_map
from .pipeline import ScenarioResult, analyse_scenario, curves_to_frame, evidence_collapse
from .sweep import discriminability_pivot, run_sweep

__all__ = [
    "APConfig",
    "APError",
    "APSpec",
    "CombinerError",
    "Config",
    "ConfigError",
    "CrossValidationResult",
    "Curve",
    "Dataset",
    "DetectionConfig",
    "FigFormatError",
    "Grid",
    "GridConfig",
    "GridError",
    "LikelihoodError",
    "MetricError",
    "MonteCarloConfig",
    "MonteCarloResult",
    "NormalizeConfig",
    "NormalizeError",
    "OutputConfig",
    "PathLossError",
    "RadioMap",
    "RadioMapConfig",
    "Scenario",
    "ScenarioConfig",
    "ScenarioResult",
    "SweepConfig",
    "__version__",
    "analyse_scenario",
    "argmax_offset",
    "available_combiners",
    "bhattacharyya_coefficient",
    "bhattacharyya_distance",
    "build_dataset",
    "classification_metrics",
    "combine",
    "combine_many",
    "configure_logging",
    "cross_validate",
    "curve_metrics",
    "curves_to_frame",
    "discriminability_pivot",
    "effective_support_width",
    "elementary_symmetric_bruteforce",
    "evidence_collapse",
    "excess_kurtosis",
    "fit_detector",
    "full_width_half_maximum",
    "hellinger_distance",
    "js_divergence",
    "kl_divergence",
    "legacy_curves",
    "likelihood",
    "log_elementary_symmetric",
    "log_elementary_symmetric_all",
    "log_likelihood",
    "mean_rssi_dbm",
    "normalize_log",
    "pathloss_radio_map",
    "peak_difference",
    "peak_ratio",
    "peak_to_secondary_ratio",
    "predict_observation",
    "read_fig",
    "renyi_entropy",
    "reproduce_one_attack",
    "run_montecarlo",
    "run_sweep",
    "scenario_log_likelihood",
    "shannon_entropy",
    "sign_bug_impact",
    "to_pmf",
    "total_variation",
    "truncated_mass",
    "truncation_artefact",
    "uniform_radio_map",
]
