# SymLoc

[![CI](https://github.com/recklight/SymLoc/actions/workflows/ci.yml/badge.svg)](https://github.com/recklight/SymLoc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

English | [繁體中文](README.zh-TW.md)

Probability-distribution analysis of a family of tamper-resistant likelihoods for indoor Wi-Fi fingerprint localization.

A localizer that matches the signal strengths of several access points (APs) against a fingerprint database usually scores each position with the product of one Gaussian likelihood per AP. That is optimal as long as every AP reports honestly. Tamper with **one** AP and the product collapses at every candidate position at once, and the position estimate goes with it.

The score I look at here replaces that product with a sum over subsets: take every subset of $N-M$ APs, multiply within each one, and add the products up. That sum is the **elementary symmetric polynomial** (ESP from here on), and $M$ is the number of tampered APs the score is designed to tolerate.

---

## Contents

- [Overview](#overview)
- [Method](#method)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage examples](#usage-examples)
- [CLI reference](#cli-reference)
- [Project layout](#project-layout)
- [Data preparation](#data-preparation)
- [Configuration](#configuration)
- [Results and metrics](#results-and-metrics)
- [Improvements](#improvements)
- [Development and testing](#development-and-testing)
- [References](#references)

---

## Overview

The whole analysis is **purely analytic**. Every number is computed from the model on the fly, with no data files, no downloads and no measurements, so every conclusion can be reproduced in full.

| Stage | What happens | Module |
| --- | --- | --- |
| 1. Candidate grid | Build the candidate positions, either as a sweep of mean signal strength or as real reference points | `symloc.grid` |
| 2. Radio map | Compute each AP's expected signal strength at each candidate from a path loss model | `symloc.pathloss` |
| 3. Per-AP likelihood | Gaussian log likelihood for every AP at every candidate | `symloc.likelihood` |
| 4. Combination | Combine the $N$ likelihoods into one score for each order $M$ | `symloc.combiners` |
| 5. Normalisation | Turn the scores into probability distributions while keeping what normalisation throws away | `symloc.normalize` |
| 6. Metrics | Sharpness, divergence and localization error, with confidence intervals | `symloc.metrics`, `symloc.montecarlo` |
| 7. Detection | Decide whether a set of observations was tampered with at all (cross-validated) | `symloc.detection` |
| 8. Output | Write figures, tables and the configuration together, with a run manifest | `symloc.io_` |

`symloc.legacy` also reproduces the original formulas exactly, defects included. It serves only as a regression baseline.

---

## Method

### Per-AP Gaussian likelihood

Under log-normal shadowing, the dBm reading of AP $k$ at candidate position $j$ is normally distributed around the value the radio map predicts, so the likelihood is

$$
p_k(j) = \frac{1}{\sigma_k \sqrt{2\pi}}
         \exp\left[-\frac{(r_k - \mu_{kj})^2}{2\sigma_k^2}\right]
$$

where $r_k$ is the observation, $\mu_{kj}$ the mean predicted by the map and $\sigma_k$ the shadowing standard deviation in dB.

The Gaussian is not picked for convenience; it comes out of the model, because the shadowing term is Gaussian in the **dB domain** (see `symloc.pathloss`).

### The elementary symmetric polynomial combiner

ESP($M$) takes every subset of $N-M$ out of the $N$ APs, multiplies the likelihoods inside each subset and sums over all subsets:

$$
\mathrm{ESP}(M) \;=\;
\sum_{\substack{S \subseteq \{1,\dots,N\} \\ |S| = N-M}} \; \prod_{k \in S} p_k
\;=\; e_{N-M}(p_1, \dots, p_N)
$$

That is the **elementary symmetric polynomial** of order $N-M$, $e_{N-M}$. Recognising it is what the rest of SymLoc builds on. The original analysis expanded each order by hand into a long algebraic expression that had to be derived again for every AP count, so it stayed stuck at the 3-AP and 4-AP cases. Written as $e_k$, the score can be computed for any $N$ in one pass with an $O(Nk)$ recurrence.

The two extremes are easy to read:

- $M = 0$ gives $e_N = \prod_k p_k$, the standard naive-Bayes fingerprint likelihood. A **single** tampered AP drives it towards zero.
- $M = N-1$ gives $e_1 = \sum_k p_k$, a plain sum. No single AP can veto it, but its peak is also very flat.

So $M$ is **the knob that trades tamper resistance against sharpness**.

### The recurrence in the log domain

The recurrence comes from the coefficients of

$$
\prod_{k=1}^{N} (x + p_k) = \sum_{j=0}^{N} e_j(p)\, x^{N-j}
$$

SymLoc runs it in the **log domain**, where products become sums and sums become `numpy.logaddexp`. The linear domain does not survive larger AP counts: the product of 15 AP densities underflows to 0 in float64, and after normalisation the whole curve is `nan`. The log-domain recurrence stays exact throughout. `tests/test_combiners.py` checks the recurrence against brute-force enumeration with `itertools.combinations`, and shows the linear domain breaking down where the log domain holds.

### Path loss and a real radio map

The log-distance path loss model:

$$
\mu(d) = P_{d_0} - 10\,n \log_{10}\!\left(\frac{d}{d_0}\right)
$$

where $n$ is the path loss exponent (about 2 in free space, about 3 to 5 indoors with partitions). Once the APs are placed on a floor plan, every AP has **its own** expected value at every reference point. Only then does a point on the candidate axis stand for a place on the floor, and only then can localization error be measured in metres.

The original analysis assumed that all APs share one mean at a given candidate, so its axis was really a one-dimensional sweep of mean signal strength, not a position on a floor. SymLoc supports both (`radio_map.kind` set to `uniform` or `pathloss`) and states in the run manifest whether the map is degenerate.

### Normalisation, and what it throws away

Dividing a curve by its own sum gives a probability mass function (pmf), which makes the **shapes** of different curves comparable. It also throws away two things the case for tamper resistance depends on, and SymLoc keeps both.

**The absolute magnitude.** The claim that tampering with an AP makes the fingerprint match collapse is a claim about how the absolute likelihood changes. Dividing each curve by its own sum removes exactly that difference: attacked and unattacked curves both end up summing to 1. `Curve` therefore records `log_evidence`, the log of the unnormalised sum.

**Probability mass that falls off the grid.** When an observation lies far outside the swept range (say the reading was forced to −99 dBm while the grid stops at −85 dBm), most of that AP's likelihood mass is outside the grid, and normalisation rescales it arbitrarily, by however much happened to fall off. Left unchecked, this produces spurious results: with the parameter values still set in the original scripts, the attacked naive-Bayes curve peaks **higher** than the unattacked baseline. `truncated_mass()` quantifies the loss and `check_truncation()` warns about it.

How you normalise matters as well. Dividing by the sum gives a pmf whose peak scales with the grid resolution; dividing by sum × step gives a density whose peak does not. Both modes are available, and the mode used and the step size are written into the manifest with the results.

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/recklight/SymLoc.git
cd SymLoc
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[parquet]"   # also write the result tables as Parquet (CSV is always written)
python -m pip install -e ".[dl]"        # PyTorch; nothing in the core package needs it
```

To run straight from the source tree without installing:

```bash
PYTHONPATH=src python -m symloc.cli --help
```

---

## Quick start

```bash
# Reproduce the archived results and quantify the sign error in the code behind them
symloc reproduce-legacy

# Run one scenario: curves, metrics, figures and a manifest
symloc run --config configs/one_attack.yaml

# Sweep attack strength × sigma × AP count × M and draw the discriminability heat map
symloc sweep --config configs/sweep_attack_strength.yaml --metric js_divergence

# Train the tamper detector with stratified cross-validation
symloc evaluate
```

An end-to-end demo that runs without any arguments:

```bash
python examples/end_to_end_synthetic.py
```

---

## Usage examples

### CLI

```bash
# 1. Prepare: build a labelled tamper-detection dataset from synthetic trials
symloc prepare --set detection.n_trials_per_class=500

# 2. Train: fit the detector and save it
symloc train --output results/detector.joblib

# 3. Evaluate: stratified k-fold cross-validation with the full set of classification metrics
symloc evaluate --set detection.model=random_forest

# 4. Predict: a verdict for one set of observations
#    dBm is always negative, so absolute values are accepted; this one is -70, -70, -110 dBm
symloc predict 70,70,110 --model results/detector.joblib
```

Any setting can be overridden on the spot with `--set key=value`, without editing the config file:

```bash
symloc run --set grid.step_db=0.25 --set normalize.mode=density --set scenario.combiner=trimmed
```

### Python API

```python
from symloc import Config, analyse_scenario, evidence_collapse

# Everything comes from the config; no source code needs editing
config = Config.from_yaml("configs/one_attack.yaml")
result = analyse_scenario(config)

# Peak at each order M
for m, peak in sorted(result.peaks().items()):
    print(f"M={m}: {peak:.7f}")

# Peak ratio between adjacent orders (the discriminability measure of the original analysis)
print(result.adjacent_peak_ratios())

# Change in absolute likelihood, in nats: normalisation erases it, but the tamper-resistance claim rests on it
print(evidence_collapse(result))

# Full metrics table (tidy long format)
print(result.metrics.head())
```

Tamper detection:

```python
from symloc import Config, build_dataset, cross_validate

config = Config().override({"detection.model": "logistic", "detection.n_splits": 5})
dataset = build_dataset(config)          # synthetic trials, fixed random seed
outcome = cross_validate(dataset, config)

print(outcome.report())
# logistic: UAR 0.658 +/- 0.039, sensitivity 0.540, specificity 0.776,
#           ROC-AUC 0.734 (5-fold, seed 20120601)
print(outcome.confusion)                  # confusion matrix (rows true, columns predicted)
```

---

## CLI reference

| Command | What it does | Main output |
| --- | --- | --- |
| `symloc run` | Runs the whole pipeline for one scenario | Curves, metrics table, four figures, config, manifest |
| `symloc sweep` | Multi-dimensional parameter sweep | Long-format result table, pivot tables, discriminability heat map |
| `symloc figure` | Redraws the figures only, without recomputing the tables | Figure files |
| `symloc reproduce-legacy` | Reproduces the archived results and quantifies their defects | Terminal report, optional comparison figure |
| `symloc inspect-fig` | Reads a MATLAB `.fig` you supply and prints the curves in it | Terminal report |
| `symloc prepare` | Builds a labelled tamper-detection dataset | Dataset table, manifest |
| `symloc train` | Fits the detector and saves it | `.joblib` model file |
| `symloc evaluate` | Stratified k-fold cross-validation | Per-fold metrics table, confusion-matrix figure, ROC figure |
| `symloc predict` | Gives a verdict for a single set of observations | Verdict in the terminal |

Shared options (not every subcommand takes all of them; the right-hand column lists the ones that do):

| Option | What it does | Subcommands |
| --- | --- | --- |
| `--config`, `-c` | YAML config file to use | `run`, `sweep`, `figure`, `prepare`, `train`, `evaluate`, `predict` |
| `--set key=value`, `-s` | Overrides one setting; can be repeated | Same as above |
| `--label` | Prefix for the output folder name | `run`, `sweep`, `figure`, `prepare`, `evaluate` |
| `--figures` / `--no-figures` | Whether to write figures | `run`, `evaluate` |
| `--output`, `-o` | Output path | `train` (model file), `reproduce-legacy` (folder for the comparison figure) |
| `--metric` | Metric shown in the heat map | `sweep` |
| `--model`, `-m` | Model file to load | `predict` |

---

## Project layout

```
SymLoc/
├── configs/                      config files: scenarios, sweep axes, reproduction targets
│   ├── default.yaml              defaults, and the documentation of every config key
│   ├── one_attack.yaml           3 APs, 1 tampered
│   ├── two_attack.yaml           4 APs, 2 tampered
│   ├── legacy_exact.yaml         reproduction target for the archived results (formula corrected)
│   ├── pathloss_floor.yaml       real floor plan with a path loss map
│   └── sweep_attack_strength.yaml  attack strength × sigma × AP count × M
├── data/legacy_reference/
│   └── peaks.json                read-only reference values recovered from the archived figures (derived numbers only)
├── examples/
│   ├── end_to_end_synthetic.py   end-to-end demo on synthetic data only
│   ├── reproduce_legacy_and_fix.py  reproduces the archived results and compares each item before and after the fix
│   └── combiner_robustness.py    how well each combiner holds up on a real radio map
├── src/symloc/
│   ├── grid.py                   candidate grid (dBm sweep or 2-D reference points)
│   ├── ap.py                     APSpec, RadioMap, Scenario
│   ├── pathloss.py               log-distance path loss and shadowing
│   ├── likelihood.py             per-AP Gaussian likelihood (log domain)
│   ├── combiners.py              e_k recurrence and the combining strategies
│   ├── normalize.py              normalisation modes and truncation diagnostics
│   ├── metrics.py                divergence, sharpness, localization error
│   ├── montecarlo.py             repeated trials and confidence intervals
│   ├── detection.py              tamper detection and stratified cross-validation
│   ├── pipeline.py               the full pipeline for one scenario
│   ├── sweep.py                  multi-dimensional parameter sweep
│   ├── legacy.py                 exact reproduction of the original formulas (with defect switches)
│   ├── figio.py                  read-only reader for MATLAB .fig files
│   ├── io_.py                    writes results and the run manifest
│   ├── config.py                 typed, validated config objects
│   ├── cli.py                    typer CLI
│   ├── logging_utils.py          logging setup
│   ├── py.typed                  type marker (PEP 561)
│   └── plotting/                 figures (colours, distribution plots, bar charts, heat maps)
├── tests/                        pytest, all fixtures generated in code
├── pyproject.toml
└── LICENSE
```

---

## Data preparation

**No data files are needed.** Every number is generated by the model on the fly, and `tests/` uses only fixtures generated in code. Clone the repository and everything runs.

You only need files of your own in two cases.

### 1. Inspecting your own archived MATLAB `.fig` figures

`symloc inspect-fig` reads the `.fig` file at the path you give it and pulls out its `XData`, `YData` and legend entries. The file is neither copied into the project nor modified.

```bash
symloc inspect-fig /path/to/your/figure.fig
```

A `.fig` file is a MAT-file v5 container underneath. If a file is an EPS saved under a `.fig` extension, the tool names the format it actually has instead of failing with an obscure parse error.

### 2. Plugging in a real fingerprint database

To replace the path loss model with a radio map you measured, put the AP coordinates and the reference-point coordinates in the config file (see `configs/pathloss_floor.yaml`), or build a `RadioMap` directly in Python:

```python
import numpy as np
from symloc import Grid, RadioMap, Scenario, APSpec

grid = Grid.from_points(reference_point_coordinates)   # (n_points, 2), in metres
radio_map = RadioMap(measured_means_dbm, ("AP1", "AP2", "AP3"))  # (n_ap, n_points), in dBm
scenario = Scenario.build("my_site", grid, aps, radio_map=radio_map, true_index=42)
```

Download external datasets (UJIIndoorLoc, for example; see References) yourself, keep them outside the project and give their paths in the config file. SymLoc neither distributes nor includes any dataset.

---

## Configuration

Every constant the original analysis hard-coded at the top of its scripts is a config entry here. `configs/default.yaml` explains every key and doubles as the reference documentation.

| Section | Main contents |
| --- | --- |
| `grid` | End points and step of the candidate grid |
| `radio_map` | `uniform` (degenerate) or `pathloss` (real geometry), with its parameters |
| `scenario` | AP list, standard deviations, which APs are tampered, offsets, the orders $M$ to evaluate, combiner |
| `normalize` | `sum` / `density` / `none`, and the truncation warning threshold |
| `montecarlo` | Number of trials, random seed, confidence level |
| `detection` | Classifier, number of folds, trials per class, seed, attack strengths |
| `sweep` | The sweep axes |
| `output` | Output folder, figure format, resolution |

An unknown key raises an **error** instead of being ignored, so a typo in a config file is caught at once rather than quietly leaving a default in place.

---

## Results and metrics

Every run creates a folder named after a hash of the configuration. It holds the tables, the figures, the complete YAML config of that run, and a `manifest.json` recording the package version, interpreter, library versions, random seeds and a timestamp. **A file name can never disagree with the parameters, because the name is computed from them.**

### Distribution metrics

| Metric | Meaning |
| --- | --- |
| `peak`, `peak_ratio`, `peak_difference` | The only measures the original analysis used, kept for comparison |
| `kl_to_reference`, `kl_from_reference` | KL divergence in both directions (it is not symmetric) |
| `js_divergence` | Jensen-Shannon divergence, symmetric and bounded |
| `bhattacharyya_distance`, `hellinger_distance` | How much two distributions overlap |
| `total_variation` | Total variation distance |
| `shannon_entropy`, `renyi_entropy_2` | Uncertainty of the distribution |
| `fwhm` | Full width at half maximum; `nan` when the curve does not drop back to half height inside the grid (which is itself a sign of truncation) |
| `excess_kurtosis` | Excess kurtosis, 0 for a Gaussian |
| `peak_to_secondary_ratio` | Ratio of the peak to the next-highest local maximum, to see whether the position is ambiguous |
| `effective_support_90` | Smallest width that holds 90% of the probability mass, in the units of the axis |
| `log_evidence`, `log_evidence_gap` | The absolute likelihood, and the difference that normalisation erases |
| `localization_error` | Distance between the argmax and the true position; in metres on a geometric grid |

### Classification metrics (tamper detection)

Evaluated with **stratified** k-fold cross-validation and a fixed random seed. Standardisation sits inside the pipeline, so it is fitted on the training folds only.

The metrics are `uar` (unweighted average recall, the primary metric), `accuracy`, `sensitivity`, `specificity`, `ppv`, `npv`, `f1`, `roc_auc`, and the confusion matrix.

UAR is the primary metric because with unbalanced classes a model that always answers "not attacked" can reach a good-looking accuracy, while its UAR is only 0.5.

Detection gets monotonically easier as the attack gets stronger, as it must for any correct detector. The rows below use the default scenario (3 APs, $\sigma$ of 8/10/10 dB, logistic, 5 folds, 250 trials per class, seed 20120601), and each row is reproduced exactly by its own command:

| Attack offset | UAR | ROC-AUC | Command to reproduce |
| --- | --- | --- | --- |
| −5 dB | 0.448 | 0.431 | `symloc evaluate --no-figures --set 'detection.attack_offsets_db=[-5.0]'` |
| −15 dB | 0.624 | 0.654 | `symloc evaluate --no-figures --set 'detection.attack_offsets_db=[-15.0]'` |
| −30 dB | 0.816 | 0.904 | `symloc evaluate --no-figures --set 'detection.attack_offsets_db=[-30.0]'` |
| −45 dB | 0.958 | 0.994 | `symloc evaluate --no-figures --set 'detection.attack_offsets_db=[-45.0]'` |

An offset of −5 dB sits entirely inside 8 to 10 dB of shadowing noise, so it should go undetected. A UAR of 0.448 is scatter around chance (0.5), not a defect that makes the model worse than chance. At −45 dB the attack is far outside the noise and is caught almost every time.

---

## Improvements

Each item below closes a gap in the original analysis, and the scripts under `examples/` let you see every one of them.

### Defects fixed

**1. A sign error that reached the archived figures.** In the three-AP expansion of $e_2$, the observation in the third pairwise product is missing its minus sign. The density is therefore evaluated at **+70 dBm**, where it is about $8 \times 10^{-39}$ instead of about $7 \times 10^{-4}$, and the term silently drops to zero. The $M=1$ peak in the archived figure, 0.0305445, matches the buggy expression to the seventh decimal place; the mathematically correct value is 0.0283886. **The archived peak is 7.59% too high, and that peak is exactly the quantity the discriminability argument rests on.**

| Curve | Archived | Reproduced here | Corrected |
| --- | --- | --- | --- |
| No-attack baseline | 0.0387524 | 0.0387524 | 0.0387524 |
| M=0 | 0.0390380 | 0.0390380 | 0.0390380 |
| M=1 | 0.0305445 | 0.0305445 | **0.0283886** |
| M=2 | 0.0242794 | 0.0242794 | 0.0242794 |

Both discriminability ratios move with it: $\rho(0,1)$ goes from 1.27807 to 1.37513, and $\rho(1,2)$ from 1.25805 to 1.16925. `symloc reproduce-legacy` prints the two side by side.

**2. An artefact of truncated normalisation.** With the parameter values still set in the original scripts, the attacked $M=0$ curve peaks at 0.0387726, **above** the no-attack baseline of 0.0387524. The attack did not improve localization. The tampered AP simply lost more of its probability mass off the grid, and dividing by the smaller sum lifted what was left. `truncated_mass()` reports how much each AP loses, and `check_truncation()` warns when the loss passes a threshold.

**3. The peak scales with grid resolution, and the step was recorded nowhere.** The same scenario at different steps:

| Step | Grid points | M=0 peak |
| --- | --- | --- |
| 1.0 dB | 26 | 0.0775811 |
| 0.5 dB | 51 | 0.0390380 |
| 0.25 dB | 101 | 0.0195594 |

The `density` mode gives a peak that does not depend on the step, and the step always goes into the manifest.

**4. File names that disagree with the parameters.** The archived figure's file name says the tampered AP read −99 dBm; an exact inversion of the four curve peaks gives −90 dBm. Every run writes a run manifest, and the folder name comes straight from a hash of the configuration, so a file name cannot drift away from its parameters.

**5. Axis labels that disagree with the data.** The original plotting function forced the x-axis tick **labels** without setting the tick **positions**, so the labels landed on automatically chosen ticks and repeated in a cycle. The "Reference location 0–30" printed on the figure does not correspond to the −60 to −85 dBm underneath at all. Axis labels and units come from the `Grid` object, and the plotting code never invents label text of its own.

**6. A typo in the legend.** Every archived figure spells it `Byaesian`. It reads `Bayesian` here, and a test keeps it that way.

**7. Hard-coded dimensions.** The original plotting template hard-codes `set(plot1(1))` to `set(plot1(4))` and breaks whenever the AP count is not 4; `repmat(X,4,1)` bakes in the 4 as well. Colours and markers are cycled here, so a figure with 8 APs and 8 orders of $M$ still draws (a test covers it).

**8. A malformed archive file.** One `.fig` is really an EPS carrying a `.fig` extension. `symloc inspect-fig` checks the magic bytes in the header first and says plainly which format the file is.

### Numerics and method

**9. Log domain throughout.** The original analysis multiplied densities in the linear domain. With 4 APs the smallest values were already around $10^{-8}$, and with more APs they underflow to 0. The log-domain $e_k$ recurrence with `logaddexp` stays exact at 15 APs (a test shows the linear domain already broken and the log domain still correct).

**10. A recurrence instead of hand-expanded combinations.** An $O(Nk)$ recurrence computes every order in one pass, for any number of APs. A test checks it against brute-force enumeration with `itertools.combinations`.

**11. The absolute likelihood is kept.** `log_evidence` and `log_evidence_gap` record the difference in absolute magnitude that normalisation erases.

**12. A real radio map.** A log-distance path loss model gives every AP its own expected value at every reference point. Points on the axis are then places on the floor, and localization error can be measured in metres. Whether the map is degenerate is recorded in the manifest.

**13. A full set of metrics.** Besides the peak ratio and peak difference between adjacent orders, SymLoc computes KL (both directions), JS, Bhattacharyya, Hellinger, total variation, Shannon/Rényi entropy, FWHM, excess kurtosis, peak-to-secondary ratio, effective support width and localization error.

**14. Monte Carlo and confidence intervals.** The original study was a single deterministic calculation and could not say anything about variability. Here the observation noise is resampled with fixed random seeds, and each metric is reported as a mean with a percentile interval. A normal approximation would not do, since several of the metrics are bounded or heavily skewed.

**15. Tamper detection and cross-validation.** The original analysis argued for an attack-resistant likelihood family but never posed the decision problem it implies: **was there an attack or not?** Monte Carlo trials generate a balanced, labelled dataset, a fixed feature map turns each trial into a descriptor, and stratified k-fold cross-validation reports UAR, sensitivity, specificity, the confusion matrix and ROC-AUC. No feature sees the label, the attack offset or which AP was tampered with. The most discriminative one is the spread of the per-AP log likelihoods at the position the whole set of APs agrees on (feature `consensus_logp_range`, Cohen's $d \approx 0.88$ in the default scenario).

### Engineering

**16. Config files instead of bare constants.** The AP count, observations, standard deviations and attack scenario all come from the config file, with full validation ($\sigma > 0$, $M < N$, observations inside the grid, unknown keys rejected with an error).

**17. One parameterised pipeline.** Where the original scripts copy-paste the `y1` / `y2` blocks and unroll their normalisation loops five times and then four times, SymLoc runs one config-driven pipeline. The no-attack baseline goes down the same path instead of being a separately hard-coded expression.

**18. Multi-dimensional sweeps.** `symloc sweep` uses `itertools.product` to sweep attack strength × sigma × AP count × $M$ × combiner, and writes a long-format result table and a discriminability heat map.

**19. Results saved to disk, with version information.** Tables are written as CSV (always) and Parquet (when an engine is installed), together with a full manifest. In the original analysis, `MP3` was computed and then left sitting in the workspace, never printed and never written to a file.

**20. Tests, logging, input validation, type hints.** 578 tests (doctests included), all on fixtures generated in code; library code logs through `logging` rather than `print`; full type annotations and numpy-style docstrings. A dedicated Python 3.10 compatibility gate runs `ast.parse(feature_version=(3, 10))` and scans for standard-library names newer than 3.10 and for what NumPy 2.0 removed, because passing on 3.13 does not mean 3.10 can run it.

### New findings

**21. The leave-M-out sum does not buy the tamper resistance it was meant to.** The original analysis could not have found this: its axis had no geometry, so it could not measure distance.

The setup has five APs on a 20 × 15 m floor, one of them tampered. The sweep covers 40 real positions × 5 choices of tampered AP, 200 combinations in all, and each combiner is scored at its **best** intermediate order $M$. This is the output of `python examples/combiner_robustness.py` (default seed):

| Combiner | Mean error | Median error | p90 error | Max error | Exact hits |
| --- | --- | --- | --- | --- | --- |
| `esp` (elementary symmetric polynomial) | 0.662 m | 0.000 m | 0.00 m | 14.318 m | 91.0% |
| `trimmed` (least trimmed) | 0.000 m | 0.000 m | 0.00 m | 0.000 m | **100.0%** |
| `median` | 0.000 m | 0.000 m | 0.00 m | 0.000 m | **100.0%** |
| `softmin` | 7.031 m | 7.000 m | 11.18 m | 15.133 m | 1.0% |

The cause is in the definition: $e_{N-M}$ averages over **all** subsets of size $N-M$, including the ones that contain the tampered AP, and those keep their full weight. A tampered reading that happens to fit some wrong position well survives the averaging. Trimming **picks** the best-fitting subset instead of averaging over all of them, so it can drop that reading outright.

Soft-min does worst for the same reason seen from the other side: it gives the most weight to the AP that fits **worst**, so it follows the tampered reading. It is in the table to show that the aggregation that feels most conservative is the wrong tool here.

Reproduce it with `python examples/combiner_robustness.py`.

---

## Development and testing

```bash
python -m pytest -q                       # 578 tests (doctests included)
python -m ruff check .                    # lint
python -m ruff format --check .           # formatting
```

The tests fall into a few groups:

- Numerical correctness: the recurrence against brute-force enumeration; KL and Bhattacharyya against the closed forms for two Gaussians; FWHM against $2\sqrt{2\ln 2}\,\sigma$.
- Regression: `tests/test_legacy_parity.py` checks that the archived values can be reproduced and pins down how large the defects are.
- Invariants: with no attack every curve equals its control; a stronger attack pushes the posterior further from the baseline; a larger $M$ tolerates more tampering but gives a less sharp peak.
- Compatibility: `tests/test_python310_compat.py` is the Python 3.10 gate.
- End to end: every CLI subcommand is run once the way a user would run it, and the test checks that every output it promises is there.

Every test fixture is generated in code, and no real data file is read. PyTorch is optional: the whole package imports and passes its tests without it.

---

## References

### Fingerprint localization

- P. Bahl and V. N. Padmanabhan, "RADAR: An In-Building RF-Based User Location and Tracking System," *Proc. IEEE INFOCOM*, 2000, pp. 775-784.
- T. Roos, P. Myllymäki, H. Tirri, P. Misikangas, and J. Sievänen, "A Probabilistic Approach to WLAN User Location Estimation," *International Journal of Wireless Information Networks*, vol. 9, no. 3, pp. 155-164, 2002.
- M. Youssef and A. Agrawala, "The Horus WLAN Location Determination System," *Proc. ACM MobiSys*, 2005, pp. 205-218.
- T. S. Rappaport, *Wireless Communications: Principles and Practice*, 2nd ed., Prentice Hall, 2002.

### Attack-resistant localization

- D. Liu, P. Ning, and W. Du, "Attack-Resistant Location Estimation in Sensor Networks," *Proc. ACM/IEEE IPSN*, 2005, pp. 99-106.
- Z. Li, W. Trappe, Y. Zhang, and B. Nath, "Robust Statistical Methods for Securing Wireless Localization in Sensor Networks," *Proc. ACM/IEEE IPSN*, 2005, pp. 91-98.
- Y. Chen, W. Trappe, and R. P. Martin, "Detecting and Localizing Wireless Spoofing Attacks," *Proc. IEEE SECON*, 2007, pp. 193-202.

### Symmetric functions and robust statistics

- I. G. Macdonald, *Symmetric Functions and Hall Polynomials*, 2nd ed., Oxford University Press, 1995, Chapter I.2.
- P. J. Rousseeuw, "Least Median of Squares Regression," *Journal of the American Statistical Association*, vol. 79, no. 388, pp. 871-880, 1984.
- P. J. Huber, *Robust Statistics*, John Wiley & Sons, 1981.

### Divergence measures

- S. Kullback and R. A. Leibler, "On Information and Sufficiency," *Annals of Mathematical Statistics*, vol. 22, no. 1, pp. 79-86, 1951.
- A. Bhattacharyya, "On a Measure of Divergence Between Two Statistical Populations Defined by Their Probability Distributions," *Bulletin of the Calcutta Mathematical Society*, vol. 35, pp. 99-109, 1943.
- T. Kailath, "The Divergence and Bhattacharyya Distance Measures in Signal Selection," *IEEE Transactions on Communication Technology*, vol. 15, no. 1, pp. 52-60, 1967.
- J. Lin, "Divergence Measures Based on the Shannon Entropy," *IEEE Transactions on Information Theory*, vol. 37, no. 1, pp. 145-151, 1991.

### Public datasets

These datasets are not included in or distributed with SymLoc. To use one, download it yourself and keep it outside the project.

- J. Torres-Sospedra et al., "UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems," *Proc. International Conference on Indoor Positioning and Indoor Navigation (IPIN)*, 2014, pp. 261-270.
- "Wireless Indoor Localization" data set, UCI Machine Learning Repository.

---

## License

MIT, see [LICENSE](LICENSE). Author: RL.
