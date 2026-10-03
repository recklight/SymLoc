"""Shared fixtures.

Every fixture here is synthetic: arrays built in memory from known constants.
Nothing reads a recording, a radio map or any other measurement, so the suite
runs anywhere and no dataset is ever copied into the repository.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Figures are written to files in these tests, never displayed.
matplotlib.use("Agg")

from symloc.ap import APSpec, Scenario
from symloc.config import Config
from symloc.grid import Grid

#: Root of the checkout, for the read-only legacy reference values.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Root of the checkout, so tests can reach configs/ and data/."""
    return _PROJECT_ROOT


@pytest.fixture
def grid() -> Grid:
    """The 51-point sweep the original analysis used."""
    return Grid.from_dbm_sweep(-85.0, -60.0, 0.5)


@pytest.fixture
def fine_grid() -> Grid:
    """A finely sampled wide grid, for comparing against analytic results."""
    return Grid.from_dbm_sweep(-200.0, 200.0, 0.05)


@pytest.fixture
def three_ap_scenario(grid: Grid) -> Scenario:
    """Three access points, the last one tampered with by -20 dB."""
    aps = [
        APSpec("AP1", -70.0, 8.0),
        APSpec("AP2", -70.0, 10.0),
        APSpec("AP3", -70.0, 10.0, attacked=True, attack_offset_db=-20.0),
    ]
    return Scenario.build("three_ap", grid, aps)


@pytest.fixture
def config() -> Config:
    """A small default configuration, fast enough for the whole suite."""
    return Config().override(
        {
            "montecarlo.n_trials": 12,
            "detection.n_trials_per_class": 40,
            "detection.n_splits": 3,
        }
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so every test that samples is reproducible."""
    return np.random.default_rng(20120601)


@pytest.fixture
def random_log_likelihoods(rng: np.random.Generator):
    """Factory for well-conditioned random log-likelihood blocks."""

    def _make(n_ap: int = 4, n_points: int = 9) -> np.ndarray:
        return np.log(rng.uniform(0.05, 2.0, size=(n_ap, n_points)))

    return _make
