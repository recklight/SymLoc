"""Typed, validated configuration for every tunable in the project.

Nothing in this package reads a constant out of the source.  The candidate
grid, the access points, their standard deviations, which of them are
tampered with and by how much, the orders of :math:`M` to evaluate, the
normalisation mode, the Monte-Carlo settings, the cross-validation layout and
the output locations all live in one of the dataclasses defined here.  A
configuration can be loaded from YAML, overridden with dotted keys from the
command line, and written back out beside the results, so a run can always be
rebuilt from its own artefacts.

That is the direct answer to the most damaging gap in the original work: the
archived figures carry filenames claiming parameters that do not match the
curves inside them, and no record of the real settings survives anywhere.  Here
the settings are the input, and :mod:`symloc.io_` writes them next to every
figure it produces.

Examples
--------
>>> cfg = Config()
>>> cfg.grid.step_db
0.5
>>> cfg.override({"grid.step_db": 0.25}).grid.step_db
0.25
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .ap import APSpec, Scenario
from .combiners import available_combiners
from .grid import Grid
from .normalize import NORMALIZE_MODES
from .pathloss import pathloss_radio_map

_LOGGER = logging.getLogger(__name__)

#: Radio map kinds accepted by :class:`RadioMapConfig`.
RADIO_MAP_KINDS: tuple[str, ...] = ("uniform", "pathloss")

#: Classifier names accepted by :class:`DetectionConfig`.
DETECTOR_MODELS: tuple[str, ...] = ("logistic", "svm", "random_forest", "mlp")

#: Output formats accepted by :class:`OutputConfig`.
FIGURE_FORMATS: tuple[str, ...] = ("pdf", "png", "svg")

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigError(ValueError):
    """Raised when a configuration value is missing, unknown or out of range."""


def _check_choice(name: str, value: str, allowed: Sequence[str]) -> None:
    """Validate that ``value`` is one of ``allowed``, naming the alternatives."""
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; choose one of {list(allowed)}")


def _check_positive(name: str, value: float) -> None:
    """Validate that a numeric entry is strictly positive."""
    if not value > 0:
        raise ConfigError(f"{name} must be strictly positive, got {value!r}")


@dataclass(frozen=True, slots=True)
class GridConfig:
    """The candidate grid.

    Attributes
    ----------
    low_dbm, high_dbm : float
        Inclusive endpoints of the mean-RSSI sweep, in dBm.  The defaults
        reproduce the 51-point range the original analysis used.
    step_db : float
        Spacing in dB.  Every reported peak of a ``"sum"``-normalised curve
        scales with this, which is why it is recorded in the manifest.
    """

    low_dbm: float = -85.0
    high_dbm: float = -60.0
    step_db: float = 0.5

    def __post_init__(self) -> None:
        _check_positive("grid.step_db", self.step_db)
        if not self.high_dbm > self.low_dbm:
            raise ConfigError(
                f"grid.high_dbm ({self.high_dbm}) must exceed grid.low_dbm ({self.low_dbm})"
            )

    def build(self) -> Grid:
        """Construct the :class:`~symloc.grid.Grid` this configuration describes."""
        return Grid.from_dbm_sweep(self.low_dbm, self.high_dbm, self.step_db)


@dataclass(frozen=True, slots=True)
class APConfig:
    """One access point, before any attack is applied.

    Attributes
    ----------
    name : str
        Identifier used in legends and tables.
    true_rssi_dbm : float
        What this access point reports when honest.
    sigma_db : float
        Shadowing standard deviation in dB; must be strictly positive.
    position_m : tuple of float or None
        Location on the floor plan, required only by the ``"pathloss"`` radio
        map.
    """

    name: str
    true_rssi_dbm: float = -70.0
    sigma_db: float = 8.0
    position_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ConfigError("every access point needs a non-empty name")
        _check_positive(f"ap[{self.name}].sigma_db", self.sigma_db)
        if self.position_m is not None:
            position = tuple(float(v) for v in self.position_m)
            if len(position) != 2:
                raise ConfigError(
                    f"ap[{self.name}].position_m must have two entries, got {self.position_m!r}"
                )
            object.__setattr__(self, "position_m", position)


@dataclass(frozen=True, slots=True)
class RadioMapConfig:
    """How the expected mean of each access point at each candidate is built.

    Attributes
    ----------
    kind : str
        ``"uniform"`` reproduces the original degenerate assumption, where
        every access point shares one mean per candidate and the axis is a
        signal level rather than a place.  ``"pathloss"`` gives every access
        point its own mean from its own distance, so the axis carries geometry.
    floor_width_m, floor_depth_m, spacing_m : float
        Extent and resolution of the reference-point grid, for ``"pathloss"``.
    reference_rssi_dbm : float
        Received level at ``reference_distance_m``.
    path_loss_exponent : float
        The exponent :math:`n` of the log-distance model.
    reference_distance_m, min_distance_m : float
        Reference distance and the clip applied before the logarithm.
    """

    kind: str = "uniform"
    floor_width_m: float = 20.0
    floor_depth_m: float = 15.0
    spacing_m: float = 1.0
    reference_rssi_dbm: float = -40.0
    path_loss_exponent: float = 3.0
    reference_distance_m: float = 1.0
    min_distance_m: float = 0.5

    def __post_init__(self) -> None:
        _check_choice("radio_map.kind", self.kind, RADIO_MAP_KINDS)
        for name in (
            "floor_width_m",
            "floor_depth_m",
            "spacing_m",
            "path_loss_exponent",
            "reference_distance_m",
            "min_distance_m",
        ):
            _check_positive(f"radio_map.{name}", getattr(self, name))


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Which access points exist, which are tampered with, and by how much.

    Attributes
    ----------
    name : str
        Scenario identifier, carried into filenames and result tables.
    aps : tuple of APConfig
        The access points.
    attacked_indices : tuple of int
        Zero-based indices of the tampered access points.  An empty tuple is
        the honest control, produced by the same code path as every other case.
    attack_offset_db : float
        Offset added to the honest reading of each tampered access point.
        Negative values suppress the apparent signal.
    m_values : tuple of int
        Orders of :math:`M` to evaluate.  Empty means every order from 0 to
        :math:`N-1`.
    combiner : str
        Strategy name from :func:`~symloc.combiners.available_combiners`.
    true_index : int or None
        Index of the true position on the grid; enables localization error.
    observations_from_radio_map : bool
        When a ``"pathloss"`` radio map is used together with a ``true_index``,
        take each access point's honest reading from the map at that position
        instead of from ``APConfig.true_rssi_dbm``.  Leaving this off lets a
        hand-written reading disagree with the map it is scored against, which
        would put the true position nowhere near the argmax and make the
        localization error meaningless.  It has no effect on a uniform map.
    """

    name: str = "default"
    aps: tuple[APConfig, ...] = (
        APConfig("AP1", -70.0, 8.0),
        APConfig("AP2", -70.0, 10.0),
        APConfig("AP3", -70.0, 10.0),
    )
    attacked_indices: tuple[int, ...] = (2,)
    attack_offset_db: float = -20.0
    m_values: tuple[int, ...] = ()
    combiner: str = "esp"
    true_index: int | None = None
    observations_from_radio_map: bool = True

    def __post_init__(self) -> None:
        if not self.aps:
            raise ConfigError(f"scenario {self.name!r} has no access points")
        # Access points may arrive as mappings, from YAML or from a --set
        # override; coercing here means every entry point gets the same
        # validation instead of only the YAML one.
        object.__setattr__(
            self, "aps", tuple(_as_ap_config(ap, index) for index, ap in enumerate(self.aps))
        )
        names = [ap.name for ap in self.aps]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ConfigError(
                f"scenario {self.name!r} has duplicate access point names: {sorted(duplicates)}"
            )

        n_ap = len(self.aps)
        indices = tuple(int(i) for i in self.attacked_indices)
        for index in indices:
            if not 0 <= index < n_ap:
                raise ConfigError(
                    f"scenario {self.name!r}: attacked index {index} is outside "
                    f"0..{n_ap - 1} for {n_ap} access points"
                )
        if len(set(indices)) != len(indices):
            raise ConfigError(
                f"scenario {self.name!r}: attacked_indices contains duplicates {indices}"
            )
        object.__setattr__(self, "attacked_indices", indices)

        _check_choice("scenario.combiner", self.combiner, available_combiners())

        orders = tuple(int(m) for m in self.m_values) or tuple(range(n_ap))
        for order in orders:
            if not 0 <= order <= n_ap - 1:
                raise ConfigError(
                    f"scenario {self.name!r}: M={order} is out of range for {n_ap} "
                    f"access points; M must lie in 0..{n_ap - 1} so that at least "
                    "one access point is always kept"
                )
        object.__setattr__(self, "m_values", orders)

    @property
    def n_ap(self) -> int:
        """Number of access points."""
        return len(self.aps)

    @property
    def n_attacked(self) -> int:
        """Number of tampered access points."""
        return len(self.attacked_indices)


@dataclass(frozen=True, slots=True)
class NormalizeConfig:
    """How combined scores become distributions.

    Attributes
    ----------
    mode : str
        One of :data:`~symloc.normalize.NORMALIZE_MODES`.
    warn_truncation_above : float
        Fraction of off-grid likelihood mass above which a warning is logged.
    """

    mode: str = "sum"
    warn_truncation_above: float = 0.5

    def __post_init__(self) -> None:
        _check_choice("normalize.mode", self.mode, NORMALIZE_MODES)
        if not 0 < self.warn_truncation_above <= 1:
            raise ConfigError(
                "normalize.warn_truncation_above must lie in (0, 1], "
                f"got {self.warn_truncation_above!r}"
            )


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Repeated trials with resampled observation noise.

    The original study was a single deterministic evaluation, so it could not
    state whether any difference it reported was larger than run-to-run
    variation.  These settings drive the repeated trials that answer that.

    Attributes
    ----------
    n_trials : int
        Number of independent observation vectors drawn.
    seed : int
        Seed for :func:`numpy.random.default_rng`; fixing it makes every run
        reproducible.
    confidence : float
        Confidence level of the reported interval, in ``(0, 1)``.
    """

    n_trials: int = 200
    seed: int = 20120601
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ConfigError(f"montecarlo.n_trials must be at least 1, got {self.n_trials}")
        if not 0 < self.confidence < 1:
            raise ConfigError(f"montecarlo.confidence must lie in (0, 1), got {self.confidence!r}")


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Supervised spoofing detection and its cross-validation layout.

    Attributes
    ----------
    model : str
        Classifier name from :data:`DETECTOR_MODELS`.
    n_splits : int
        Folds of the stratified cross-validation.
    n_trials_per_class : int
        Trials generated per class before splitting.
    seed : int
        Seed shared by the data generator and the fold assignment.
    attack_offsets_db : tuple of float
        Attack strengths sampled when generating the attacked class.
    """

    model: str = "logistic"
    n_splits: int = 5
    n_trials_per_class: int = 250
    seed: int = 20120601
    attack_offsets_db: tuple[float, ...] = (-30.0, -25.0, -20.0, -15.0, -10.0)

    def __post_init__(self) -> None:
        _check_choice("detection.model", self.model, DETECTOR_MODELS)
        if self.n_splits < 2:
            raise ConfigError(f"detection.n_splits must be at least 2, got {self.n_splits}")
        if self.n_trials_per_class < self.n_splits:
            raise ConfigError(
                f"detection.n_trials_per_class ({self.n_trials_per_class}) must be at "
                f"least detection.n_splits ({self.n_splits}) so every fold has a sample "
                "of each class"
            )
        offsets = tuple(float(v) for v in self.attack_offsets_db)
        if not offsets:
            raise ConfigError("detection.attack_offsets_db is empty")
        if any(offset == 0.0 for offset in offsets):
            raise ConfigError(
                "detection.attack_offsets_db contains 0.0, which would make an "
                "attacked trial identical to an honest one and the labels unlearnable"
            )
        object.__setattr__(self, "attack_offsets_db", offsets)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Axes of a multi-dimensional parameter sweep.

    Attributes
    ----------
    attack_offsets_db : tuple of float
        Attack strengths to sweep.
    sigmas_db : tuple of float
        Shadowing standard deviations; each value is applied to every access
        point.  Empty means keep the per-access-point values from the scenario.
    n_ap_values : tuple of int
        Access point counts.  Empty means keep the scenario's own count.
    m_values : tuple of int
        Orders of :math:`M`.  Empty means every valid order.
    combiners : tuple of str
        Strategies to compare.
    """

    attack_offsets_db: tuple[float, ...] = (-5.0, -10.0, -15.0, -20.0, -25.0, -30.0)
    sigmas_db: tuple[float, ...] = ()
    n_ap_values: tuple[int, ...] = ()
    m_values: tuple[int, ...] = ()
    combiners: tuple[str, ...] = ("esp",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attack_offsets_db", tuple(float(v) for v in self.attack_offsets_db)
        )
        object.__setattr__(self, "sigmas_db", tuple(float(v) for v in self.sigmas_db))
        object.__setattr__(self, "n_ap_values", tuple(int(v) for v in self.n_ap_values))
        object.__setattr__(self, "m_values", tuple(int(v) for v in self.m_values))
        object.__setattr__(self, "combiners", tuple(str(v) for v in self.combiners))
        if not self.attack_offsets_db:
            raise ConfigError("sweep.attack_offsets_db is empty")
        for sigma in self.sigmas_db:
            _check_positive("sweep.sigmas_db", sigma)
        for count in self.n_ap_values:
            if count < 2:
                raise ConfigError(
                    f"sweep.n_ap_values contains {count}; at least two access points "
                    "are needed for M to have any range"
                )
        for name in self.combiners:
            _check_choice("sweep.combiners", name, available_combiners())


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Where results are written and in what formats.

    Attributes
    ----------
    results_dir : pathlib.Path
        Root of the output tree; each run gets a sub-folder named by a hash of
        its configuration, so a filename can never disagree with the parameters
        that produced it.
    figure_formats : tuple of str
        Formats written for every figure.
    dpi : int
        Resolution of the raster formats.
    write_parquet : bool
        Whether to write Parquet beside the CSV.  Ignored with a warning when
        no Parquet engine is installed.
    """

    results_dir: Path = Path("results")
    figure_formats: tuple[str, ...] = ("pdf", "png")
    dpi: int = 300
    write_parquet: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        formats = tuple(str(f).lower() for f in self.figure_formats)
        if not formats:
            raise ConfigError("output.figure_formats is empty")
        for name in formats:
            _check_choice("output.figure_formats", name, FIGURE_FORMATS)
        object.__setattr__(self, "figure_formats", formats)
        if self.dpi < 50:
            raise ConfigError(f"output.dpi must be at least 50, got {self.dpi}")


@dataclass(frozen=True, slots=True)
class Config:
    """The whole configuration of a run.

    Examples
    --------
    >>> cfg = Config()
    >>> cfg.scenario.n_ap
    3
    >>> cfg.build_scenario().n_attacked
    1
    """

    grid: GridConfig = field(default_factory=GridConfig)
    radio_map: RadioMapConfig = field(default_factory=RadioMapConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    montecarlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.log_level.upper() not in _LOG_LEVELS:
            raise ConfigError(
                f"log_level={self.log_level!r} is not a logging level; "
                f"choose one of {sorted(_LOG_LEVELS)}"
            )
        object.__setattr__(self, "log_level", self.log_level.upper())
        if self.radio_map.kind == "pathloss":
            missing = [ap.name for ap in self.scenario.aps if ap.position_m is None]
            if missing:
                raise ConfigError(
                    "radio_map.kind='pathloss' needs a position_m for every access "
                    f"point; missing for {missing}"
                )

    # -- construction ----------------------------------------------------

    def build_grid(self) -> Grid:
        """Build the candidate grid, respecting the radio map kind.

        A ``"pathloss"`` radio map needs reference points with coordinates, so
        the grid becomes a floor plan rather than a signal-level sweep.
        """
        if self.radio_map.kind == "pathloss":
            return Grid.rectangular(
                self.radio_map.floor_width_m,
                self.radio_map.floor_depth_m,
                self.radio_map.spacing_m,
            )
        return self.grid.build()

    def build_scenario(self, attack_offset_db: float | None = None) -> Scenario:
        """Assemble the :class:`~symloc.ap.Scenario` this configuration describes.

        Parameters
        ----------
        attack_offset_db : float, optional
            Overrides ``scenario.attack_offset_db``, which is what the sweep
            driver varies.

        Returns
        -------
        Scenario
            A validated scenario.
        """
        grid = self.build_grid()
        offset = (
            self.scenario.attack_offset_db if attack_offset_db is None else float(attack_offset_db)
        )
        attacked = set(self.scenario.attacked_indices)

        radio_map = None
        honest_readings = [ap.true_rssi_dbm for ap in self.scenario.aps]
        if self.radio_map.kind == "pathloss":
            positions = np.array([ap.position_m for ap in self.scenario.aps], dtype=np.float64)
            radio_map = pathloss_radio_map(
                grid,
                positions,
                [ap.name for ap in self.scenario.aps],
                reference_rssi_dbm=self.radio_map.reference_rssi_dbm,
                path_loss_exponent=self.radio_map.path_loss_exponent,
                reference_distance_m=self.radio_map.reference_distance_m,
                min_distance_m=self.radio_map.min_distance_m,
            )
            if self.scenario.observations_from_radio_map and self.scenario.true_index is not None:
                # An honest access point reports what the map predicts where the
                # user actually is; a hand-written value here would simply
                # disagree with the map it is about to be scored against.
                honest_readings = [
                    float(value) for value in radio_map.means_dbm[:, self.scenario.true_index]
                ]
                _LOGGER.debug(
                    "honest readings taken from the radio map at index %d: %s",
                    self.scenario.true_index,
                    [round(v, 2) for v in honest_readings],
                )

        aps = tuple(
            APSpec(
                name=ap.name,
                true_rssi_dbm=honest_readings[index],
                sigma_db=ap.sigma_db,
                attacked=index in attacked,
                attack_offset_db=offset if index in attacked else 0.0,
            )
            for index, ap in enumerate(self.scenario.aps)
        )

        return Scenario.build(
            name=self.scenario.name,
            grid=grid,
            aps=aps,
            radio_map=radio_map,
            true_index=self.scenario.true_index,
        )

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dictionary of every setting, ready for YAML or JSON.

        Examples
        --------
        >>> Config().to_dict()["grid"]["step_db"]
        0.5
        """
        return _to_plain(dataclasses.asdict(self))

    def to_yaml(self) -> str:
        """Serialise to a YAML document."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save(self, path: Path) -> Path:
        """Write the configuration to ``path`` as YAML, creating parent folders."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_yaml(), encoding="utf-8")
        _LOGGER.debug("wrote configuration to %s", destination)
        return destination

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Config:
        """Build a configuration from a nested dictionary.

        Unknown keys are rejected rather than ignored, so a typo in a YAML file
        is reported instead of silently leaving a default in place.

        Raises
        ------
        ConfigError
            If a section or key is unknown, or a value has the wrong type.

        Examples
        --------
        >>> Config.from_dict({"grid": {"step_db": 0.25}}).grid.step_db
        0.25
        """
        if not isinstance(payload, dict):
            raise ConfigError(f"configuration must be a mapping, got {type(payload).__name__}")

        sections = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(payload) - set(sections)
        if unknown:
            raise ConfigError(
                f"unknown configuration section(s) {sorted(unknown)}; "
                f"expected some of {sorted(sections)}"
            )

        kwargs: dict[str, Any] = {}
        for name, value in payload.items():
            if name == "log_level":
                kwargs[name] = str(value)
            elif name == "scenario":
                kwargs[name] = _build_scenario_config(value)
            else:
                kwargs[name] = _build_section(_SECTION_TYPES[name], value, name)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load a configuration from a YAML file.

        Raises
        ------
        ConfigError
            If the file does not exist or does not contain a mapping.
        """
        source = Path(path)
        if not source.is_file():
            raise ConfigError(f"configuration file not found: {source}")
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        _LOGGER.debug("loaded configuration from %s", source)
        return cls.from_dict(payload)

    def override(self, updates: dict[str, Any]) -> Config:
        """Return a copy with dotted keys replaced, e.g. ``{"grid.step_db": 0.25}``.

        Parameters
        ----------
        updates : dict
            Mapping of dotted paths to values.

        Returns
        -------
        Config
            A new, revalidated configuration.

        Raises
        ------
        ConfigError
            If a path names an unknown section or field.

        Examples
        --------
        >>> Config().override({"normalize.mode": "density"}).normalize.mode
        'density'
        >>> Config().override({"log_level": "DEBUG"}).log_level
        'DEBUG'
        """
        nested: dict[str, dict[str, Any]] = {}
        top: dict[str, Any] = {}
        for dotted, value in updates.items():
            if "." not in dotted:
                if dotted not in {f.name for f in dataclasses.fields(self)}:
                    raise ConfigError(f"unknown configuration key {dotted!r}")
                top[dotted] = value
                continue
            section, _, key = dotted.partition(".")
            if section not in {f.name for f in dataclasses.fields(self)}:
                raise ConfigError(f"unknown configuration section {section!r} in key {dotted!r}")
            nested.setdefault(section, {})[key] = value

        changes: dict[str, Any] = dict(top)
        for section, fields_ in nested.items():
            current = getattr(self, section)
            known = {f.name for f in dataclasses.fields(current)}
            unknown = set(fields_) - known
            if unknown:
                raise ConfigError(
                    f"unknown key(s) {sorted(unknown)} for section {section!r}; "
                    f"expected some of {sorted(known)}"
                )
            changes[section] = replace(current, **fields_)
        return replace(self, **changes)


#: Section name to dataclass, used when rebuilding a configuration from a mapping.
_SECTION_TYPES: dict[str, type] = {
    "grid": GridConfig,
    "radio_map": RadioMapConfig,
    "normalize": NormalizeConfig,
    "montecarlo": MonteCarloConfig,
    "detection": DetectionConfig,
    "sweep": SweepConfig,
    "output": OutputConfig,
}


def _build_section(section_type: type, value: Any, name: str) -> Any:
    """Instantiate one configuration section from a mapping."""
    if isinstance(value, section_type):
        return value
    if not isinstance(value, dict):
        raise ConfigError(f"section {name!r} must be a mapping, got {type(value).__name__}")
    known = {f.name for f in dataclasses.fields(section_type)}
    unknown = set(value) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in section {name!r}; "
            f"expected some of {sorted(known)}"
        )
    return section_type(**value)


def _as_ap_config(entry: Any, index: int) -> APConfig:
    """Coerce one access point entry, from a mapping or an :class:`APConfig`."""
    if isinstance(entry, APConfig):
        return entry
    if not isinstance(entry, dict):
        raise ConfigError(
            f"scenario.aps[{index}] must be a mapping or an APConfig, got {type(entry).__name__}"
        )
    known = {f.name for f in dataclasses.fields(APConfig)}
    unknown = set(entry) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in scenario.aps[{index}]; "
            f"expected some of {sorted(known)}"
        )
    if "name" not in entry:
        raise ConfigError(f"scenario.aps[{index}] has no 'name'")
    fields_ = dict(entry)
    position = fields_.get("position_m")
    if position is not None:
        fields_["position_m"] = tuple(float(v) for v in position)
    return APConfig(**fields_)


def _build_scenario_config(value: Any) -> ScenarioConfig:
    """Instantiate a :class:`ScenarioConfig` from a mapping.

    The access point list is coerced by :class:`ScenarioConfig` itself, so YAML
    loading and dotted overrides share one code path.
    """
    if isinstance(value, ScenarioConfig):
        return value
    if not isinstance(value, dict):
        raise ConfigError(f"section 'scenario' must be a mapping, got {type(value).__name__}")

    known = {f.name for f in dataclasses.fields(ScenarioConfig)}
    unknown = set(value) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in section 'scenario'; "
            f"expected some of {sorted(known)}"
        )
    return ScenarioConfig(**value)


def _to_plain(value: Any) -> Any:
    """Convert dataclass output into YAML-friendly built-in types."""
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
