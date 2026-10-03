"""Access points, their observations, and the radio map they are scored against.

Three objects carry everything a likelihood needs:

:class:`APSpec`
    One access point: what it reports, how noisy it is, and whether an attacker
    has tampered with the report.  An attacked access point is described by an
    *offset* applied to the honest value rather than by a hard-coded observation,
    so attack strength becomes a parameter that can be swept instead of a
    constant that has to be edited in the source.

:class:`RadioMap`
    The mean RSSI each access point is expected to show at each candidate
    position, shape ``(n_ap, n_points)``.  This is the fingerprint database.

:class:`Scenario`
    A grid, a tuple of access points and a radio map, checked against each
    other once so that every downstream function can assume they agree.

The legacy analysis had no radio map worth the name: every access point was
assumed to share a single mean level at a given candidate, which is what
:func:`uniform_radio_map` reproduces.  :func:`symloc.pathloss.pathloss_radio_map`
builds the non-degenerate alternative, where each access point has its own mean
at each point because it sits somewhere specific on the floor.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .grid import Grid

_LOGGER = logging.getLogger(__name__)


class APError(ValueError):
    """Raised when an access point or a scenario is inconsistent."""


@dataclass(frozen=True, slots=True)
class APSpec:
    """One access point: its honest reading, its noise, and any tampering.

    Attributes
    ----------
    name : str
        Short identifier used in tables, legends and log messages.
    true_rssi_dbm : float
        The value an honest access point reports at the true position.
    sigma_db : float
        Standard deviation of the log-normal shadowing, in dB.  Must be
        strictly positive: a zero here would make the Gaussian likelihood a
        delta function and every combiner degenerate.
    attacked : bool, optional
        Whether this access point has been tampered with.
    attack_offset_db : float, optional
        Offset added to ``true_rssi_dbm`` when ``attacked`` is true.  Negative
        values model an attacker suppressing the apparent signal, which is the
        case the original analysis studied.

    Examples
    --------
    >>> honest = APSpec("AP1", true_rssi_dbm=-70.0, sigma_db=8.0)
    >>> honest.observed_rssi_dbm
    -70.0
    >>> spoofed = APSpec("AP3", -70.0, 10.0, attacked=True, attack_offset_db=-20.0)
    >>> spoofed.observed_rssi_dbm
    -90.0
    """

    name: str
    true_rssi_dbm: float
    sigma_db: float
    attacked: bool = False
    attack_offset_db: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.name):
            raise APError("access point name must be a non-empty string")
        if not np.isfinite(self.true_rssi_dbm):
            raise APError(f"{self.name}: true_rssi_dbm must be finite, got {self.true_rssi_dbm!r}")
        if not self.sigma_db > 0:
            raise APError(
                f"{self.name}: sigma_db must be strictly positive, got {self.sigma_db!r}; "
                "a zero standard deviation collapses the Gaussian likelihood to a delta"
            )
        if not np.isfinite(self.sigma_db):
            raise APError(f"{self.name}: sigma_db must be finite, got {self.sigma_db!r}")
        if not np.isfinite(self.attack_offset_db):
            raise APError(f"{self.name}: attack_offset_db must be finite")

    @property
    def observed_rssi_dbm(self) -> float:
        """The value actually fed to the likelihood, tampering included."""
        offset = self.attack_offset_db if self.attacked else 0.0
        return float(self.true_rssi_dbm + offset)

    def with_attack(self, offset_db: float) -> APSpec:
        """Return a copy of this access point under an attack of ``offset_db``.

        Examples
        --------
        >>> APSpec("AP1", -70.0, 8.0).with_attack(-25.0).observed_rssi_dbm
        -95.0
        """
        return replace(self, attacked=True, attack_offset_db=float(offset_db))

    def without_attack(self) -> APSpec:
        """Return an honest copy of this access point.

        Examples
        --------
        >>> APSpec("AP1", -70.0, 8.0, attacked=True, attack_offset_db=-20).without_attack().attacked
        False
        """
        return replace(self, attacked=False, attack_offset_db=0.0)


@dataclass(frozen=True, slots=True, eq=False)
class RadioMap:
    """Expected mean RSSI of every access point at every candidate position.

    Attributes
    ----------
    means_dbm : numpy.ndarray
        Shape ``(n_ap, n_points)``, in dBm.
    ap_names : tuple of str
        Names in the same order as the rows of ``means_dbm``.

    Examples
    --------
    >>> import numpy as np
    >>> rm = RadioMap(np.array([[-60.0, -70.0], [-65.0, -75.0]]), ("AP1", "AP2"))
    >>> rm.n_ap, rm.n_points
    (2, 2)
    """

    means_dbm: np.ndarray
    ap_names: tuple[str, ...]

    def __post_init__(self) -> None:
        means = np.asarray(self.means_dbm, dtype=np.float64)
        if means.ndim != 2:
            raise APError(f"means_dbm must be 2-D (n_ap, n_points), got shape {means.shape}")
        if not np.all(np.isfinite(means)):
            raise APError("means_dbm must be finite everywhere")
        names = tuple(str(n) for n in self.ap_names)
        if len(names) != means.shape[0]:
            raise APError(
                f"radio map has {means.shape[0]} rows but {len(names)} access point names"
            )
        object.__setattr__(self, "means_dbm", means)
        object.__setattr__(self, "ap_names", names)

    @property
    def n_ap(self) -> int:
        """Number of access points."""
        return int(self.means_dbm.shape[0])

    @property
    def n_points(self) -> int:
        """Number of candidate positions."""
        return int(self.means_dbm.shape[1])

    @property
    def is_degenerate(self) -> bool:
        """Whether every access point shares the same mean at every candidate.

        A degenerate radio map carries no geometric information: the candidate
        axis is a single shared signal level, so no arrangement of access
        points can be distinguished.  This is exactly the assumption the
        original analysis made, and reporting it keeps that assumption visible
        rather than implicit.

        Examples
        --------
        >>> import numpy as np
        >>> RadioMap(np.array([[-60.0, -70.0]] * 2), ("A", "B")).is_degenerate
        True
        """
        return bool(np.allclose(self.means_dbm, self.means_dbm[0], rtol=0.0, atol=1e-12))


def uniform_radio_map(grid: Grid, ap_names: Sequence[str]) -> RadioMap:
    """Build the degenerate radio map the original analysis assumed.

    Every access point takes the candidate value itself as its mean, so the
    candidate axis is one shared mean signal level in dBm.

    Parameters
    ----------
    grid : Grid
        A dBm sweep; the candidate values are used directly as means.
    ap_names : sequence of str
        Names of the access points.

    Returns
    -------
    RadioMap
        Shape ``(len(ap_names), grid.size)``, every row identical.

    Raises
    ------
    APError
        If ``grid`` is not a dBm axis, or no names are given.

    Examples
    --------
    >>> from symloc.grid import Grid
    >>> rm = uniform_radio_map(Grid.from_dbm_sweep(-85.0, -60.0, 0.5), ["A", "B", "C"])
    >>> rm.means_dbm.shape, rm.is_degenerate
    ((3, 51), True)
    """
    names = tuple(str(n) for n in ap_names)
    if not names:
        raise APError("uniform_radio_map needs at least one access point name")
    if grid.unit != "dBm":
        raise APError(
            f"uniform_radio_map expects a dBm sweep, got a grid in {grid.unit!r}; "
            "use symloc.pathloss.pathloss_radio_map for grids of reference points"
        )
    means = np.tile(grid.values, (len(names), 1))
    return RadioMap(means_dbm=means, ap_names=names)


@dataclass(frozen=True, slots=True, eq=False)
class Scenario:
    """A grid, a set of access points and a radio map, validated together.

    Attributes
    ----------
    name : str
        Identifier carried into result tables and figure filenames.
    grid : Grid
        Candidate positions.
    aps : tuple of APSpec
        Access points, in the same order as the rows of ``radio_map``.
    radio_map : RadioMap
        Expected means.
    true_index : int or None
        Index of the true position on the grid, when it is known.  Localization
        error is only reported when this is set.

    Examples
    --------
    >>> from symloc.grid import Grid
    >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
    >>> aps = [APSpec("A", -70.0, 8.0), APSpec("B", -70.0, 10.0)]
    >>> sc = Scenario.build("demo", g, aps)
    >>> sc.n_ap, sc.observed_rssi_dbm.tolist()
    (2, [-70.0, -70.0])
    """

    name: str
    grid: Grid
    aps: tuple[APSpec, ...]
    radio_map: RadioMap
    true_index: int | None = field(default=None)

    def __post_init__(self) -> None:
        if not self.aps:
            raise APError(f"scenario {self.name!r} has no access points")
        if self.radio_map.n_ap != len(self.aps):
            raise APError(
                f"scenario {self.name!r}: radio map has {self.radio_map.n_ap} access points "
                f"but {len(self.aps)} were given"
            )
        if self.radio_map.n_points != self.grid.size:
            raise APError(
                f"scenario {self.name!r}: radio map covers {self.radio_map.n_points} candidates "
                f"but the grid has {self.grid.size}"
            )
        map_names = self.radio_map.ap_names
        ap_names = tuple(ap.name for ap in self.aps)
        if map_names != ap_names:
            raise APError(
                f"scenario {self.name!r}: access point order disagrees with the radio map: "
                f"{ap_names} vs {map_names}"
            )
        # Names key every per-access-point report this package produces --
        # truncation diagnostics, result tables, legends -- so two access points
        # sharing one name would silently collapse into a single entry there.
        # The clash is rejected where it is introduced instead.
        duplicates = sorted({name for name in ap_names if ap_names.count(name) > 1})
        if duplicates:
            raise APError(
                f"scenario {self.name!r}: access point names must be unique, but "
                f"{duplicates} occur more than once"
            )
        if self.true_index is not None:
            index = int(self.true_index)
            if not 0 <= index < self.grid.size:
                raise APError(
                    f"scenario {self.name!r}: true_index {index} is outside the grid "
                    f"(0..{self.grid.size - 1})"
                )
            object.__setattr__(self, "true_index", index)

    @classmethod
    def build(
        cls,
        name: str,
        grid: Grid,
        aps: Iterable[APSpec],
        radio_map: RadioMap | None = None,
        true_index: int | None = None,
    ) -> Scenario:
        """Assemble a scenario, defaulting to the degenerate radio map.

        Parameters
        ----------
        name : str
            Scenario identifier.
        grid : Grid
            Candidate positions.
        aps : iterable of APSpec
            Access points.
        radio_map : RadioMap, optional
            Expected means.  When omitted, :func:`uniform_radio_map` is used,
            which reproduces the original assumption.
        true_index : int, optional
            Index of the true position on the grid.

        Returns
        -------
        Scenario
            A validated scenario.
        """
        ap_tuple = tuple(aps)
        if radio_map is None:
            radio_map = uniform_radio_map(grid, [ap.name for ap in ap_tuple])
        return cls(
            name=str(name),
            grid=grid,
            aps=ap_tuple,
            radio_map=radio_map,
            true_index=true_index,
        )

    @property
    def n_ap(self) -> int:
        """Number of access points."""
        return len(self.aps)

    @property
    def observed_rssi_dbm(self) -> np.ndarray:
        """Observations of every access point, shape ``(n_ap,)``, tampering included."""
        return np.array([ap.observed_rssi_dbm for ap in self.aps], dtype=np.float64)

    @property
    def sigma_db(self) -> np.ndarray:
        """Shadowing standard deviations, shape ``(n_ap,)``."""
        return np.array([ap.sigma_db for ap in self.aps], dtype=np.float64)

    @property
    def attacked_mask(self) -> np.ndarray:
        """Boolean mask of the tampered access points, shape ``(n_ap,)``."""
        return np.array([ap.attacked for ap in self.aps], dtype=bool)

    @property
    def n_attacked(self) -> int:
        """How many access points are tampered with."""
        return int(self.attacked_mask.sum())

    def honest(self) -> Scenario:
        """Return the same scenario with every attack removed.

        This is the control curve the whole argument rests on, and it is
        produced by the same code path as the attacked scenarios rather than by
        a separate hard-coded expression.

        Examples
        --------
        >>> from symloc.grid import Grid
        >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
        >>> aps = [APSpec("A", -70.0, 8.0), APSpec("B", -70.0, 10.0, attacked=True,
        ...                                        attack_offset_db=-20.0)]
        >>> Scenario.build("demo", g, aps).honest().n_attacked
        0
        """
        return replace(
            self,
            name=f"{self.name}__honest",
            aps=tuple(ap.without_attack() for ap in self.aps),
        )

    def covered_mean_span(self) -> np.ndarray:
        """Range of means the radio map predicts for each access point.

        Shape ``(n_ap, 2)`` holding ``(lowest, highest)`` mean in dBm.  This is
        the interval over which that access point's likelihood is actually
        evaluated, and it is the right reference for asking whether an
        observation falls inside the modelled range -- the grid's own span is
        in dBm only for a mean-RSSI sweep, and is a point index for a floor
        plan.

        Examples
        --------
        >>> from symloc.grid import Grid
        >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
        >>> Scenario.build("demo", g, [APSpec("A", -70.0, 8.0)]).covered_mean_span()
        array([[-85., -60.]])
        """
        means = self.radio_map.means_dbm
        return np.column_stack([means.min(axis=1), means.max(axis=1)])

    def out_of_grid_observations(self) -> tuple[str, ...]:
        """Names of access points whose observation lies outside the modelled means.

        An observation far outside the range of means the radio map predicts
        puts most of that access point's likelihood mass beyond the edge of the
        grid, so normalising on the grid silently rescales it.  The original
        analysis did exactly this and never checked; the check is reported here
        and logged by the pipeline.

        Examples
        --------
        >>> from symloc.grid import Grid
        >>> g = Grid.from_dbm_sweep(-85.0, -60.0, 0.5)
        >>> aps = [APSpec("A", -70.0, 8.0), APSpec("B", -99.0, 10.0)]
        >>> Scenario.build("demo", g, aps).out_of_grid_observations()
        ('B',)
        """
        spans = self.covered_mean_span()
        return tuple(
            ap.name
            for ap, (low, high) in zip(self.aps, spans, strict=True)
            if not low <= ap.observed_rssi_dbm <= high
        )
