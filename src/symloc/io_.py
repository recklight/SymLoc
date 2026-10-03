"""Writing results down, with enough provenance to rebuild them.

The original study produced its figures by hand: constants were edited at the
top of a script, the script was run, and the figure was saved from a menu.
Nothing recorded which constants produced which file, and the archived
filenames do not match the curves inside them.  That is not a small
inconvenience -- it means the archived figures cannot be regenerated from the
surviving code at all.

Every run here writes a folder named after a hash of its own configuration,
containing the tables, the figures, the exact configuration as YAML and a
manifest listing the package version, the interpreter, the library versions,
the random seed and the timestamp.  A result can therefore always be traced
back to the parameters that produced it, and a filename can never disagree with
its contents, because the filename is derived from them.

Tables are written as CSV always and as Parquet when an engine is available,
because CSV is readable without any tooling while Parquet keeps the dtypes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config

_LOGGER = logging.getLogger(__name__)

#: Length of the configuration hash used in run folder names.
HASH_LENGTH = 12


class ResultIOError(OSError):
    """Raised when results cannot be written."""


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Folders of one run.

    Attributes
    ----------
    root : pathlib.Path
        Run folder, named ``<label>-<config hash>``.
    figures, tables : pathlib.Path
        Sub-folders for figures and tables.
    """

    root: Path
    figures: Path
    tables: Path

    def ensure(self) -> RunPaths:
        """Create every folder, returning self for chaining."""
        for folder in (self.root, self.figures, self.tables):
            folder.mkdir(parents=True, exist_ok=True)
        return self


def config_hash(config: Config, length: int = HASH_LENGTH) -> str:
    """Stable short hash of a configuration.

    Derived from the canonical JSON form of the configuration, so two runs with
    the same settings land in the same folder and two runs with different
    settings never collide.

    Parameters
    ----------
    config : Config
        The configuration to fingerprint.
    length : int, optional
        Number of hex characters to keep.

    Returns
    -------
    str
        Lower-case hex digest prefix.

    Examples
    --------
    >>> from symloc.config import Config
    >>> a, b = config_hash(Config()), config_hash(Config())
    >>> a == b, len(a)
    (True, 12)
    >>> config_hash(Config().override({"grid.step_db": 0.25})) != a
    True
    """
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]


def prepare_run(config: Config, label: str = "run") -> RunPaths:
    """Create the folder tree for a run and return its paths.

    Parameters
    ----------
    config : Config
        Settings; ``config.output.results_dir`` is the root.
    label : str, optional
        Human-readable prefix of the folder name.

    Returns
    -------
    RunPaths
        Created folders.

    Examples
    --------
    >>> import tempfile
    >>> from symloc.config import Config
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     cfg = Config().override({"output.results_dir": folder})
    ...     paths = prepare_run(cfg, "demo")
    ...     paths.root.name.startswith("demo-"), paths.tables.is_dir()
    (True, True)
    """
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label) or "run"
    root = Path(config.output.results_dir) / f"{safe_label}-{config_hash(config)}"
    paths = RunPaths(root=root, figures=root / "figures", tables=root / "tables").ensure()
    _LOGGER.debug("run folder: %s", paths.root)
    return paths


def write_table(
    frame: pd.DataFrame,
    path: Path,
    write_parquet: bool = True,
) -> tuple[Path, ...]:
    """Write a table as CSV, and as Parquet when an engine is installed.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to write.
    path : pathlib.Path
        Destination without a suffix; the suffixes are added here.
    write_parquet : bool, optional
        Whether to attempt Parquet alongside the CSV.

    Returns
    -------
    tuple of pathlib.Path
        Files actually written.

    Examples
    --------
    >>> import tempfile, pandas as pd
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     written = write_table(pd.DataFrame({"a": [1]}),
    ...                           Path(folder) / "t", write_parquet=False)
    ...     [p.suffix for p in written]
    ['.csv']
    """
    destination = Path(path).with_suffix("")
    destination.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    csv_path = destination.with_suffix(".csv")
    try:
        frame.to_csv(csv_path, index=False, encoding="utf-8")
    except OSError as exc:
        raise ResultIOError(f"could not write {csv_path}: {exc}") from exc
    written.append(csv_path)

    if write_parquet:
        parquet_path = destination.with_suffix(".parquet")
        try:
            frame.to_parquet(parquet_path, index=False)
            written.append(parquet_path)
        except (ImportError, ValueError) as exc:
            _LOGGER.warning(
                "no Parquet engine available (%s); wrote %s only. "
                "Install the 'parquet' extra to keep dtypes.",
                exc,
                csv_path.name,
            )

    _LOGGER.debug("wrote %s", ", ".join(str(p) for p in written))
    return tuple(written)


def write_manifest(
    path: Path,
    config: Config,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the run manifest: settings, versions, seed and timestamp.

    Parameters
    ----------
    path : pathlib.Path
        Destination JSON file.
    config : Config
        Settings, embedded in full.
    extra : dict, optional
        Additional provenance, such as the caveats a scenario produced.

    Returns
    -------
    pathlib.Path
        The file written.

    Examples
    --------
    >>> import json, tempfile
    >>> from symloc.config import Config
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     written = write_manifest(Path(folder) / "manifest.json", Config())
    ...     payload = json.loads(written.read_text(encoding="utf-8"))
    ...     sorted(payload)[:3]
    ['config', 'config_hash', 'created_utc']
    """
    from . import __version__

    manifest: dict[str, Any] = {
        # timezone.utc rather than datetime.UTC, which is 3.11+ only.
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package": "symloc",
        "version": __version__,
        "config_hash": config_hash(config),
        "config": config.to_dict(),
        "seeds": {
            "montecarlo": config.montecarlo.seed,
            "detection": config.detection.seed,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    try:
        import scipy

        manifest["environment"]["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        pass
    try:
        import sklearn

        manifest["environment"]["scikit-learn"] = sklearn.__version__
    except ImportError:  # pragma: no cover - scikit-learn is a hard dependency
        pass

    if extra:
        manifest["notes"] = _jsonable(extra)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ResultIOError(f"could not write {destination}: {exc}") from exc
    _LOGGER.info("wrote run manifest to %s", destination)
    return destination


def save_figure(figure: Any, path: Path, config: Config) -> tuple[Path, ...]:
    """Save a matplotlib figure in every configured format.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    path : pathlib.Path
        Destination without a suffix.
    config : Config
        Settings; supplies the formats and the raster resolution.

    Returns
    -------
    tuple of pathlib.Path
        Files written.
    """
    destination = Path(path).with_suffix("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in config.output.figure_formats:
        target = destination.with_suffix(f".{suffix}")
        figure.savefig(target, dpi=config.output.dpi, bbox_inches="tight")
        written.append(target)
    _LOGGER.debug("wrote figure %s", destination.name)
    return tuple(written)


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays into JSON-serialisable built-ins."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
