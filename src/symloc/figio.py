"""Read-only extraction of plotted data from archived MATLAB ``.fig`` files.

A ``.fig`` file is a MAT-file version 5 container holding one variable,
``hgS_070000``, which is a handle-graphics tree: every node carries a ``type``,
a ``properties`` struct and a list of ``children``.  Line objects keep their
plotted data in ``properties.XData`` and ``properties.YData`` and their legend
entry in ``properties.DisplayName``, and text objects keep axis labels in
``properties.String``.  All of it is reachable with :func:`scipy.io.loadmat`.

This matters because an archived figure is sometimes the only surviving record
of what a run actually computed.  When the filename claims one parameter set
and the curves inside imply another, the curves are the evidence.  Everything
here is read-only and takes a path the user supplies: no figure, and no data of
any kind, ships with this package.

One trap is handled explicitly.  A file saved through the wrong menu entry can
carry a ``.fig`` extension while actually containing Encapsulated PostScript,
at which point :func:`scipy.io.loadmat` fails with an opaque message about an
unknown MAT-file version.  :func:`read_fig` checks the magic bytes first and
says what the file really is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

#: Leading bytes of an Encapsulated PostScript file.
_EPS_MAGIC = b"%!PS"

#: Leading text of a MAT-file version 5 container, which is what a .fig is.
_MAT5_MAGIC = b"MATLAB 5.0 MAT-file"

#: Name of the single variable a MATLAB figure stores its object tree under.
_FIG_ROOT_VARIABLE = "hgS_070000"

#: Bytes of the header that :func:`detect_format` inspects.  The MAT-file
#: description field is 116 bytes, so this covers it with room to spare.
_HEADER_BYTES = 128


class FigFormatError(ValueError):
    """Raised when a file is not the MATLAB figure it claims to be."""


@dataclass(frozen=True, slots=True, eq=False)
class FigCurve:
    """One plotted line recovered from a figure.

    Attributes
    ----------
    display_name : str
        Legend entry, empty when the line had none.
    x, y : numpy.ndarray
        The plotted data.
    """

    display_name: str
    x: np.ndarray
    y: np.ndarray

    @property
    def peak(self) -> float:
        """Largest ``y`` value of the curve."""
        return float(np.max(self.y))

    def __len__(self) -> int:
        return int(self.y.size)


@dataclass(frozen=True, slots=True, eq=False)
class FigDocument:
    """Everything recovered from one archived figure.

    Attributes
    ----------
    path : pathlib.Path
        Where it was read from.
    curves : tuple of FigCurve
        Plotted lines, in the order the figure stored them.
    texts : tuple of str
        Strings of every text object, which is where axis labels live.
    """

    path: Path
    curves: tuple[FigCurve, ...] = field(default=())
    texts: tuple[str, ...] = field(default=())

    def peaks(self) -> dict[str, float]:
        """Peak of every named curve, keyed by legend entry.

        Unnamed curves are keyed by their position, so nothing is silently
        dropped when a figure has lines that were drawn without a name -- the
        usual sign of something added by hand at the command prompt rather than
        by the script that made the figure.
        """
        out: dict[str, float] = {}
        for index, curve in enumerate(self.curves):
            key = curve.display_name or f"curve_{index}"
            out[key] = curve.peak
        return out

    def summary(self) -> str:
        """Human-readable description, for the ``inspect-fig`` command."""
        lines = [f"{self.path.name}: {len(self.curves)} curve(s)"]
        for index, curve in enumerate(self.curves):
            name = curve.display_name or "(unnamed)"
            lines.append(
                f"  [{index}] {name}: {len(curve)} points, "
                f"x in [{curve.x.min():.4g}, {curve.x.max():.4g}], "
                f"peak {curve.peak:.7g}"
            )
        if self.texts:
            lines.append(f"  text: {', '.join(self.texts)}")
        return "\n".join(lines)


def detect_format(path: Path) -> str:
    """Identify a file from its leading bytes, ignoring the extension.

    Parameters
    ----------
    path : pathlib.Path
        File to inspect.

    Returns
    -------
    str
        ``"mat5"``, ``"eps"`` or ``"unknown"``.

    Raises
    ------
    FigFormatError
        If the file does not exist or cannot be read.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     sample = pathlib.Path(folder) / "sample.fig"
    ...     _ = sample.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0")
    ...     detect_format(sample)
    'eps'
    """
    source = Path(path)
    if not source.is_file():
        raise FigFormatError(f"file not found: {source}")
    try:
        # Only the header is needed, and a figure can be tens of megabytes, so
        # read the first bytes rather than the whole file.
        with source.open("rb") as handle:
            header = handle.read(_HEADER_BYTES)
    except OSError as exc:
        raise FigFormatError(f"could not read {source}: {exc}") from exc
    if header.startswith(_EPS_MAGIC):
        return "eps"
    if _MAT5_MAGIC in header:
        return "mat5"
    return "unknown"


def read_fig(path: Path) -> FigDocument:
    """Recover every plotted curve and text string from a MATLAB figure.

    Parameters
    ----------
    path : pathlib.Path
        Path to a ``.fig`` file the user supplies.

    Returns
    -------
    FigDocument
        The recovered curves and texts.

    Raises
    ------
    FigFormatError
        If the file is missing, is Encapsulated PostScript wearing a ``.fig``
        extension, or contains no handle-graphics tree.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     sample = pathlib.Path(folder) / "wrong.fig"
    ...     _ = sample.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\\n")
    ...     try:
    ...         read_fig(sample)
    ...     except FigFormatError as exc:
    ...         print("Encapsulated PostScript" in str(exc))
    True
    """
    source = Path(path)
    kind = detect_format(source)
    if kind == "eps":
        raise FigFormatError(
            f"{source} is Encapsulated PostScript, not a MATLAB figure, despite its "
            "extension; it was most likely saved through the wrong export menu. "
            "Open it with a vector graphics viewer instead -- the plotted values "
            "cannot be recovered from it numerically."
        )
    if kind != "mat5":
        raise FigFormatError(
            f"{source} is not a MAT-file version 5 container, so it is not a "
            "MATLAB figure this reader can open"
        )

    from scipy.io import loadmat

    try:
        payload = loadmat(str(source), squeeze_me=True, struct_as_record=False)
    except (ValueError, NotImplementedError) as exc:
        raise FigFormatError(f"could not parse {source} as a MATLAB figure: {exc}") from exc

    root = payload.get(_FIG_ROOT_VARIABLE)
    if root is None:
        available = [key for key in payload if not key.startswith("__")]
        raise FigFormatError(
            f"{source} contains no {_FIG_ROOT_VARIABLE!r} tree; "
            f"it holds {available} instead, so it is a data MAT-file rather than a figure"
        )

    curves: list[FigCurve] = []
    texts: list[str] = []
    _walk(root, curves, texts)
    _LOGGER.info("read %d curve(s) from %s", len(curves), source)
    return FigDocument(path=source, curves=tuple(curves), texts=tuple(texts))


def _walk(node: Any, curves: list[FigCurve], texts: list[str]) -> None:
    """Depth-first traversal of the handle-graphics tree."""
    if isinstance(node, np.ndarray) and node.dtype == object:
        for item in node.ravel():
            _walk(item, curves, texts)
        return
    if not hasattr(node, "type"):
        return

    node_type = str(getattr(node, "type", ""))
    properties = getattr(node, "properties", None)

    if node_type == "graph2d.lineseries" or node_type == "line":
        x = _get(properties, "XData")
        y = _get(properties, "YData")
        if x is not None and y is not None:
            x_array = np.atleast_1d(np.asarray(x, dtype=np.float64))
            y_array = np.atleast_1d(np.asarray(y, dtype=np.float64))
            if x_array.size == y_array.size and y_array.size > 0:
                curves.append(
                    FigCurve(
                        display_name=str(_get(properties, "DisplayName") or ""),
                        x=x_array,
                        y=y_array,
                    )
                )
    elif node_type == "text":
        value = _get(properties, "String")
        if value is not None:
            text = " ".join(np.atleast_1d(np.asarray(value, dtype=object)).astype(str))
            if text.strip():
                texts.append(text.strip())

    children = getattr(node, "children", None)
    if children is not None:
        _walk(children, curves, texts)


def _get(properties: Any, name: str) -> Any:
    """Read one property, returning ``None`` when the object does not carry it."""
    if properties is None:
        return None
    return getattr(properties, name, None)


def compare_to_reference(
    document: FigDocument,
    expected_peaks: dict[str, float],
    tolerance: float = 1e-5,
) -> dict[str, dict[str, float]]:
    """Check an archived figure's curve peaks against expected values.

    Used to confirm which parameter set actually produced an archived figure,
    when the filename and the code disagree.

    Parameters
    ----------
    document : FigDocument
        A figure read by :func:`read_fig`.
    expected_peaks : dict of str to float
        Expected peak keyed by legend entry.
    tolerance : float, optional
        Absolute tolerance for the match.

    Returns
    -------
    dict
        Per key: ``expected``, ``found`` and ``matches``.

    Examples
    --------
    >>> import numpy as np
    >>> doc = FigDocument(Path("demo.fig"), (FigCurve("ESP (M=0)", np.arange(3.0),
    ...                                               np.array([0.1, 0.2, 0.1])),))
    >>> compare_to_reference(doc, {"ESP (M=0)": 0.2})["ESP (M=0)"]["matches"]
    1.0
    """
    found = document.peaks()
    report: dict[str, dict[str, float]] = {}
    for key, expected in expected_peaks.items():
        actual = found.get(key, float("nan"))
        report[key] = {
            "expected": float(expected),
            "found": float(actual),
            "matches": float(bool(np.isfinite(actual) and abs(actual - expected) <= tolerance)),
        }
    return report
