"""The Python 3.10 gate.

The package declares ``requires-python = ">=3.10"`` but is developed on a newer
interpreter, so a passing test suite proves nothing about 3.10 on its own: a
3.11-only construct runs perfectly here and fails for anyone on Ubuntu 22.04,
Colab or a default conda environment.

Two checks close that gap without a second interpreter.  The syntax check
compiles every module with :func:`ast.parse` pinned to the 3.10 grammar, which
rejects PEP 695 generics, ``except*`` and ``type`` aliases.  The library check
scans for standard-library names that only exist in 3.11 or later, and for the
NumPy aliases removed in 2.0.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

#: Token kinds that carry prose rather than code, and so must not be scanned:
#: a comment explaining why ``datetime.UTC`` is avoided is not a use of it.
_PROSE_TOKENS = frozenset({"COMMENT", "STRING", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"})


def code_only(source: str) -> str:
    """Strip comments, docstrings and string literals, keeping executable code.

    Without this the scanners below flag the very comments that document why a
    construct is avoided, which would make the honest thing to write also the
    thing that fails the build.
    """
    pieces: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if tokenize.tok_name.get(token.type, "") in _PROSE_TOKENS:
                continue
            pieces.append(token.string)
    except (tokenize.TokenError, IndentationError):  # pragma: no cover - well-formed sources
        return source
    return "\n".join(pieces)


#: Standard-library names introduced after 3.10, with the 3.10-safe alternative.
FORBIDDEN_STDLIB: dict[str, str] = {
    "tomllib": "tomli, or read the file another way (3.11+)",
    "typing.Self": "the class name as a string, or a TypeVar (3.11+)",
    "typing.LiteralString": "str (3.11+)",
    "typing.assert_never": "raise AssertionError (3.11+)",
    "typing.Unpack": "explicit parameters (3.11+)",
    "TypeVarTuple": "a plain TypeVar (3.11+)",
    "StrEnum": "class Foo(str, Enum) (3.11+)",
    "ReprEnum": "Enum (3.11+)",
    "datetime.UTC": "datetime.timezone.utc (3.11+)",
    "asyncio.TaskGroup": "asyncio.gather (3.11+)",
    "ExceptionGroup": "a single exception (3.11+)",
    "contextlib.chdir": "os.chdir in a try/finally (3.11+)",
    "hashlib.file_digest": "read the file and hash it (3.11+)",
    "math.cbrt": "x ** (1/3) (3.11+)",
    "inspect.getmembers_static": "inspect.getmembers (3.11+)",
    "typing.override": "a plain method (3.12+)",
    "itertools.batched": "an explicit slice loop (3.12+)",
    "Path.walk": "os.walk, or Path.rglob (3.12+)",
    "typing.ReadOnly": "a plain annotation (3.13+)",
    "copy.replace": "dataclasses.replace (3.13+)",
}

#: NumPy aliases removed in 2.0, with their replacements.
FORBIDDEN_NUMPY: dict[str, str] = {
    "np.float_": "np.float64",
    "np.int0": "np.intp",
    "np.NaN": "np.nan",
    "np.Inf": "np.inf",
    "np.product": "np.prod",
    "np.alltrue": "np.all",
    "np.round_": "np.round",
    "np.msort": "np.sort",
    "np.unicode_": "np.str_",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _python_files() -> list[Path]:
    """Every Python file that ships with the project."""
    files: list[Path] = []
    for folder in ("src", "tests", "examples"):
        files.extend(sorted((PROJECT_ROOT / folder).rglob("*.py")))
    return files


def test_there_is_something_to_check():
    assert len(_python_files()) > 15


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_every_module_parses_under_the_python_310_grammar(path: Path):
    """Compiling against the 3.10 grammar rejects any newer syntax."""
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path), feature_version=(3, 10))
    except SyntaxError as exc:
        pytest.fail(
            f"{path.relative_to(PROJECT_ROOT)} does not parse as Python 3.10: "
            f"{exc.msg} (line {exc.lineno})"
        )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_module_uses_a_standard_library_name_newer_than_310(path: Path):
    """Syntax alone is not enough: a 3.11 stdlib name parses fine and fails at import."""
    source = code_only(path.read_text(encoding="utf-8"))
    if path.name == "test_python310_compat.py":
        pytest.skip("this file names the forbidden constructs on purpose")

    offences: list[str] = []
    for name, replacement in FORBIDDEN_STDLIB.items():
        # Word boundaries so "datetime.UTC" does not match "datetime.UTCNOW".
        if re.search(rf"(?<![\w.]){re.escape(name)}\b", source):
            offences.append(f"{name} -> use {replacement}")
    assert not offences, f"{path.relative_to(PROJECT_ROOT)}: " + "; ".join(offences)


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_module_uses_a_numpy_alias_removed_in_2_0(path: Path):
    """The code must run on both NumPy 1.x and 2.x."""
    source = code_only(path.read_text(encoding="utf-8"))
    if path.name == "test_python310_compat.py":
        pytest.skip("this file names the forbidden aliases on purpose")

    offences = [
        f"{name} -> use {replacement}"
        for name, replacement in FORBIDDEN_NUMPY.items()
        if re.search(rf"(?<![\w.]){re.escape(name)}\b", source)
    ]
    assert not offences, f"{path.relative_to(PROJECT_ROOT)}: " + "; ".join(offences)


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_ndarray_ptp_method_is_not_used(path: Path):
    """``ndarray.ptp()`` was removed in NumPy 2.0; ``np.ptp(arr)`` works on both."""
    source = code_only(path.read_text(encoding="utf-8"))
    if path.name == "test_python310_compat.py":
        pytest.skip("this file names the forbidden call on purpose")
    assert not re.search(r"\)\.ptp\(|\]\.ptp\(|\w+\.ptp\(\)", source), (
        f"{path.relative_to(PROJECT_ROOT)}: use np.ptp(arr) instead of arr.ptp()"
    )


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "src").rglob("*.py")), ids=lambda p: p.name)
def test_every_module_defers_annotation_evaluation(path: Path):
    """``from __future__ import annotations`` keeps modern annotations valid on 3.10."""
    source = path.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source, (
        f"{path.relative_to(PROJECT_ROOT)} is missing the future import"
    )


def test_the_declared_python_floor_and_ruff_target_agree():
    """A mismatch here would let ruff bless syntax the metadata forbids."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in text
    assert 'target-version = "py310"' in text


def test_dependency_floors_are_reachable_from_python_310():
    """Pinning numpy>=2.0 would exclude the interpreters this package targets."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for requirement in (
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.6",
        "pandas>=1.5",
    ):
        assert requirement in text, requirement
    assert "numpy>=2" not in text


def test_torch_is_optional_and_never_imported_at_module_scope():
    """The package must import and test cleanly with no torch installed."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dl = ["torch>=2.2"]' in text

    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import torch", "from torch")):
                assert line.startswith(" "), (
                    f"{path.relative_to(PROJECT_ROOT)} imports torch at module scope; "
                    "it must be imported lazily inside a function"
                )


def test_the_package_imports_without_torch():
    """Nothing in the public API may require the optional deep-learning extra."""
    import symloc

    assert symloc.__version__ == "0.1.0"
    assert "torch" not in str(symloc.__doc__ or "")
