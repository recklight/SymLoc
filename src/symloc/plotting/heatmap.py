"""Heatmaps over the sweep grid, and the classification diagnostics.

A heatmap of attack strength against :math:`M` answers in one picture the
question the original workflow could only approach one hand-edited run at a
time: over what range of attack strengths does tolerating more tampered access
points actually buy anything?  A single scenario cannot show that; a sweep can.

The ROC curve and confusion matrix belong here too, because they describe the
detection experiment the original work implied but never ran.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .style import curve_color, style_context

_LOGGER = logging.getLogger(__name__)

#: Sequential colormap used for every heatmap; perceptually uniform and
#: readable in greyscale.
SEQUENTIAL_CMAP = "viridis"


def plot_discriminability_heatmap(
    table: pd.DataFrame,
    metric_name: str = "Jensen-Shannon divergence",
    annotate: bool = True,
) -> Figure:
    """Heatmap of a metric over attack strength and :math:`M`.

    Parameters
    ----------
    table : pandas.DataFrame
        A pivot from :func:`~symloc.sweep.discriminability_pivot`: rows indexed by
        attack offset, columns by :math:`M`.
    metric_name : str, optional
        Name shown on the colour bar.
    annotate : bool, optional
        Whether to print each cell's value.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from symloc.config import Config
    >>> from symloc.sweep import run_sweep, discriminability_pivot
    >>> cfg = Config().override({"sweep.attack_offsets_db": (-10.0, -20.0)})
    >>> figure = plot_discriminability_heatmap(discriminability_pivot(run_sweep(cfg), "peak"))
    >>> len(figure.axes)
    2
    >>> import matplotlib.pyplot as plt; plt.close(figure)
    """
    values = table.to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("the pivot table is empty; nothing to plot")

    with style_context():
        size = (1.4 * table.shape[1] + 3.2, 0.5 * table.shape[0] + 2.4)
        figure, axes = plt.subplots(figsize=size)
        image = axes.imshow(values, aspect="auto", cmap=SEQUENTIAL_CMAP, origin="lower")

        axes.set_xticks(np.arange(table.shape[1]))
        axes.set_xticklabels([f"M={c}" for c in table.columns])
        axes.set_yticks(np.arange(table.shape[0]))
        axes.set_yticklabels([f"{v:+.0f} dB" for v in table.index])
        axes.set_xlabel("M (number of tampered APs tolerated)")
        axes.set_ylabel("attack offset")
        axes.set_title(metric_name)
        axes.grid(False)

        if annotate and np.any(np.isfinite(values)):
            # Some metrics are legitimately nan across a whole sweep -- the full
            # width at half maximum is nan wherever the curve never falls back
            # below half its peak -- and nanmin of an all-nan slice warns.
            finite = values[np.isfinite(values)]
            midpoint = float(finite.min() + 0.5 * (finite.max() - finite.min()))
            for row in range(table.shape[0]):
                for column in range(table.shape[1]):
                    value = values[row, column]
                    if not np.isfinite(value):
                        continue
                    axes.text(
                        column,
                        row,
                        f"{value:.4g}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value < midpoint else "black",
                    )

        figure.colorbar(image, ax=axes, label=metric_name)
    return figure


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: tuple[str, str] = ("honest", "attacked"),
    title: str = "Spoofing detection, pooled over folds",
) -> Figure:
    """Confusion matrix of the detection experiment, with counts and row shares.

    Parameters
    ----------
    matrix : array_like
        Shape ``(2, 2)``, rows true and columns predicted.
    class_names : tuple of str, optional
        Row and column labels.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> import matplotlib, numpy as np
    >>> matplotlib.use("Agg")
    >>> figure = plot_confusion_matrix(np.array([[8, 2], [1, 9]]))
    >>> len(figure.axes)
    1
    >>> import matplotlib.pyplot as plt; plt.close(figure)
    """
    counts = np.asarray(matrix, dtype=float)
    if counts.shape != (2, 2):
        raise ValueError(f"confusion matrix must be 2x2, got {counts.shape}")
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    with style_context():
        figure, axes = plt.subplots(figsize=(4.6, 4.0))
        axes.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)
        for row in range(2):
            for column in range(2):
                axes.text(
                    column,
                    row,
                    f"{int(counts[row, column])}\n{shares[row, column]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if shares[row, column] > 0.5 else "black",
                )
        axes.set_xticks([0, 1])
        axes.set_xticklabels(class_names)
        axes.set_yticks([0, 1])
        axes.set_yticklabels(class_names)
        axes.set_xlabel("predicted")
        axes.set_ylabel("true")
        axes.set_title(title)
        axes.grid(False)
    return figure


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, label: str = "detector") -> Figure:
    """ROC curve of the spoofing detector against the chance diagonal.

    Parameters
    ----------
    y_true : array_like
        True labels, 0 for honest and 1 for tampered.
    y_score : array_like
        Scores for the positive class.
    label : str, optional
        Legend entry.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> import matplotlib, numpy as np
    >>> matplotlib.use("Agg")
    >>> figure = plot_roc_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.6, 0.9]))
    >>> len(figure.axes)
    1
    >>> import matplotlib.pyplot as plt; plt.close(figure)
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    true = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    false_positive, true_positive, _ = roc_curve(true, score)
    area = roc_auc_score(true, score)

    with style_context():
        figure, axes = plt.subplots(figsize=(4.8, 4.4))
        axes.plot(
            false_positive,
            true_positive,
            color=curve_color(0),
            label=f"{label} (AUC={area:.3f})",
        )
        axes.plot([0, 1], [0, 1], linestyle=":", color="#888888", label="chance")
        axes.set_xlabel("false positive rate (1 - specificity)")
        axes.set_ylabel("true positive rate (sensitivity)")
        axes.set_title("Spoofing detection ROC")
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1.02)
        axes.legend(loc="lower right")
    return figure
