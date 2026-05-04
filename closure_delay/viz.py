from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.figsize": (10, 6),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
        }
    )


def plot_scatter_with_regression(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
    hue: Sequence[float] | None = None,
):
    set_plot_style()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig, ax = plt.subplots()
    if hue is not None:
        hue = np.asarray(hue, dtype=float)
        pos = hue > 0.5
        neg = ~pos
        ax.scatter(x[pos], y[pos], c="#2c7bb6", alpha=0.7, s=30, label="Correct")
        ax.scatter(x[neg], y[neg], c="#d7191c", alpha=0.7, s=30, label="Incorrect")
        ax.legend(fontsize=9)
    else:
        ax.scatter(x, y, c="#2c7bb6", alpha=0.7, s=30)

    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() >= 3 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
        from scipy import stats

        slope, intercept, r, p, _ = stats.linregress(x[valid], y[valid])
        xs_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(xs_line, slope * xs_line + intercept, "k--", linewidth=1.2, alpha=0.7)
        ax.text(
            0.05,
            0.95,
            f"r={r:.3f}  p={p:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_closure_curves(
    curves_by_condition: Dict[str, Dict],
    output_path: str,
):
    set_plot_style()
    colors = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3", "#fb9a99"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (condition, payload) in enumerate(curves_by_condition.items()):
        curve = payload.get("attacked_risk_curve") if condition != "baseline" else payload.get("baseline_risk_curve")
        if not curve:
            continue
        fractions = np.asarray(curve.get("fractions", []), dtype=float)
        means = np.asarray(curve.get("means", []), dtype=float)
        stds = np.asarray(curve.get("stds", np.zeros_like(means)), dtype=float)
        counts = np.asarray(curve.get("counts", np.ones_like(means)), dtype=float)
        if len(fractions) == 0:
            continue
        se = stds / np.sqrt(np.maximum(counts, 1.0))
        color = colors[idx % len(colors)]
        ax.plot(fractions, means, marker="o", linewidth=1.8, color=color, label=condition)
        ax.fill_between(fractions, means - 1.96 * se, means + 1.96 * se, color=color, alpha=0.12)

    ax.set_xlabel("Baseline Token Progress")
    ax.set_ylabel("Closure Risk")
    ax.set_title("Closure Risk Curves")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
