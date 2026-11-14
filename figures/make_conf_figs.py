"""Conference figures: suite/city query reduction and holdout calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.6,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

BLUE = "#315B8A"
TEAL = "#2A7F78"
RUST = "#B95B4A"
GRAY = "#A9B3BC"
INK = "#1F2933"


def panel_label(ax, label):
    ax.text(-0.11, 1.06, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=INK)


def grouped_bars(ax, labels, lazy, screened, title):
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, lazy, width, color=BLUE, edgecolor="white", linewidth=0.4, label="Lazy-0")
    ax.bar(x + width / 2, screened, width, color=TEAL, edgecolor="white", linewidth=0.4, label="Screened")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 75)
    ax.set_ylabel("Query reduction (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E3E8EC", linewidth=0.5)
    ax.set_axisbelow(True)


def make_results() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.72), constrained_layout=True)
    suites = ["Proxy", "Lattice", "Hybrid", "Real"]
    grouped_bars(
        axes[0],
        suites,
        [57.64, 59.16, 36.43, 48.53],
        [59.43, 63.84, 44.44, 59.87],
        "Four suites",
    )
    panel_label(axes[0], "a")
    cities = ["Denver", "Boston", "Paris"]
    grouped_bars(
        axes[1],
        cities,
        [52.98, 44.95, 47.66],
        [65.67, 52.02, 61.92],
        "Real maps",
    )
    panel_label(axes[1], "b")
    axes[0].legend(loc="upper center", ncol=2, fontsize=5.4, frameon=False, handlelength=1.2, columnspacing=0.9)
    fig.savefig(OUT / "fig_results.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_results.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def make_risk() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.72), constrained_layout=True)
    rules = ["Geo", "CP", "FW"]
    selected = np.array([896, 223, 180])
    failures = np.array([595, 0, 0])
    passes = selected - failures
    x = np.arange(len(rules))
    axes[0].bar(x, passes, 0.55, color=GRAY, edgecolor="white", linewidth=0.4, label="Holdout passes")
    axes[0].bar(x, failures, 0.55, bottom=passes, color=RUST, edgecolor="white", linewidth=0.4, label="Holdout failures")
    axes[0].set_xticks(x, rules)
    axes[0].set_ylabel("Candidates")
    axes[0].set_ylim(0, 1000)
    axes[0].set_title("Selected candidates", loc="left", fontweight="bold")

    axes[0].grid(axis="y", color="#E3E8EC", linewidth=0.5)
    axes[0].set_axisbelow(True)
    panel_label(axes[0], "a")

    rates = np.array([66.41, 0.0, 0.0])
    lower = rates - np.array([63.25, 0.0, 0.0])
    upper = np.array([69.42, 1.64, 2.03]) - rates
    axes[1].errorbar(x, rates, yerr=[lower, upper], fmt="o", color=RUST, ecolor=INK,
                     elinewidth=0.8, capsize=2.5, markersize=4)
    axes[1].set_xticks(x, rules)
    axes[1].set_ylim(-3, 78)
    axes[1].set_ylabel("Holdout failure rate (%)")
    axes[1].set_title("Observed failure rate", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#E3E8EC", linewidth=0.5)
    axes[1].set_axisbelow(True)
    axes[1].axhline(0, color=GRAY, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right", fontsize=5.4, frameon=False, handlelength=1.2)
    panel_label(axes[1], "b")
    fig.savefig(OUT / "fig_risk.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_risk.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_results()
    make_risk()
    print("figures written")
