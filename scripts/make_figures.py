#!/usr/bin/env python3
"""Regenerate the statistical figures of the paper from exact data.

Outputs (in figures/):
  fig_xp_density.pdf   exact density of X_p for n = 2 near p = 10^5
  fig_xp_scales.pdf    exact CDFs of X_p for n = 2 from p = 10^3 to 10^30
  fig_xp_regimes.pdf   the two regimes for n = 3 on the same primes, and the bound

Run:  python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kxn import stats  # noqa: E402
from make_numbers import DENSITY_WINDOW, PRIMES_PER_WINDOW, REGIME_WINDOW, first_primes_above  # noqa: E402

OUT = ROOT / "figures"

# Colors: one blue ramp for ordered scales, blue/orange for the two regimes.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)
XLABEL = r"$X_p=\log L_p/\log p$"


def fig_density():
    primes = stats.window(*DENSITY_WINDOW)
    d = stats.pooled(primes, 2, lambda p: 0)
    edges, h = stats.histogram(d, bins=30)
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.bar(edges[:-1], h, width=np.diff(edges) * 0.9, align="edge", color=BLUE, linewidth=0)
    m = stats.moments(d)
    ax.axvline(m["mean"], color=INK, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.annotate(
        f"mean {m['mean']:.3f}", (m["mean"], h.max()), xytext=(-5, -2), textcoords="offset points",
        ha="right", va="top", color=INK, fontsize=8,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("density")
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_xp_density.pdf")
    plt.close(fig)


def fig_scales():
    exps = [3, 6, 12, 21, 30]
    grid = np.linspace(0, 1, 1201)
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    for j, col in zip(exps, RAMP):
        primes = first_primes_above(10**j, PRIMES_PER_WINDOW)
        d = stats.pooled(primes, 2, lambda p: 0)
        F = stats.cdf(d, grid)
        ax.plot(grid, F, color=col, linewidth=1.3, label=rf"$p\approx10^{{{j}}}$")
        mean = stats.moments(d)["mean"]
        ax.plot([mean], [float(np.interp(mean, grid, F))], marker="o", markersize=4.5,
                markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("cumulative probability")
    ax.legend(loc="upper left", fontsize=8, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_xp_scales.pdf")
    plt.close(fig)


def fig_regimes():
    primes = stats.window(*REGIME_WINDOW)
    fp = stats.pooled(primes, 3, lambda p: 0)
    fpf = stats.pooled(primes, 3, lambda p: 1)
    grid = np.linspace(0, 1, 1201)
    gaps, bounds = [], []
    for p in primes:
        a = stats.moments(stats.x_distribution(p, 3, 0))["mean"]
        b = stats.moments(stats.x_distribution(p, 3, 1))["mean"]
        gaps.append(b - a)
        bounds.append(stats.coupling_bound(p, 3))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    ax1.plot(grid, stats.cdf(fp, grid), color=BLUE, linewidth=1.3, label="fixed-point regime")
    ax1.plot(grid, stats.cdf(fpf, grid), color=ORANGE, linewidth=1.3, linestyle=(0, (5, 2)),
             label="fixed-point-free regime")
    ax1.set_xlim(0.2, 1)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel(XLABEL)
    ax1.set_ylabel("cumulative probability")
    ax1.legend(loc="upper left", fontsize=7.5, handlelength=2.2)
    ax1.set_title("(a) same primes, both regimes", fontsize=9, color=INK, loc="left")

    top = max(bounds) * 1.06
    ax2.plot([0, top], [0, top], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax2.scatter(bounds, gaps, s=11, color=BLUE, edgecolors="white", linewidths=0.4, zorder=3)
    ax2.annotate("bound", (top * 0.78, top * 0.78), xytext=(-4, 4), textcoords="offset points",
                 ha="right", va="bottom", color=MUTED, fontsize=8)
    ax2.set_xlim(0, top)
    ax2.set_ylim(0, top)
    ax2.set_xlabel(r"bound $\eta_3(p)=v_2(p-1)\log 2/\log p$")
    ax2.set_ylabel("gap in mean of $X_p$")
    ax2.set_title("(b) per-prime gap against the bound", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_xp_regimes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_density()
    fig_scales()
    fig_regimes()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig_xp_*.pdf"))))
