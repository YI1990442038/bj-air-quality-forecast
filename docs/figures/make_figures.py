#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the README figures for bj-air-quality-forecast.

All numbers are copied verbatim from RESULTS.md / M5_深度学习结果.md, so the
figures can never drift from the written record without this file changing too.

Usage (matplotlib is NOT a project dependency — install it anywhere):
    python -m pip install --target .tmp_pylibs matplotlib
    PYTHONPATH=.tmp_pylibs python docs/figures/make_figures.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- palette
BLUE = "#2f6fed"
BLUE_LIGHT = "#9dbcfa"
SLATE = "#94a3b8"
RED = "#e05252"
AMBER = "#e8a33d"
INK = "#0f172a"
MUTED = "#475569"
GRID = "#e2e8f0"

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "axes.labelcolor": MUTED,
        "axes.titlesize": 11,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _clean(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------- 1. headline result
def fig_main():
    leads = ["h = 1", "h = 3", "h = 6"]
    model = [2.802, 5.965, 8.588]
    base = [3.228, 7.425, 11.592]
    gain = ["+13.21%", "+19.66%", "+25.91%"]

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = range(len(leads))
    w = 0.34

    b1 = ax.bar([i - w / 2 for i in x], base, w, label="Persistence baseline B1",
                color=SLATE, zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], model, w, label="LightGBM v4",
                color=BLUE, zorder=3)

    for rect, v in zip(b1, base):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.18, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color=MUTED)
    for rect, v in zip(b2, model):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.18, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=BLUE, fontweight="bold")
    for i, g in zip(x, gain):
        ax.annotate(g, xy=(i, max(base[i], model[i]) + 1.15), ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=INK,
                    bbox=dict(boxstyle="round,pad=0.32", fc="#eef4ff", ec="#c7d9fb", lw=0.8))

    ax.set_xticks(list(x))
    ax.set_xticklabels(leads, fontsize=10.5)
    ax.set_ylabel("MAE  (µg/m³, lower is better)")
    ax.set_ylim(0, 14.2)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    _clean(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    save(fig, "fig1_main_results.png")


# ------------------------------------------------- 2. model generation
def fig_progression():
    versions = ["v1\n34 feat\nno meteo", "v2\n+ sample\nweights", "v3\n+ graph\nfeatures",
                "v4\n+ ERA5\nmeteorology"]
    series = {
        "h = 1": ([13.89, 13.24, 13.79, 13.21], SLATE),
        "h = 3": ([16.33, 14.48, 15.15, 19.66], AMBER),
        "h = 6": ([18.27, 16.57, 16.61, 25.91], BLUE),
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for name, (vals, color) in series.items():
        ax.plot(range(4), vals, marker="o", markersize=6.5, linewidth=2.2,
                color=color, label=name, zorder=3)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 11 if name != "h = 1" else -15),
                        textcoords="offset points", ha="center",
                        fontsize=8.8, color=color, fontweight="bold")

    ax.axvspan(2.55, 3.45, color="#eef4ff", zorder=0)

    ax.set_xticks(range(4))
    ax.set_xticklabels(versions, fontsize=9)
    ax.set_ylabel("MAE improvement vs B1  (%)")
    ax.set_ylim(11.4, 28)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim(-0.4, 3.5)
    _clean(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, ncol=3,
              bbox_to_anchor=(0.0, 1.02))
    save(fig, "fig2_progression.png")


# ------------------------------------------------- 3. failure mode
def fig_concentration():
    bins = ["< 35\ngood", "35–75\nmoderate", "75–115\nlight", "115–150\nmedium",
            "150–250\nheavy", "> 250\nsevere"]
    v1 = [22.1, 17.0, 7.5, 5.9, -23.1, -9.7]
    v2 = [18.0, 13.2, 15.9, 19.8, 18.5, 4.9]

    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    w = 0.36
    for i in range(len(bins)):
        c1 = RED if v1[i] < 0 else BLUE_LIGHT
        c2 = RED if v2[i] < 0 else BLUE
        ax.bar(i - w / 2, v1[i], w, color=c1, zorder=3)
        ax.bar(i + w / 2, v2[i], w, color=c2, zorder=3)
        for dx, v in ((-w / 2, v1[i]), (w / 2, v2[i])):
            off = -1.4 if v < 0 else 1.4
            ax.text(i + dx, v + off, f"{v:+.1f}", ha="center",
                    va="top" if v < 0 else "bottom", fontsize=7.8,
                    color=RED if v < 0 else INK, fontweight="bold")

    ax.axhline(0, color="#94a3b8", linewidth=1.0, zorder=4)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, fontsize=8.8)
    ax.set_ylabel("gain vs B1 at h = 6  (%)")
    ax.set_ylim(-30, 30)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    _clean(ax)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE_LIGHT, label="v1 unweighted (34 feat)"),
                       Patch(color=BLUE, label="v2 weighted (≥75 ×2, ≥150 ×5)")],
              frameon=False, loc="lower left", fontsize=9)
    ax.annotate("model beat the baseline everywhere\nonce heavy episodes were weighted",
                xy=(4.35, 20), xytext=(4.35, 26.5), ha="center", fontsize=8.6,
                color=MUTED)
    save(fig, "fig3_concentration.png")


# ------------------------------------------------- 4. DL comparison
def fig_models():
    # measured in the same GPU run on the A40 host (M5_深度学习结果.md §4)
    models = [
        ("TCN v1  (5 ch)", 3.345, 7.223, 10.945),
        ("LSTM v1  (5 ch)", 2.993, 6.876, 10.679),
        ("TCN v2  (14 ch)", 3.108, 6.550, 9.697),
        ("LSTM v2  (14 ch)", 2.846, 6.265, 9.425),
        ("LightGBM v4", 2.839, 6.010, 8.627),
    ]
    colors = [SLATE, AMBER, BLUE]

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    n = len(models)
    hgt = 0.26
    for i, (name, *vals) in enumerate(models):
        for k, (v, c) in enumerate(zip(vals, colors)):
            y = i + (k - 1) * hgt
            ax.barh(y, v, hgt * 0.9, color=c, zorder=3)
            ax.text(v + 0.12, y, f"{v:.3f}", va="center", ha="left",
                    fontsize=7.6, color=INK)

    ax.set_yticks(range(n))
    ax.set_yticklabels([m[0] for m in models], fontsize=9)
    ax.set_xlabel("test MAE  (µg/m³, lower is better)")
    ax.set_xlim(0, 12.6)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.invert_yaxis()
    _clean(ax, grid_axis="x")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[0], label="h = 1"),
                       Patch(color=colors[1], label="h = 3"),
                       Patch(color=colors[2], label="h = 6")],
              frameon=False, fontsize=9, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig4_models.png")


if __name__ == "__main__":
    fig_main()
    fig_progression()
    fig_concentration()
    fig_models()
