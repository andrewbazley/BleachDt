#!/usr/bin/env python3
"""
Generate a pre_mean vs droplet_radius plot, grouped by condition.

This script searches a base directory recursively for `FRAP_condition_summary.csv`
files (produced by `Crop_PDE_Cursor_V7.py`). It then:
  - filters rows to a selected model (default: "Separate")
  - groups rows by a derived "condition key" from the `filename` column
  - optionally filters brightness outliers within each condition (± N SD)
  - plots one panel per condition and saves a single PNG

The key goal is to be *condition-name agnostic* (no hard-coded mutants like VF/T229A).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:
    scipy_stats = None

from condition_utils import extract_condition_key, extract_replicate_id


Point = Tuple[float, float, Optional[str], Optional[str], str]  # (pre_mean, radius, rep, ligand, filename)


def _infer_ligand_from_filename(filename: str) -> Optional[str]:
    tokens = [t for t in filename.split("_") if t]
    for tok in tokens:
        up = tok.upper()
        if up in {"ADP", "ATP", "AMP", "GTP", "GDP"}:
            return up
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot pre_mean vs droplet_radius, grouped by condition.",
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Directory to search recursively for FRAP_condition_summary.csv (default: current directory).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: <base_dir>/pre_mean_vs_radius_conditions.png).",
    )
    parser.add_argument(
        "--model",
        default="Separate",
        help='Value of the "Models" column to keep (default: Separate). Use "" to disable filtering.',
    )
    parser.add_argument(
        "--brightness-sd",
        type=float,
        default=2.0,
        help="Filter pre_mean outliers per condition outside ±N SD (default: 2.0). Set 0 to disable.",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=3,
        help="Maximum subplot columns (default: 3).",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_path = args.out or os.path.join(base_dir, "pre_mean_vs_radius_conditions.png")

    # condition -> list[Point]
    cond_points: DefaultDict[str, List[Point]] = defaultdict(list)

    summary_paths: List[str] = []
    for root, _dirs, files in os.walk(base_dir):
        if "FRAP_condition_summary.csv" in files:
            summary_paths.append(os.path.join(root, "FRAP_condition_summary.csv"))

    if not summary_paths:
        print(f"No FRAP_condition_summary.csv files found under: {base_dir}")
        return 1

    for csv_path in sorted(summary_paths):
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if args.model and row.get("Models", "").strip() != args.model:
                        continue

                    filename = (row.get("filename") or "").strip()
                    if not filename:
                        continue

                    try:
                        pre_mean = float(row.get("pre_mean", "nan"))
                        radius = float(row.get("droplet_radius_um", "nan"))
                    except ValueError:
                        continue

                    if not (np.isfinite(pre_mean) and np.isfinite(radius)):
                        continue

                    condition = extract_condition_key(filename) or "Unknown"
                    rep = extract_replicate_id(filename)
                    ligand = _infer_ligand_from_filename(filename)
                    cond_points[condition].append((pre_mean, radius, rep, ligand, filename))
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    if not cond_points:
        print("No usable data found (after filtering).")
        return 1

    conditions = sorted(cond_points.keys())
    n_cond = len(conditions)

    # Palette for replicate colors (tab10-like)
    rep_palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    ncols = max(1, min(int(args.max_cols), n_cond))
    nrows = int(math.ceil(n_cond / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n_cond:]:
        ax.set_visible(False)

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]
        pts = cond_points[cond]
        if not pts:
            ax.text(0.5, 0.5, f"No data for {cond}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cond)
            ax.set_xlabel("pre_mean")
            ax.set_ylabel("droplet_radius_um")
            continue

        pre_means = np.array([p[0] for p in pts], dtype=float)
        radii = np.array([p[1] for p in pts], dtype=float)
        reps = [p[2] for p in pts]
        ligands = [p[3] for p in pts]

        # Brightness outlier filter (per condition)
        mask_keep = np.ones(pre_means.shape[0], dtype=bool)
        if args.brightness_sd and args.brightness_sd > 0:
            mu = float(np.nanmean(pre_means))
            sigma = float(np.nanstd(pre_means))
            if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
                lo = mu - args.brightness_sd * sigma
                hi = mu + args.brightness_sd * sigma
                mask_keep &= (pre_means >= lo) & (pre_means <= hi)

        pre_f = pre_means[mask_keep]
        rad_f = radii[mask_keep]
        reps_f = [r for i, r in enumerate(reps) if mask_keep[i]]
        lig_f = [l for i, l in enumerate(ligands) if mask_keep[i]]

        # Replicate colors (within condition)
        rep_labels = sorted({r for r in reps_f if r is not None}) or ["NA"]
        rep_to_color = {rep: rep_palette[i % len(rep_palette)] for i, rep in enumerate(rep_labels)}

        # Scatter
        for x, y, rep, lig in zip(pre_f, rad_f, reps_f, lig_f):
            rep_key = rep if rep is not None else "NA"
            face = rep_to_color.get(rep_key, "#000000")

            # Optional ligand styling (only if present)
            marker = "o"
            edge = "black"
            if lig == "ADP":
                marker = "s"
                edge = "blue"
            elif lig == "ATP":
                marker = "o"
                edge = "red"

            ax.scatter(
                x,
                y,
                s=40,
                alpha=0.75,
                marker=marker,
                color=face,
                edgecolors=edge,
                linewidths=1.2,
            )

        # Regression on filtered points (optional)
        mask_xy = np.isfinite(pre_f) & np.isfinite(rad_f)
        x_clean = pre_f[mask_xy]
        y_clean = rad_f[mask_xy]
        if x_clean.size >= 2 and scipy_stats is not None:
            lr = scipy_stats.linregress(x_clean, y_clean)
            r2 = lr.rvalue**2
            x_line = np.linspace(float(x_clean.min()), float(x_clean.max()), 100)
            y_line = lr.slope * x_line + lr.intercept
            ax.plot(x_line, y_line, "k--", linewidth=1.5)
            ax.text(
                0.02,
                0.98,
                f"R²={r2:.3f}, p={lr.pvalue:.2e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Legend (replicates)
        handles = []
        labels = []
        for rep in rep_labels:
            h = ax.scatter([], [], s=40, color=rep_to_color[rep], edgecolors="black", linewidths=1.0)
            handles.append(h)
            labels.append(f"rep {rep}")
        if handles:
            ax.legend(handles, labels, fontsize=8, loc="best", title="Replicate")

        ax.set_title(f"{cond} (n={x_clean.size})")
        ax.set_xlabel("Average droplet pre-bleach intensity (pre_mean)")
        ax.set_ylabel("Droplet radius (µm)")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    print(f"Saving figure to {out_path}")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

