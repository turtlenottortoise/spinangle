#!/usr/bin/env python3
"""Generate the required plot artifacts from results/all_runs.csv.

Produces (when the relevant columns are populated):
  plots/success_vs_steps.png        success rate vs training steps, per variant
  plots/rollout_error_vs_horizon.png  k-step latent error vs horizon, per variant
  plots/retrieval_vs_steps.png      future-state retrieval@1 vs training steps
  plots/rank_clumping.png           effective rank vs pairwise-cosine clumping
  plots/planning_budget_curve.png   success vs number of sampled plans (CEM budget)

Uses only matplotlib + stdlib csv (no pandas). Rows missing a metric are skipped
for that plot; a plot with no usable rows is annotated as "awaiting results" so the
artifact always exists. Run: python scripts/make_plots.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "all_runs.csv"
PLOTS = ROOT / "plots"


def load_rows():
    if not CSV.exists():
        return []
    with CSV.open() as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    v = row.get(key, "")
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def _empty(ax, msg="awaiting results"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="gray")


def by_variant(rows):
    d = defaultdict(list)
    for r in rows:
        d[r.get("variant", "?")].append(r)
    return d


def plot_xy(rows, xkey, ykey, title, xlabel, ylabel, fname, sort=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    any_pts = False
    for variant, rs in sorted(by_variant(rows).items()):
        pts = [(fnum(r, xkey), fnum(r, ykey)) for r in rs]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if not pts:
            continue
        if sort:
            pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=variant)
        any_pts = True
    if any_pts:
        ax.legend(fontsize=8)
    else:
        _empty(ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=120)
    plt.close(fig)
    print(f"  wrote plots/{fname}{'' if any_pts else '  (placeholder)'}")


def plot_rollout(rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    horizons = [1, 5, 10, 20]
    keys = [f"roll_err_{h}" for h in horizons]
    any_pts = False
    for variant, rs in sorted(by_variant(rows).items()):
        r = rs[-1]  # most recent row for this variant
        ys = [fnum(r, k) for k in keys]
        pairs = [(h, y) for h, y in zip(horizons, ys) if y is not None]
        if not pairs:
            continue
        xs, yy = zip(*pairs)
        ax.plot(xs, yy, marker="o", label=variant)
        any_pts = True
    if any_pts:
        ax.legend(fontsize=8)
    else:
        _empty(ax)
    ax.set(title="Latent rollout error vs horizon", xlabel="rollout horizon (steps)",
           ylabel="prediction error (cosine dist or MSE)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "rollout_error_vs_horizon.png", dpi=120)
    plt.close(fig)
    print(f"  wrote plots/rollout_error_vs_horizon.png{'' if any_pts else '  (placeholder)'}")


def plot_rank_clumping(rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    any_pts = False
    for variant, rs in sorted(by_variant(rows).items()):
        r = rs[-1]
        x, y = fnum(r, "eff_rank"), fnum(r, "clumping")
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=60)
        ax.annotate(variant, (x, y), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
        any_pts = True
    if not any_pts:
        _empty(ax)
    ax.set(title="Representation geometry", xlabel="effective rank (higher = richer)",
           ylabel="mean pairwise cosine (lower = less clumping)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "rank_clumping.png", dpi=120)
    plt.close(fig)
    print(f"  wrote plots/rank_clumping.png{'' if any_pts else '  (placeholder)'}")


def main():
    PLOTS.mkdir(exist_ok=True)
    rows = load_rows()
    print(f"loaded {len(rows)} run row(s) from {CSV}")
    plot_xy(rows, "train_steps", "success",
            "Task success vs training steps", "training steps", "success rate",
            "success_vs_steps.png")
    plot_rollout(rows)
    plot_xy(rows, "train_steps", "fut_r1",
            "Future-state retrieval@1 vs training steps", "training steps",
            "future-state retrieval@1", "retrieval_vs_steps.png")
    plot_rank_clumping(rows)
    plot_xy(rows, "plan_samples", "success",
            "Planning-budget curve", "number of sampled plans (CEM)", "success rate",
            "planning_budget_curve.png", sort=True)


if __name__ == "__main__":
    main()
