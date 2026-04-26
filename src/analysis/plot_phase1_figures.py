from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.calibration_metrics_extended import adaptive_bin_stats, uniform_bin_stats


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def reliability_panel(df: pd.DataFrame, score_col: str, out_path: Path, title: str):
    levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharex=True, sharey=True)

    for ax, lv in zip(axes, levels):
        sub = df[df["nominal_occlusion"] == lv].copy() if "nominal_occlusion" in df.columns else df.copy()
        stats = uniform_bin_stats(sub, confidence_col=score_col, correctness_col="correct", n_bins=15)
        xs = stats["avg_confidence"].dropna().to_numpy()
        ys = stats["avg_accuracy"].dropna().to_numpy()

        ax.plot([0, 1], [0, 1])
        if len(xs):
            ax.plot(xs, ys, marker="o")
        ax.set_title(f"occ={lv:.1f}")
        ax.set_xlabel("Confidence")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    save_plot(fig, out_path)


def risk_coverage_curve(df: pd.DataFrame, score_col: str, out_path: Path, title: str):
    scores = df[score_col].astype(float).to_numpy()
    labels = df["correct"].astype(float).to_numpy()

    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    coverages = []
    risks = []
    precisions = []

    n = len(scores)
    for k in np.linspace(10, n, 100, dtype=int):
        kept = labels[:k]
        precision = kept.mean() if len(kept) else 0.0
        risk = 1.0 - precision
        coverage = k / n

        coverages.append(float(coverage))
        risks.append(float(risk))
        precisions.append(float(precision))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(coverages, risks)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (1 - precision)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_plot(fig, out_path)

    curve_df = pd.DataFrame(
        {"coverage": coverages, "selective_risk": risks, "precision": precisions, "score_column": score_col}
    )
    curve_df.to_csv(out_path.with_suffix(".csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Build reliability diagrams and risk-coverage curves.")
    parser.add_argument("--matched_pred_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gt_path", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.matched_pred_path)
    if args.gt_path:
        gt = pd.read_csv(args.gt_path)
        keep_cols = [c for c in ["image_key", "nominal_occlusion"] if c in gt.columns]
        if keep_cols:
            gt_small = gt[keep_cols].drop_duplicates("image_key")
            df = df.merge(gt_small, on="image_key", how="left")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_columns = ["score"]
    for col in ["score_global_ts", "score_oc_ts"]:
        if col in df.columns:
            score_columns.append(col)

    for col in score_columns:
        reliability_panel(
            df,
            score_col=col,
            out_path=out_dir / f"{col}_reliability_panel.png",
            title=f"Reliability diagrams ({col})",
        )
        risk_coverage_curve(
            df,
            score_col=col,
            out_path=out_dir / f"{col}_risk_coverage.png",
            title=f"Risk-coverage ({col})",
        )

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
