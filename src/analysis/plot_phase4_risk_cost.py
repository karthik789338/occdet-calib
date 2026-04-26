from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot risk-cost curves.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    sub = df[df["scope"] == "by_occlusion"].copy()

    # keep only numeric occlusion levels
    sub["occlusion_level_num"] = pd.to_numeric(sub["occlusion_level"], errors="coerce")
    sub = sub.dropna(subset=["occlusion_level_num"]).copy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for occ in sorted(sub["occlusion_level_num"].unique()):
        occ_sub = sub[sub["occlusion_level_num"] == occ].sort_values("threshold")
        ax.plot(occ_sub["threshold"], occ_sub["cost_per_gt"], marker="o", label=f"occ={occ:.1f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cost per GT")
    ax.set_title(f"{args.model_name}: simple risk-cost interpretation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_plot(fig, Path(args.out_path))
    print(f"Saved risk-cost figure to: {args.out_path}")


if __name__ == "__main__":
    main()
