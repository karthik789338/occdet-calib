from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(summary: pd.DataFrame, out_dir: Path, model_name: str):
    for dp in sorted(summary["delta_percentile"].unique()):
        sub = summary[summary["delta_percentile"] == dp].copy()
        pivot = sub.pivot(index="rho", columns="pi", values="num_flagged")

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pivot.values)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center")

        ax.set_xlabel("pi floor")
        ax.set_ylabel("rho floor")
        ax.set_title(f"{model_name}: DZ count heatmap (delta pct={int(dp)})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        save_plot(fig, out_dir / f"{model_name.lower().replace(' ', '_')}_dz_heatmap_dp{int(dp)}.png")


def plot_flag_lines(detail: pd.DataFrame, out_dir: Path, model_name: str):
    # show 3 representative configurations
    wanted = [
        (0.50, 0.10, 50),
        (0.60, 0.15, 70),
        (0.70, 0.20, 80),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for rho, pi, dp in wanted:
        sub = detail[
            (detail["rho"] == rho)
            & (detail["pi"] == pi)
            & (detail["delta_percentile"] == dp)
        ].sort_values("occlusion_level")

        if len(sub) == 0:
            continue

        ax.plot(
            sub["occlusion_level"],
            sub["dangerous_zone"].astype(int),
            marker="o",
            label=f"rho={rho:.2f}, pi={pi:.2f}, dp={int(dp)}",
        )

    ax.set_xlabel("Occlusion level")
    ax.set_ylabel("Dangerous zone flag")
    ax.set_yticks([0, 1])
    ax.set_title(f"{model_name}: DZ sensitivity examples")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_plot(fig, out_dir / f"{model_name.lower().replace(' ', '_')}_dz_flag_lines.png")


def main():
    parser = argparse.ArgumentParser(description="Plot Dangerous Zone robustness figures.")
    parser.add_argument("--detail_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    detail = pd.read_csv(args.detail_csv)
    summary = pd.read_csv(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmaps(summary, out_dir, args.model_name)
    plot_flag_lines(detail, out_dir, args.model_name)

    print(f"Saved Phase 4 DZ robustness figures to: {out_dir}")


if __name__ == "__main__":
    main()
