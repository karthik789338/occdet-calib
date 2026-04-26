from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, title: str):
    models = df["model"].unique().tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        subm = df[df["model"] == model]
        for control in subm["control"].unique():
            sub = subm[subm["control"] == control].sort_values("occlusion_level")
            ax.plot(sub["occlusion_level"], sub[metric], marker="o", label=control)
        ax.set_title(model)
        ax.set_xlabel("Occlusion level")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel(metric.upper())

    axes[-1].legend()
    fig.suptitle(title)
    save_plot(fig, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf_csv", required=True)
    parser.add_argument("--dece_csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    perf = pd.read_csv(args.perf_csv)
    dece = pd.read_csv(args.dece_csv)
    outdir = Path(args.outdir)

    plot_metric(perf, "f1", outdir / "phase2_controls_f1.png", "Cross-architecture controls: F1")
    plot_metric(perf, "recall", outdir / "phase2_controls_recall.png", "Cross-architecture controls: Recall")
    plot_metric(dece, "dece", outdir / "phase2_controls_dece.png", "Cross-architecture controls: DECE")

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
