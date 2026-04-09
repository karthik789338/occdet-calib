from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_main_f1(results: pd.DataFrame, outdir: Path):
    df = results[results["experiment"] == "overlap_main"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("occlusion_level")
        ax.plot(sub["occlusion_level"], sub["f1"], marker="o", label=model)
    ax.set_xlabel("Occlusion level")
    ax.set_ylabel("F1")
    ax.set_title("Main overlap benchmark: F1 vs occlusion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "fig_overlap_main_f1.png")


def plot_overlap_main_dece(dece: pd.DataFrame, outdir: Path):
    df = dece[dece["experiment"] == "overlap_main"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("occlusion_level")
        ax.plot(sub["occlusion_level"], sub["dece"], marker="o", label=model)
    ax.set_xlabel("Occlusion level")
    ax.set_ylabel("DECE")
    ax.set_title("Main overlap benchmark: DECE vs occlusion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "fig_overlap_main_dece.png")


def plot_yolo_controls(results: pd.DataFrame, dece: pd.DataFrame, outdir: Path):
    r = results[results["model"] == "YOLOv8m"].copy()
    d = dece[dece["model"] == "YOLOv8m"].copy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for exp in ["overlap_main", "distractor_control", "truncation_control"]:
        sub = r[r["experiment"] == exp].sort_values("occlusion_level")
        if len(sub):
            ax.plot(sub["occlusion_level"], sub["f1"], marker="o", label=exp)
    ax.set_xlabel("Occlusion level")
    ax.set_ylabel("F1")
    ax.set_title("YOLO controls: F1 vs occlusion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "fig_yolo_controls_f1.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for exp in ["overlap_main", "distractor_control", "truncation_control"]:
        sub = d[d["experiment"] == exp].sort_values("occlusion_level")
        if len(sub):
            ax.plot(sub["occlusion_level"], sub["dece"], marker="o", label=exp)
    ax.set_xlabel("Occlusion level")
    ax.set_ylabel("DECE")
    ax.set_title("YOLO controls: DECE vs occlusion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "fig_yolo_controls_dece.png")


def plot_dangerous_zone(dz: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in dz["model"].unique():
        sub = dz[dz["model"] == model].sort_values("occlusion_level")
        ax.plot(sub["occlusion_level"], sub["dangerous_zone"].astype(int), marker="o", label=model)
    ax.set_xlabel("Occlusion level")
    ax.set_ylabel("Dangerous zone flag")
    ax.set_title("Dangerous Zone by architecture")
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, outdir / "fig_dangerous_zone_flags.png")


def plot_overlap_vs_inpainting(outdir: Path):
    perf_path = Path("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_inpainting_scale500/yolo_overlap_vs_inpainting_perf.csv")
    dece_path = Path("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_inpainting_scale500/yolo_overlap_vs_inpainting_dece.csv")
    if not perf_path.exists() or not dece_path.exists():
        return

    perf = pd.read_csv(perf_path)
    dece = pd.read_csv(dece_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in perf["condition"].unique():
        sub = perf[perf["condition"] == cond].sort_values("occlusion_level")
        ax.plot(sub["occlusion_level"], sub["f1"], marker="o", label=cond)
    ax.set_xlabel("Occlusion level")
    ax.set_ylabel("F1")
    ax.set_title("YOLO: overlap vs inpainting")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "fig_overlap_vs_inpainting_f1.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in dece["condition"].unique():
        sub = dece[dece["condition"] == cond].sort_values("occlusion_level")
        ax.plot(sub["occlusion_level"], sub["dece"], marker="o", label=cond)
    ax.set_xlabel("Occlusion level")
    ax.set_ylabel("DECE")
    ax.set_title("YOLO: overlap vs inpainting DECE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / "fig_overlap_vs_inpainting_dece.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--dece_csv", required=True)
    parser.add_argument("--dangerous_zone_csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    dece = pd.read_csv(args.dece_csv)
    dz = pd.read_csv(args.dangerous_zone_csv)
    outdir = Path(args.outdir)

    plot_overlap_main_f1(results, outdir)
    plot_overlap_main_dece(dece, outdir)
    plot_yolo_controls(results, dece, outdir)
    plot_dangerous_zone(dz, outdir)
    plot_overlap_vs_inpainting(outdir)

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
