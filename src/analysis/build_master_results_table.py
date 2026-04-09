from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_with_condition(path: str, model: str, experiment: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["model"] = model
    df["experiment"] = experiment
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build master results table for paper.")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    dfs = []

    # Main overlap
    dfs.append(load_with_condition(
        "/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_overlap_scale2000/yolo_overlap_by_occlusion_thr040.csv",
        "YOLOv8m", "overlap_main"
    ))
    dfs.append(load_with_condition(
        "/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_overlap_scale1000/fcos_overlap_by_occlusion_thr020.csv",
        "FCOS-R50", "overlap_main"
    ))
    dfs.append(load_with_condition(
        "/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_overlap_scale1000/deformable_detr_overlap_by_occlusion_thr030.csv",
        "Deformable-DETR-R50", "overlap_main"
    ))

    # YOLO controls
    dfs.append(load_with_condition(
        "/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_distractor_scale500/yolo_distractor_by_level_thr040.csv",
        "YOLOv8m", "distractor_control"
    ))
    dfs.append(load_with_condition(
        "/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_truncation_scale500/yolo_truncation_by_level_thr040.csv",
        "YOLOv8m", "truncation_control"
    ))

    out = pd.concat(dfs, ignore_index=True)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
