from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def tag_df(path: str, model: str, control: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["model"] = model
    df["control"] = control
    return df


def main():
    parser = argparse.ArgumentParser(description="Build cross-architecture control tables.")
    parser.add_argument("--perf_out", required=True)
    parser.add_argument("--dece_out", required=True)
    args = parser.parse_args()

    perf = []
    dece = []

    # YOLO
    perf.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_distractor_scale500/yolo_distractor_by_level_thr040.csv", "YOLOv8m", "distractor"))
    perf.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_truncation_scale500/yolo_truncation_by_level_thr040.csv", "YOLOv8m", "truncation"))

    dece.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_distractor_scale500/yolo_distractor_dece_by_level_thr040.csv", "YOLOv8m", "distractor"))
    dece.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_truncation_scale500/yolo_truncation_dece_by_level_thr040.csv", "YOLOv8m", "truncation"))

    # FCOS
    perf.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_distractor_scale500/fcos_distractor_by_level_thr020.csv", "FCOS-R50", "distractor"))
    perf.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_truncation_scale500/fcos_truncation_by_level_thr020.csv", "FCOS-R50", "truncation"))

    dece.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_distractor_scale500/fcos_distractor_dece_by_level_thr020.csv", "FCOS-R50", "distractor"))
    dece.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_truncation_scale500/fcos_truncation_dece_by_level_thr020.csv", "FCOS-R50", "truncation"))

    # DDETR
    perf.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_distractor_scale500/deformable_detr_distractor_by_level_thr030.csv", "Deformable-DETR-R50", "distractor"))
    perf.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_truncation_scale500/deformable_detr_truncation_by_level_thr030.csv", "Deformable-DETR-R50", "truncation"))

    dece.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_distractor_scale500/deformable_detr_distractor_dece_by_level_thr030.csv", "Deformable-DETR-R50", "distractor"))
    dece.append(tag_df("/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_truncation_scale500/deformable_detr_truncation_dece_by_level_thr030.csv", "Deformable-DETR-R50", "truncation"))

    perf_df = pd.concat(perf, ignore_index=True)
    dece_df = pd.concat(dece, ignore_index=True)

    Path(args.perf_out).parent.mkdir(parents=True, exist_ok=True)
    perf_df.to_csv(args.perf_out, index=False)
    dece_df.to_csv(args.dece_out, index=False)

    print("Saved:", args.perf_out)
    print("Saved:", args.dece_out)
    print(perf_df.head())
    print(dece_df.head())


if __name__ == "__main__":
    main()
