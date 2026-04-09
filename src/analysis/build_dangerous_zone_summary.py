from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_tag(path: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["model"] = model
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Dangerous Zone summary table.")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    dfs = [
        load_tag("/home/karthikadari/occdet-calib/outputs/analysis/dangerous_zone/yolo_overlap_scale2000.csv", "YOLOv8m"),
        load_tag("/home/karthikadari/occdet-calib/outputs/analysis/dangerous_zone/fcos_overlap_scale1000.csv", "FCOS-R50"),
        load_tag("/home/karthikadari/occdet-calib/outputs/analysis/dangerous_zone/ddetr_overlap_scale1000.csv", "Deformable-DETR-R50"),
    ]

    out = pd.concat(dfs, ignore_index=True)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
