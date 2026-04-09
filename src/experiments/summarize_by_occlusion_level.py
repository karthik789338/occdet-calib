from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched predictions by occlusion level.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--occlusion_col", type=str, default="nominal_occlusion")
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path)

    rows = []

    for occ_value, gt_subset in gt.groupby(args.occlusion_col):
        image_keys = set(gt_subset["image_key"].astype(str).tolist())
        pred_subset = matched[matched["image_key"].astype(str).isin(image_keys)].copy()

        tp = int((pred_subset["correct"] == 1).sum())
        fp = int((pred_subset["correct"] == 0).sum())
        total_gt = int(len(gt_subset))
        fn = max(0, total_gt - tp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, total_gt)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "occlusion_level": occ_value,
                "num_predictions": len(pred_subset),
                "num_ground_truth": total_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("occlusion_level")
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(out_df)


if __name__ == "__main__":
    main()
