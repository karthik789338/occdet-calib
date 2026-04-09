from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bucket_from_occlusion(x: float) -> str:
    if x <= 0.2:
        return "high_visibility"
    if x <= 0.4:
        return "medium_visibility"
    if x <= 0.6:
        return "low_visibility"
    return "very_low_visibility"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched predictions by visibility bucket.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path).copy()

    if "estimated_occlusion" not in gt.columns:
        raise KeyError("GT must contain estimated_occlusion")

    gt["visibility_bucket"] = gt["estimated_occlusion"].map(bucket_from_occlusion)

    rows = []
    for bucket, gt_subset in gt.groupby("visibility_bucket"):
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
                "visibility_bucket": bucket,
                "num_predictions": len(pred_subset),
                "num_ground_truth": total_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_correct": float(pred_subset["correct"].mean()) if len(pred_subset) else None,
                "avg_score": float(pred_subset["score"].mean()) if len(pred_subset) else None,
            }
        )

    out = pd.DataFrame(rows).sort_values("visibility_bucket")
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
