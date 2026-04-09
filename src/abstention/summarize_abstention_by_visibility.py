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
    parser = argparse.ArgumentParser(description="Summarize abstention at one threshold by visibility bucket.")
    parser.add_argument("--matched_with_oc_ts_csv", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["raw", "global_ts", "oc_ts"], required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.matched_with_oc_ts_csv)
    gt = pd.read_csv(args.gt_path).copy()

    score_col = {
        "raw": "score",
        "global_ts": "score_global_ts",
        "oc_ts": "score_oc_ts",
    }[args.mode]

    gt["visibility_bucket"] = gt["estimated_occlusion"].map(bucket_from_occlusion)
    gt_small = gt[["image_key", "visibility_bucket"]].drop_duplicates("image_key")

    df = df.merge(gt_small, on="image_key", how="left")
    df = df[df[score_col] >= args.threshold].copy()

    rows = []

    for bucket, gt_subset in gt.groupby("visibility_bucket"):
        image_keys = set(gt_subset["image_key"].astype(str).tolist())
        sub = df[df["image_key"].astype(str).isin(image_keys)].copy()

        tp = int((sub["correct"] == 1).sum())
        fp = int((sub["correct"] == 0).sum())
        total_gt = int(len(gt_subset))
        fn = max(0, total_gt - tp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, total_gt)
        f1 = safe_div(2 * precision * recall, precision + recall)
        selective_risk = 1.0 - precision if len(sub) > 0 else None

        rows.append(
            {
                "mode": args.mode,
                "threshold": args.threshold,
                "visibility_bucket": bucket,
                "num_predictions_kept": len(sub),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "selective_risk": selective_risk,
                "avg_score_kept": float(sub[score_col].mean()) if len(sub) else None,
            }
        )

    out = pd.DataFrame(rows).sort_values("visibility_bucket")
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
