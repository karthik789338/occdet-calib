from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics.calibration_metrics_extended import summarize_calibration_metrics


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)


def main():
    parser = argparse.ArgumentParser(description="Summarize matched predictions by BDD natural group.")
    parser.add_argument("--matched_pred_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path)

    rows = []

    for group, gt_sub in gt.groupby("natural_group"):
        keys = set(gt_sub["image_key"].astype(str).tolist())
        pred_sub = matched[matched["image_key"].astype(str).isin(keys)].copy()

        tp = int((pred_sub["correct"] == 1).sum())
        fp = int((pred_sub["correct"] == 0).sum())
        total_gt = int(len(gt_sub))
        fn = max(0, total_gt - tp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, total_gt)
        f1 = safe_div(2 * precision * recall, precision + recall)

        cal = summarize_calibration_metrics(pred_sub, confidence_col="score", correctness_col="correct", n_bins=15)

        rows.append(
            {
                "natural_group": group,
                "num_predictions": len(pred_sub),
                "num_ground_truth": total_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "d_ece": cal["d_ece"],
                "ace": cal["ace"],
                "mce": cal["mce"],
                "brier": cal["brier"],
                "nll": cal["nll"],
                "avg_score": float(pred_sub["score"].mean()) if len(pred_sub) else None,
                "avg_correct": float(pred_sub["correct"].mean()) if len(pred_sub) else None,
            }
        )

    out = pd.DataFrame(rows).sort_values("natural_group")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
