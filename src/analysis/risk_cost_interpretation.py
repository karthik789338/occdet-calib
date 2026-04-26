from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Simple risk-cost interpretation from matched predictions.")
    parser.add_argument("--matched_pred_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--score_column", default="score")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    parser.add_argument("--fp_cost", type=float, default=1.0)
    parser.add_argument("--fn_cost", type=float, default=5.0)
    parser.add_argument("--abstain_cost", type=float, default=0.25)
    args = parser.parse_args()

    pred = pd.read_csv(args.matched_pred_path).copy()
    gt = pd.read_csv(args.gt_path).copy()

    if args.score_column not in pred.columns:
        raise KeyError(f"Missing score column: {args.score_column}")

    keep_cols = [c for c in ["image_key", "nominal_occlusion"] if c in gt.columns]
    gt_small = gt[keep_cols].drop_duplicates("image_key") if keep_cols else None

    if gt_small is not None and len(gt_small):
        pred = pred.merge(gt_small, on="image_key", how="left")

    rows = []

    # overall
    total_gt_all = int(len(gt))
    total_pred_all = int(len(pred))

    for thr in args.thresholds:
        sub = pred[pred[args.score_column] >= thr].copy()

        tp = int((sub["correct"] == 1).sum())
        fp = int((sub["correct"] == 0).sum())
        fn = max(0, total_gt_all - tp)
        abstained = max(0, total_pred_all - len(sub))

        total_cost = args.fp_cost * fp + args.fn_cost * fn + args.abstain_cost * abstained
        rows.append(
            {
                "scope": "overall",
                "occlusion_level": "all",
                "score_column": args.score_column,
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "abstained_predictions": abstained,
                "total_cost": total_cost,
                "cost_per_gt": total_cost / max(1, total_gt_all),
            }
        )

    # by occlusion
    if "nominal_occlusion" in pred.columns:
        for occ, pred_sub in pred.groupby("nominal_occlusion"):
            gt_count = int(gt[gt["nominal_occlusion"] == occ].shape[0]) if "nominal_occlusion" in gt.columns else 0
            total_pred = int(len(pred_sub))

            for thr in args.thresholds:
                kept = pred_sub[pred_sub[args.score_column] >= thr].copy()

                tp = int((kept["correct"] == 1).sum())
                fp = int((kept["correct"] == 0).sum())
                fn = max(0, gt_count - tp)
                abstained = max(0, total_pred - len(kept))

                total_cost = args.fp_cost * fp + args.fn_cost * fn + args.abstain_cost * abstained
                rows.append(
                    {
                        "scope": "by_occlusion",
                        "occlusion_level": occ,
                        "score_column": args.score_column,
                        "threshold": thr,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "abstained_predictions": abstained,
                        "total_cost": total_cost,
                        "cost_per_gt": total_cost / max(1, gt_count),
                    }
                )

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(out)


if __name__ == "__main__":
    main()
