from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep abstention thresholds for raw/global_ts/oc_ts.")
    parser.add_argument("--matched_with_oc_ts_csv", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    args = parser.parse_args()

    df = pd.read_csv(args.matched_with_oc_ts_csv)
    gt = pd.read_csv(args.gt_path)
    total_gt = int(len(gt))
    total_preds = int(len(df))

    mode_to_col = {
        "raw": "score",
        "global_ts": "score_global_ts",
        "oc_ts": "score_oc_ts",
    }

    rows = []

    for mode, col in mode_to_col.items():
        if col not in df.columns:
            continue

        for thr in args.thresholds:
            sub = df[df[col] >= thr].copy()

            tp = int((sub["correct"] == 1).sum())
            fp = int((sub["correct"] == 0).sum())
            fn = max(0, total_gt - tp)

            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, total_gt)
            f1 = safe_div(2 * precision * recall, precision + recall)

            coverage_pred = safe_div(len(sub), total_preds)
            selective_risk = 1.0 - precision if len(sub) > 0 else None

            rows.append(
                {
                    "mode": mode,
                    "threshold": thr,
                    "num_predictions_kept": len(sub),
                    "coverage_pred": coverage_pred,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "selective_risk": selective_risk,
                    "avg_score_kept": float(sub[col].mean()) if len(sub) else None,
                }
            )

    out = pd.DataFrame(rows)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
