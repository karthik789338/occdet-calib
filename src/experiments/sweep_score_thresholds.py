from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics.map_eval import summarize_detection_metrics
from src.metrics.reliability import summarize_reliability
from src.metrics.monotonicity import compute_monotonicity_from_dataframe


def save_table(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {p.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep score thresholds on matched predictions.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    matched_df = pd.read_csv(args.matched_pred_path)
    gt_df = pd.read_csv(args.gt_path)

    rows = []

    for thr in args.thresholds:
        df_thr = matched_df[matched_df["score"] >= thr].copy()

        overall = summarize_detection_metrics(df_thr, gt_df)
        reliability = summarize_reliability(
            df_thr,
            confidence_col="score",
            correctness_col="correct",
            n_bins=args.n_bins,
        ) if len(df_thr) > 0 else {"dece": None}

        monotonicity = compute_monotonicity_from_dataframe(
            df_thr,
            confidence_col="score",
            correctness_col="correct",
            n_bins=args.n_bins,
        ) if len(df_thr) > 0 else {
            "monotonic": None,
            "inversion_count": None,
        }

        rows.append(
            {
                "score_threshold": thr,
                "num_predictions": len(df_thr),
                "tp": overall["tp"],
                "fp": overall["fp"],
                "fn": overall["fn"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "dece": reliability["dece"],
                "monotonic": monotonicity["monotonic"],
                "inversion_count": monotonicity["inversion_count"],
            }
        )

    out_df = pd.DataFrame(rows)
    save_table(out_df, args.output_path)
    print(out_df)


if __name__ == "__main__":
    main()
