from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter matched prediction table by score threshold.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--score_threshold", type=float, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.matched_pred_path)
    if "score" not in df.columns:
        raise KeyError("Matched prediction table must contain 'score' column.")

    out = df[df["score"] >= args.score_threshold].copy()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved thresholded matched predictions to: {output_path}")
    print(f"Rows written: {len(out)}")
    print(f"Score threshold: {args.score_threshold}")


if __name__ == "__main__":
    main()
