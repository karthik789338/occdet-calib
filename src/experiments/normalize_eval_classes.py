from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Normalize class_id to class_name strings for evaluation.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv).copy()
    if "class_name" not in df.columns:
        raise KeyError("class_name column is required")

    df["class_id"] = df["class_name"].astype(str)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print("Saved:", args.output_csv)
    print(df.head())


if __name__ == "__main__":
    main()
