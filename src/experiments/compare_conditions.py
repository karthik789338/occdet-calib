from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple condition summary CSVs.")
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    if len(args.csv_paths) != len(args.labels):
        raise ValueError("csv_paths and labels must have the same length.")

    dfs = []
    for path, label in zip(args.csv_paths, args.labels):
        df = pd.read_csv(path).copy()
        df["condition"] = label
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
