from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare overlap vs distractor summaries.")
    parser.add_argument("--overlap_csv", type=str, required=True)
    parser.add_argument("--distractor_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    overlap = pd.read_csv(args.overlap_csv).copy()
    distractor = pd.read_csv(args.distractor_csv).copy()

    overlap["condition"] = "overlap"
    distractor["condition"] = "distractor"

    out = pd.concat([overlap, distractor], ignore_index=True)
    out.to_csv(args.output_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
