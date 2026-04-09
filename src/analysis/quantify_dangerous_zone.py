from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify candidate Dangerous Zone levels.")
    parser.add_argument("--perf_csv", type=str, required=True)
    parser.add_argument("--dece_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--recall_floor", type=float, default=0.60)
    parser.add_argument("--dece_floor", type=float, default=0.45)
    parser.add_argument("--precision_floor", type=float, default=0.15)
    args = parser.parse_args()

    perf = pd.read_csv(args.perf_csv)
    dece = pd.read_csv(args.dece_csv)

    merged = perf.merge(
        dece[["occlusion_level", "dece", "monotonic", "inversion_count"]],
        on="occlusion_level",
        how="inner",
    )

    merged["dangerous_zone"] = (
        (merged["recall"] >= args.recall_floor) &
        (merged["dece"] >= args.dece_floor) &
        (merged["precision"] >= args.precision_floor)
    )

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print(merged)
    print("\nDangerous-zone candidates:")
    print(merged[merged["dangerous_zone"] == True])


if __name__ == "__main__":
    main()
