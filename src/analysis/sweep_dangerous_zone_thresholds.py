from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sweep Dangerous Zone thresholds.")
    parser.add_argument("--perf_csv", required=True)
    parser.add_argument("--dece_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--rho_values", type=float, nargs="+", default=[0.50, 0.60, 0.70])
    parser.add_argument("--pi_values", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    parser.add_argument("--delta_percentiles", type=float, nargs="+", default=[50, 60, 70, 80])
    args = parser.parse_args()

    perf = pd.read_csv(args.perf_csv).copy()
    dece = pd.read_csv(args.dece_csv).copy()

    merged = perf.merge(
        dece[["occlusion_level", "dece", "monotonic", "inversion_count"]],
        on="occlusion_level",
        how="inner",
    ).sort_values("occlusion_level")

    if len(merged) == 0:
        raise RuntimeError("No rows after merge.")

    delta_lookup = {}
    for p in args.delta_percentiles:
        delta_lookup[p] = float(np.percentile(merged["dece"].to_numpy(), p))

    rows = []
    for rho in args.rho_values:
        for pi in args.pi_values:
            for p in args.delta_percentiles:
                delta = delta_lookup[p]
                for _, row in merged.iterrows():
                    dangerous = (
                        (float(row["recall"]) >= float(rho))
                        and (float(row["precision"]) >= float(pi))
                        and (float(row["dece"]) >= float(delta))
                    )

                    rows.append(
                        {
                            "rho": rho,
                            "pi": pi,
                            "delta_percentile": p,
                            "delta_floor": delta,
                            "occlusion_level": float(row["occlusion_level"]),
                            "precision": float(row["precision"]),
                            "recall": float(row["recall"]),
                            "f1": float(row["f1"]),
                            "dece": float(row["dece"]),
                            "monotonic": bool(row["monotonic"]),
                            "inversion_count": int(row["inversion_count"]),
                            "dangerous_zone": bool(dangerous),
                        }
                    )

    out = pd.DataFrame(rows)
    summary = (
        out.groupby(["rho", "pi", "delta_percentile", "delta_floor"], as_index=False)
        .agg(
            num_flagged=("dangerous_zone", "sum"),
            flagged_levels=("occlusion_level", lambda x: ",".join(
                [f"{v:.1f}" for v, dz in zip(x, out.loc[x.index, "dangerous_zone"]) if dz]
            )),
        )
        .sort_values(["rho", "pi", "delta_percentile"])
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    print("Saved detailed sweep:", args.output_csv)
    print("Saved summary sweep:", args.summary_csv)
    print(summary)


if __name__ == "__main__":
    main()
