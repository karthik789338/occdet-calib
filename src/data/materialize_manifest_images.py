from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Materialize manifest images into a flat folder via symlink or copy.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for _, row in df.iterrows():
        src = Path(str(row["image_path"]))
        dst = out_dir / Path(str(row["image_key"])).name
        if not src.exists():
            continue
        if dst.exists():
            continue

        if args.mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
        created += 1

    print(f"Created {created} files in {out_dir}")


if __name__ == "__main__":
    main()
