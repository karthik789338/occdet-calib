from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def label_image(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, 140, 30), fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--ann_id", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)

    if args.ann_id is not None:
        sub = df[df["ann_id"] == args.ann_id].copy()
    else:
        counts = df.groupby("ann_id")["nominal_occlusion"].nunique().reset_index()
        good = counts[counts["nominal_occlusion"] >= 5]["ann_id"].tolist()
        if not good:
            raise RuntimeError("No annotation with all 5 occlusion levels found.")
        sub = df[df["ann_id"] == good[0]].copy()

    sub = sub.sort_values("nominal_occlusion")
    levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    rows = []
    for lv in levels:
        row = sub[sub["nominal_occlusion"] == lv]
        if len(row) == 0:
            continue
        rows.append(row.iloc[0])

    if not rows:
        raise RuntimeError("No montage rows found.")

    images = []
    for r in rows:
        path = Path(r["variant_image_path"])
        img = Image.open(path).convert("RGB")
        img = img.resize((320, 240))
        img = label_image(img, f"occ={r['nominal_occlusion']:.1f}")
        images.append(img)

    total_w = 320 * len(images)
    total_h = 240
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    for i, img in enumerate(images):
        canvas.paste(img, (320 * i, 0))

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Saved montage to: {out}")


if __name__ == "__main__":
    main()
