from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


def ensure_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB") if im.mode != "RGB" else im


def pil_to_np(im: Image.Image) -> np.ndarray:
    return np.array(im)


def build_diff_mask(
    base_img: Image.Image,
    overlap_img: Image.Image,
    diff_threshold: int = 18,
    dilate_kernel: int = 9,
    dilate_iters: int = 2,
    blur_kernel: int = 9,
) -> np.ndarray:
    a = pil_to_np(ensure_rgb(base_img))
    b = pil_to_np(ensure_rgb(overlap_img))

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    mask = (diff >= diff_threshold).astype(np.uint8) * 255

    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iters)
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    return mask


def crop_from_mask(mask: np.ndarray, margin: int = 32) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 16)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    h, w = mask.shape

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return int(x1), int(y1), int(x2), int(y2)


def resize_pair(image: Image.Image, mask: Image.Image, size: int = 512) -> tuple[Image.Image, Image.Image]:
    return (
        image.resize((size, size), Image.Resampling.BILINEAR),
        mask.resize((size, size), Image.Resampling.NEAREST),
    )


def load_inpaint_pipeline(model_id: str):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if use_cuda else None,
        use_safetensors=True,
    )
    if use_cuda:
        pipe = pipe.to("cuda")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Generate diffusion-inpainted subset from overlap variants.")
    parser.add_argument("--overlap_metadata_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_image_dir", type=str, required=True)
    parser.add_argument("--output_metadata_csv", type=str, required=True)
    parser.add_argument("--sample_fraction", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--diff_threshold", type=int, default=18)
    parser.add_argument("--crop_margin", type=int, default=40)
    parser.add_argument("--resize_size", type=int, default=512)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    meta = pd.read_csv(args.overlap_metadata_csv).copy()
    if "nominal_occlusion" not in meta.columns:
        raise KeyError("overlap_metadata_csv must contain nominal_occlusion")
    if "variant_image_path" not in meta.columns:
        raise KeyError("overlap_metadata_csv must contain variant_image_path")
    if "image_key" not in meta.columns:
        raise KeyError("overlap_metadata_csv must contain image_key")

    meta = meta[meta["nominal_occlusion"] > 0.0].copy()
    if len(meta) == 0:
        raise RuntimeError("No occluded overlap rows found in overlap_metadata_csv.")

    sample_n = max(1, int(round(len(meta) * args.sample_fraction)))
    sample_n = min(sample_n, len(meta))
    meta = meta.sample(n=sample_n, random_state=args.random_seed).copy()

    output_dir = Path(args.output_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading inpainting pipeline: {args.model_id}")
    pipe = load_inpaint_pipeline(args.model_id)

    rows = []
    success = 0
    skipped = 0

    for idx, row in meta.iterrows():
        image_key = str(row["image_key"])
        overlap_path = Path(str(row["variant_image_path"]))
        base_path = Path(args.image_root) / image_key

        if not overlap_path.exists() or not base_path.exists():
            skipped += 1
            continue

        try:
            base_img = ensure_rgb(Image.open(base_path))
            overlap_img = ensure_rgb(Image.open(overlap_path))

            mask_np = build_diff_mask(
                base_img=base_img,
                overlap_img=overlap_img,
                diff_threshold=args.diff_threshold,
            )
            crop_box = crop_from_mask(mask_np, margin=args.crop_margin)
            if crop_box is None:
                skipped += 1
                continue

            x1, y1, x2, y2 = crop_box
            if (x2 - x1) < 16 or (y2 - y1) < 16:
                skipped += 1
                continue

            overlap_crop = overlap_img.crop((x1, y1, x2, y2))
            mask_crop = Image.fromarray(mask_np[y1:y2, x1:x2]).convert("L")

            small_img, small_mask = resize_pair(
                overlap_crop,
                mask_crop,
                size=args.resize_size,
            )

            if torch.cuda.is_available():
                generator = torch.Generator(device="cuda").manual_seed(args.random_seed + success)
            else:
                generator = torch.Generator().manual_seed(args.random_seed + success)

            prompt = "a realistic photorealistic natural scene with a plausible foreground occluder"

            out = pipe(
                prompt=prompt,
                image=small_img,
                mask_image=small_mask,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            out = ensure_rgb(out).resize(overlap_crop.size, Image.Resampling.BILINEAR)

            full = overlap_img.copy()
            full.paste(out, (x1, y1))

            stem = overlap_path.stem + "_inp"
            save_path = output_dir / f"{stem}.png"
            full.save(save_path)

            new_row = row.to_dict()
            new_row["variant_type"] = "inpainting_overlap"
            new_row["variant_image_path"] = str(save_path)
            rows.append(new_row)
            success += 1

            if success % 10 == 0:
                print(f"Generated {success}/{sample_n} inpainted samples")

        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping row idx={idx}, image_key={image_key}: {e}")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_metadata_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Requested subset size: {sample_n}")
    print(f"Generated inpainting subset: {len(out_df)}")
    print(f"Skipped: {skipped}")
    print(f"Saved metadata to: {out_path}")


if __name__ == "__main__":
    main()
