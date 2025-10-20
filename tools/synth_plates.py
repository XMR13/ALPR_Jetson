#!/usr/bin/env python3
"""
Generate synthetic Indonesian license plate crops + labels.

This script renders plausible Indonesian plate strings with PIL,
applies camera-like augmentations (perspective, rotation, blur, JPEG artifacts,
morphology, bolts, etc.), and writes:
  - crops/*.jpg
  - labels_train.csv, labels_val.csv
  - charset.txt

Example
  python tools/synth_plates.py \
    --outdir data/ocr/synth \
    --count 50000 \
    --fonts-dir assets/fonts \
    --width 160 --height 32 \
    --seed 123

Notes
- Provide 1–3 sans/condensed fonts similar to plate fonts in --fonts-dir (TTF/OTF).
- Outputs 32x160 by default; tune --height/--width for your model.
- Labels format: "filename,text".
"""

from __future__ import annotations

import argparse
import math
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import cv2  # type: ignore


# A broader (still not exhaustive) set of prefixes across regions
ALLOWED_PREFIX = [
    # Jakarta/West/Central/East Java & DIY
    "A","B","D","E","F","T","Z","G","H","K","R","AA","AB","AD","AE",
    # East Java
    "L","M","N","P","S","W",
    # Sumatra (examples)
    "BA","BB","BD","BE","BH","BK","BL","BM","BP","BG","BN",
    # Kalimantan (examples)
    "DA","KB","KT","KU","KH",
    # Sulawesi (examples)
    "DB","DD","DN","DT","DW","DC",
    # Bali/NT/Papua (examples)
    "DK","DR","EA","ED","PA","PB",
]

# Charset for OCR labels
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "


# ---------- Plate string synthesis ----------

def rnd_prefix() -> str:
    # Real plates: 1–2 letters, often from the list above
    return random.choice(ALLOWED_PREFIX)


def rnd_number() -> str:
    # 1–4 digits
    return str(random.randint(1, 9999))


def rnd_suffix() -> str:
    # 0–3 letters, but bias to 2–3
    k = random.choices([0, 1, 2, 3], weights=[0.05, 0.20, 0.5, 0.25], k=1)[0]
    return "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(k))


def maybe_small_region_code(p: float = 0.3) -> str:
    # Optional small 1–2 digits trailing code
    if random.random() < p:
        return f" {random.randint(1, 99)}"
    return ""


def make_plate_text() -> str:
    p, n, s = rnd_prefix(), rnd_number(), rnd_suffix()
    core = f"{p} {n}" if not s else f"{p} {n} {s}"
    return core + maybe_small_region_code(0.30)


# ---------- Font handling & text rendering ----------

def list_font_paths(fonts_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in ("*.ttf", "*.otf"):
        paths.extend(fonts_dir.rglob(ext))
    return paths


def autosize_text_bbox(draw: ImageDraw.ImageDraw, text: str, font_path: Path, target_w: int, target_h: int, pad: int) -> ImageFont.FreeTypeFont:
    # Try descending sizes until it fits within (target_w - 2*pad, target_h - 2*pad)
    size = int(target_h * 0.8)
    size_min = 8
    while size >= size_min:
        ft = ImageFont.truetype(str(font_path), size=size, layout_engine=ImageFont.LAYOUT_BASIC)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=ft)
        tw, th = right - left, bottom - top
        if tw <= (target_w - 2 * pad) and th <= (target_h - 2 * pad):
            return ft
        size -= 1
    # Fallback tiny
    return ImageFont.truetype(str(font_path), size=size_min)


def rounded_rect_mask(w: int, h: int, radius: Optional[int] = None) -> Image.Image:
    if radius is None:
        radius = max(2, min(w, h) // 12)
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    return mask


def render_plate(text: str, font_paths: List[Path], w: int, h: int) -> Image.Image:
    # Background slightly off-white, thin border, rounded corners, centered text, light stroke
    pad = int(h * 0.12)
    bg = random.randint(238, 252)
    plate = Image.new("RGB", (w, h), color=(bg, bg, bg))
    draw = ImageDraw.Draw(plate)

    # Thin border
    draw.rectangle([1, 1, w - 2, h - 2], outline=(55, 55, 55), width=1)

    # Rounded corners
    mask = rounded_rect_mask(w, h)
    plate = Image.composite(plate, Image.new("RGB", (w, h), (bg, bg, bg)), mask)

    # Choose a font and autosize to fit
    base_path = random.choice(font_paths)
    font = autosize_text_bbox(draw, text, base_path, w, h, pad)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    x = (w - tw) // 2
    y = (h - th) // 2

    # Stroke (fake emboss/print bleed)
    stroke_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in stroke_offsets:
        draw.text((x + dx, y + dy), text, font=font, fill=(35, 35, 35))
    draw.text((x, y), text, font=font, fill=(10, 10, 10))

    return plate


# ---------- Augmentations ----------

def persp_jitter(im: Image.Image, max_deg: float = 4.0) -> Image.Image:
    w, h = im.size
    jitter = int(max(w, h) * math.tan(math.radians(max_deg)) * 0.1)

    # Use np.array(..., dtype=np.float32) instead of np.float32([...])
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = src + np.array(
        [
            [random.randint(-jitter, jitter), random.randint(-jitter, jitter)],
            [random.randint(-jitter, jitter), random.randint(-jitter, jitter)],
            [random.randint(-jitter, jitter), random.randint(-jitter, jitter)],
            [random.randint(-jitter, jitter), random.randint(-jitter, jitter)],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    arr = np.array(im)
    warped = cv2.warpPerspective(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def small_rotate(im: Image.Image, deg: float = 3.0) -> Image.Image:
    angle = random.uniform(-deg, deg)
    bg = tuple(im.getpixel((0, 0)))
    return im.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=bg)


def photometric(im: Image.Image) -> Image.Image:
    # random blur
    if random.random() < 0.35:
        if random.random() < 0.7:
            im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
        else:
            # simple motion blur via kernel conv
            arr = np.array(im)
            k = random.choice([3, 5, 7])
            kernel = np.zeros((k, k), dtype=np.float32)
            if random.random() < 0.5:
                kernel[k // 2, :] = 1.0 / k
            else:
                kernel[:, k // 2] = 1.0 / k
            arr = cv2.filter2D(arr, -1, kernel)
            im = Image.fromarray(arr)

    # brightness/contrast
    arr = np.array(im).astype(np.float32)
    alpha = random.uniform(0.9, 1.12)  # contrast
    beta = random.uniform(-12, 12)     # brightness
    arr = np.clip(arr * alpha + beta, 0, 255)

    # noise
    if random.random() < 0.25:
        noise = np.random.normal(scale=random.uniform(2, 6), size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def morph(im: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        arr = np.array(im)
        k = random.choice([1, 1, 1, 3])  # mostly very light
        kernel = np.ones((k, k), np.uint8)
        if random.random() < 0.5:
            arr = cv2.erode(arr, kernel, iterations=1)
        else:
            arr = cv2.dilate(arr, kernel, iterations=1)
        im = Image.fromarray(arr)
    return im


def add_bolts(im: Image.Image) -> Image.Image:
    if random.random() < 0.35:
        draw = ImageDraw.Draw(im)
        w, h = im.size
        r = max(1, h // 12)
        y = h // 2 + random.randint(-h // 12, h // 12)
        for x in [int(w * 0.05), int(w * 0.95)]:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(90, 90, 90), fill=(200, 200, 200))
    return im


def vignette(im: Image.Image) -> Image.Image:
    if random.random() < 0.35:
        w, h = im.size
        # radial vignette mask
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dist = dist / dist.max()
        strength = random.uniform(0.05, 0.15)
        vign = 1.0 - strength * dist
        arr = np.array(im).astype(np.float32)
        arr *= vign[..., None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
    return im


def jpeg_artifacts(im: Image.Image) -> Image.Image:
    # Force an extra JPEG pass to simulate compression
    q = random.randint(40, 90)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------- Pipeline ----------

def generate_plate_image(txt: str, font_paths: List[Path], width: int, height: int) -> Image.Image:
    img = render_plate(txt, font_paths, width, height)
    if random.random() < 0.6:
        img = persp_jitter(img)
    if random.random() < 0.5:
        img = small_rotate(img)
    img = photometric(img)
    if random.random() < 0.3:
        img = morph(img)
    if random.random() < 0.3:
        img = add_bolts(img)
    if random.random() < 0.4:
        img = vignette(img)
    img = jpeg_artifacts(img)
    return img


# ---------- IO & dataset management ----------

def write_charset(outdir: Path, charset: str) -> None:
    (outdir / "charset.txt").write_text(charset + "\n", encoding="utf-8")


def run(outdir: Path, count: int, fonts_dir: Path, width: int, height: int, seed: Optional[int]) -> int:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    crops = outdir / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    font_paths = list_font_paths(fonts_dir)
    if not font_paths:
        raise SystemExit("No fonts found in fonts-dir (need .ttf/.otf).")

    # Dedup strings; oversample attempts to hit 'count' uniques
    rows: List[Tuple[str, str]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(count * 3, 1000)

    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        txt = make_plate_text()
        # Keep charset safety (optional)
        if any(ch not in CHARSET for ch in txt):
            continue
        if txt in seen:
            continue
        seen.add(txt)

        img = generate_plate_image(txt, font_paths, width, height)
        name = f"synth_{len(rows):06d}.jpg"
        img.save(crops / name, format="JPEG", quality=95)
        rows.append((name, txt))

    if len(rows) < count:
        print(f"Warning: generated only {len(rows)} unique strings after {attempts} attempts.")

    # Shuffle & split
    random.shuffle(rows)
    cut = int(0.95 * len(rows))
    splits = [
        ("labels_train.csv", rows[:cut]),
        ("labels_val.csv", rows[cut:]),
    ]
    for fn, part in splits:
        with (outdir / fn).open("w", encoding="utf-8", newline="") as f:
            f.write("filename,text\n")
            for a, b in part:
                # CSV-safe (simple): replace commas in text with space (unlikely here)
                f.write(f"{a},{b.replace(',', ' ')}\n")

    write_charset(outdir, CHARSET)
    print(f"Wrote {len(rows)} crops to {crops} with 95/5 split, plus charset.txt")
    return len(rows)


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic Indonesian plate crops for OCR.")
    p.add_argument("--outdir", required=True, help="Output directory for OCR dataset")
    p.add_argument("--count", type=int, default=50000, help="Number of unique samples to generate")
    p.add_argument("--fonts-dir", required=True, help="Directory of .ttf/.otf fonts")
    p.add_argument("--width", type=int, default=160, help="Output crop width")
    p.add_argument("--height", type=int, default=32, help="Output crop height")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return p


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    fonts_dir = Path(args.fonts_dir)
    if not fonts_dir.exists():
        raise SystemExit("fonts-dir not found")
    outdir.mkdir(parents=True, exist_ok=True)
    n = run(outdir, int(args.count), fonts_dir, int(args.width), int(args.height), args.seed)
    raise SystemExit(0 if n > 0 else 3)


if __name__ == "__main__":
    main()
