#!/usr/bin/env python3
"""Generate the store #0417 drawing set and its takeoff ground truth.

  * renders the floor plan as a 36" x 24" plot at 200 dpi (dev/render.py);
  * simulates a scanned print: the print was reduced to 97.3 %, then scanned
    at 200 dpi with a small skew, blur, sensor noise and JPEG compression;
  * writes the image-only PDF the agent receives, and copies the vector plan
    model (dev/plan.py) into the verifier, which derives the true quantities
    from it.

Usage: python dev/generate.py [--debug]
"""

import argparse
import io
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import plan as plan_mod
from render import DPI, render, render_rcp, render_redlines, render_schedules

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "store-remodel-takeoff"
# Each sheet was printed and scanned separately: (print reduction, skew in degrees, paper offset px).
SCANS = {"A101": (0.973, -0.38, (52, 71)), "A102": (0.964, 0.21, (60, 88)), "A601": (0.991, -0.55, (31, 40))}
SEED = 417


def scan(img, rng, print_scale, skew, offset):
    """img: H x W x 3 print (black linework multiplied by the red markups)."""
    h, w = img.shape[:2]
    reduced = cv2.resize(img, (round(w * print_scale), round(h * print_scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 255, np.uint8)
    oy, ox = offset
    canvas[oy:oy + reduced.shape[0], ox:ox + reduced.shape[1]] = reduced
    m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    rotated = cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    out = cv2.GaussianBlur(rotated.astype(np.float32), (0, 0), 0.65)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = (1.0 - 0.035 * (xx / w) - 0.02 * (yy / h))[..., None]  # uneven scanner lamp
    out = 255 - (255 - out) * 0.93                                 # toner is not quite black
    out = out * shade + rng.normal(0, 4.0, out.shape[:2])[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="also write clean and scanned PNGs to dev/out/")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    p = plan_mod.build()
    truth = plan_mod.ground_truth(plan_mod.asbuilt(p))
    prints = {"A101": render(p), "A102": render_rcp(p), "A601": render_schedules(p)}
    pages = []
    for name, gray in prints.items():
        print_ = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
        red = render_redlines(name).astype(np.float32)
        clean = (print_ * red / 255.0).astype(np.uint8)          # ink on top of toner
        scanned = scan(clean, rng, *SCANS[name])
        buf = io.BytesIO()
        Image.fromarray(scanned).save(buf, "JPEG", quality=85, subsampling=0, dpi=(DPI, DPI))
        pages.append(Image.open(io.BytesIO(buf.getvalue())))
        if args.debug:
            out = ROOT / "dev" / "out"
            out.mkdir(exist_ok=True)
            pages[-1].save(out / f"scan_{name}.jpg")
    drawings = TASK / "environment" / "data" / "drawings"
    drawings.mkdir(parents=True, exist_ok=True)
    for old in drawings.glob("*.pdf"):
        old.unlink()
    pages[0].save(drawings / "store-0417-asbuilt.pdf", "PDF", resolution=DPI, save_all=True, append_images=pages[1:],
                  title="Store 0417 as-built set", author="", creator="", producer="")

    # the verifier derives the expected quantities from the same vector plan model
    shutil.copyfile(ROOT / "dev" / "plan.py", TASK / "tests" / "plan.py")
    print(json.dumps(truth))


if __name__ == "__main__":
    main()
