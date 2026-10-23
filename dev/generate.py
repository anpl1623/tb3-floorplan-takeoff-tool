#!/usr/bin/env python3
"""Generate scanned as-built floor plans and their takeoff ground truth.

For each store seed: lay out the store (dev/plan.py), plot sheet A101 at 200 dpi
(dev/render.py), and simulate the archive scan: printed at a reduction between
95% and 100%, scanned with a skew of up to 0.8 degrees, blur, sensor noise,
uneven lamp shading and JPEG compression, then wrapped in an image-only PDF.

  --examples  the stores the agent gets, with their hand takeoffs
  --hidden    the stores the verifier runs the agent's tool on
  --preview   also write PNGs to dev/out/ to look at

Usage: python dev/generate.py [--preview SEED ...]
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
from render import DPI, render

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "floorplan-takeoff-tool"
EXAMPLES = [3101, 3102, 3103]
HIDDEN = [4201, 4202, 4203, 4204, 4205, 4206]


def scan(gray, rng):
    """Print at a reduction, scan with skew and noise. Returns the scan and its parameters."""
    print_scale = float(rng.uniform(0.95, 1.0))
    skew = float(rng.uniform(-0.8, 0.8))
    h, w = gray.shape
    reduced = cv2.resize(gray, (round(w * print_scale), round(h * print_scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w), 255, np.uint8)
    oy = int(rng.integers(10, h - reduced.shape[0] + 1)) if h > reduced.shape[0] + 10 else 0
    ox = int(rng.integers(10, w - reduced.shape[1] + 1)) if w > reduced.shape[1] + 10 else 0
    canvas[oy:oy + reduced.shape[0], ox:ox + reduced.shape[1]] = reduced
    m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    rotated = cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    out = cv2.GaussianBlur(rotated.astype(np.float32), (0, 0), float(rng.uniform(0.55, 0.8)))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.0 - float(rng.uniform(0.01, 0.05)) * (xx / w) - float(rng.uniform(0.0, 0.03)) * (yy / h)
    out = 255 - (255 - out) * float(rng.uniform(0.88, 0.96))      # toner is not quite black
    out = out * shade + rng.normal(0, float(rng.uniform(3.0, 5.0)), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), dict(print_scale=print_scale, skew=skew)


def make(seed):
    p = plan_mod.build(seed)
    gray = render(p, seed=seed)
    rng = np.random.default_rng(seed)
    scanned, params = scan(gray, rng)
    buf = io.BytesIO()
    Image.fromarray(scanned).save(buf, "JPEG", quality=85, dpi=(DPI, DPI))
    page = Image.open(io.BytesIO(buf.getvalue()))
    return p, page, params


def write_pdf(page, path, store):
    page.save(path, "PDF", resolution=DPI, title=f"Store {store} A101 as-built", author="", creator="", producer="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, nargs="*")
    args = ap.parse_args()
    if args.preview:
        out = ROOT / "dev" / "out"
        out.mkdir(exist_ok=True)
        for seed in args.preview:
            p, page, params = make(seed)
            page.save(out / f"preview_{seed}.png")
            print(seed, params, json.dumps(plan_mod.ground_truth(p)))
        return

    examples = TASK / "environment" / "data" / "examples"
    hidden = TASK / "tests" / "stores"
    for d in (examples, hidden):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for seed in EXAMPLES:
        p, page, params = make(seed)
        name = f"store-{p.store}"
        write_pdf(page, examples / f"{name}.pdf", p.store)
        (examples / f"{name}.takeoff.json").write_text(json.dumps(plan_mod.ground_truth(p), indent=2) + "\n")
        print("example", seed, name, params)
    for seed in HIDDEN:
        p, page, params = make(seed)
        write_pdf(page, hidden / f"{seed}.pdf", p.store)
        print("hidden", seed, p.store, params)
    # the verifier rebuilds each hidden store's vector model from its seed and takes the quantities off it
    shutil.copyfile(ROOT / "dev" / "plan.py", TASK / "tests" / "plan.py")


if __name__ == "__main__":
    main()
