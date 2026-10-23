#!/usr/bin/env python3
"""Raster quantity takeoff for Cedar Valley Market A101 archive sheets."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import sys
import json
import math
import argparse
import tempfile
from pathlib import Path

import cv2
import fitz
import numpy as np

from geometry import detect_walls, draw_walls
from floors import measure_floors
from doors import recognize_doors
from symbols import recognize_fixtures, fixture_exclusion

cv2.setNumThreads(2)
PPF = 24.0


def groups(mask):
    """Half-open intervals of true values in a one-dimensional array."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def image_from_pdf(path):
    with fitz.open(path) as doc:
        page = doc[0]
        images = page.get_images(full=True)
        if images:
            info = max(images, key=lambda v: v[2] * v[3])
            if info[2] * info[3] > 2_000_000:
                raw = doc.extract_image(info[0])["image"]
                return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), colorspace=fitz.csGRAY)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def deskew(a):
    factor = min(1., 3600. / max(a.shape))
    small = cv2.resize(a, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    lines = cv2.createLineSegmentDetector(0).detect(small)[0]
    if lines is None:
        raise ValueError("No drawing lines found")
    ls = lines[:, 0, :]
    dx, dy = ls[:, 2] - ls[:, 0], ls[:, 3] - ls[:, 1]
    length = np.hypot(dx, dy)
    good = (abs(dx) > abs(dy) * 4) & (length > small.shape[1] * .25)
    ix = np.flatnonzero(good)
    ix = ix[np.argsort(length[ix])[-12:]]
    if len(ix) < 2:
        raise ValueError("Could not determine the drawing's horizontal axis")
    slope = float(np.median(dy[ix] / dx[ix]))
    vertical = (abs(dy) > abs(dx)*4) & (length > small.shape[0]*.25)
    iy = np.flatnonzero(vertical)
    iy = iy[np.argsort(length[iy])[-12:]]
    vslope = float(np.median(dx[iy]/dy[iy])) if len(iy) else -slope
    angle = math.degrees(math.atan(slope))
    # Estimate the two axes separately: a scanner can introduce a small shear.
    linear = np.array([[1., -vslope], [-slope, 1.]])/math.sqrt(1-slope*vslope)
    center = np.array([a.shape[1]/2, a.shape[0]/2])
    matrix = np.c_[linear, center-linear@center]
    return cv2.warpAffine(a, matrix, (a.shape[1], a.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=255), angle


def graphic_scale(a):
    # The standard bar has black 0--4 and 8--16 foot intervals. Their
    # centroids are ten feet apart; using centroids avoids outline thickness.
    shrink = max(1, round(a.shape[1] / 7200 * 2))
    ink = (a < 110).astype(np.uint8)
    ink = cv2.erode(ink, np.ones((2 * shrink + 1, 2 * shrink + 1), np.uint8))
    n, _, stats, centers = cv2.connectedComponentsWithStats(ink)
    candidates = []
    height, width = a.shape
    for st, center in zip(stats[1:], centers[1:]):
        x, y, w, h, area = st
        if (.15 * width < x < .65 * width and .7 * height < y < .96 * height
                and .0015 * height < h < .012 * height and 2.5 < w / h < 12
                and area > .85 * w * h):
            candidates.append((st, center))
    pairs = []
    for s, c in candidates:
        for t, d in candidates:
            if (d[0] > c[0] and abs(c[1] - d[1]) < max(3, s[3] * .15)
                    and 1.7 < t[2] / s[2] < 2.3 and abs(t[3] - s[3]) < 4
                    and 2.2 < (d[0] - c[0]) / s[2] < 2.9):
                pairs.append((s[4] + t[4], (d[0] - c[0]) / 10.))
    if not pairs:
        raise ValueError("Could not locate the graphic scale")
    return max(pairs)[1]


def peak_centers(projection, threshold):
    return [float(np.average(np.arange(l, r), weights=projection[l:r]))
            for l, r in groups(projection >= threshold)]


def exterior(a):
    height, width = a.shape
    ink = (a < 125).astype(np.uint8)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 100), np.uint8))
    hp = h[:, int(width * .06):int(width * .64)].sum(1).astype(float)
    ys = peak_centers(hp, min(1300, hp.max()*.4))
    # A paired eight-inch wall face distinguishes the building from the sheet
    # frame and dimension strings, even when the print is reduced on the page.
    ypairs = [(y, z) for i, y in enumerate(ys) for z in ys[i+1:] if 13.5 < z-y < 18.5]
    if len(ypairs) < 2:
        raise ValueError("Could not locate the building's exterior horizontal walls")
    top = min(v[0] for v in ypairs)
    bottom = max(v[1] for v in ypairs)
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((100, 1), np.uint8))
    vp = v[round(top):round(bottom), :].sum(0).astype(float)
    vp[:int(width * .045)] = 0
    vp[int(width * .65):] = 0
    xs = peak_centers(vp, (bottom-top)*.4)
    xpairs = [(x, z) for i, x in enumerate(xs) for z in xs[i+1:] if 13.5 < z-x < 18.5]
    if len(xpairs) < 2:
        raise ValueError("Could not locate the building's exterior vertical walls")
    return np.array([min(v[0] for v in xpairs), top, max(v[1] for v in xpairs), bottom])


def load_plan(path):
    original = image_from_pdf(path)
    if np.median(original) < 100:
        original = 255-original
    if original.shape[0] > original.shape[1]:
        original = np.ascontiguousarray(np.rot90(original))
    for turn in [0, 2]:
        try:
            oriented = original if turn == 0 else np.ascontiguousarray(np.rot90(original, 2))
            a, angle = deskew(oriented)
            scale = graphic_scale(a)
            break
        except ValueError:
            if turn == 2:
                raise
    a = cv2.resize(a, None, fx=PPF / scale, fy=PPF / scale, interpolation=cv2.INTER_CUBIC)
    box = exterior(a)
    margin = 4 * int(PPF)
    x0, y0, x1, y1 = np.rint(box).astype(int)
    crop = a[y0-margin:y1+margin+1, x0-margin:x1+margin+1].copy()
    box = box - [x0-margin, y0-margin, x0-margin, y0-margin]
    aspect = grid_aspect(crop)
    if abs(aspect-1.) > .006:
        crop = cv2.resize(crop, None, fx=1., fy=1/aspect, interpolation=cv2.INTER_CUBIC)
        box[[1, 3]] /= aspect
    # Normalize the paper and darkest printed ink rather than tying recognition
    # thresholds to the scanner's brightness setting.
    black, paper = np.percentile(crop, [.001, 90])
    lut = np.clip((np.arange(256)-black)*234/max(100, paper-black)+20, 0, 255).astype(np.uint8)
    crop = cv2.LUT(crop, lut)
    return crop, box, {
        "scan_pixels_per_foot": scale, "rotation_degrees": angle,
        "vertical_stretch": aspect,
        "outer_width_ft": float((box[2] - box[0]) / PPF),
        "outer_height_ft": float((box[3] - box[1]) / PPF)}


def grid_aspect(a):
    """Recover nonuniform reproduction from the one-foot square VCT grid."""
    ink = (a < 225).astype(np.uint8)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 36), np.uint8))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((36, 1), np.uint8))
    estimates = []
    for y in range(12, a.shape[0]-160, 128):
        for x in range(12, a.shape[1]-160, 128):
            ys = peak_centers(h[y:y+160, x:x+160].mean(1), .4)
            xs = peak_centers(v[y:y+160, x:x+160].mean(0), .4)
            if min(len(xs), len(ys)) < 5:
                continue
            dx, dy = np.diff(xs), np.diff(ys)
            if not (20 < np.median(dx) < 28 and 18 < np.median(dy) < 31
                    and np.std(dx) < 1.4 and np.std(dy) < 1.4):
                continue
            sx = np.polyfit(np.arange(len(xs)), xs, 1)[0]
            sy = np.polyfit(np.arange(len(ys)), ys, 1)[0]
            estimates.append(sy/sx)
    return float(np.median(estimates)) if len(estimates) >= 3 else 1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan")
    parser.add_argument("output")
    parser.add_argument("--debug", help="Directory for diagnostic images and measurements")
    args = parser.parse_args()
    a, box, meta = load_plan(args.scan)
    out = None
    if args.debug:
        out = Path(args.debug)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "plan.png"), a)
        (out / "calibration.json").write_text(json.dumps(dict(meta, box=box.tolist()), indent=2))
    walls = detect_walls(a, box)
    fixtures = recognize_fixtures(a, out)
    excluded = fixture_exclusion(a, fixtures)
    floor_sf, floor_regions = measure_floors(a, walls, box, out, excluded)
    doors, door_info = recognize_doors(a, walls, out)
    result = {
        "floor_sf": {k: round(float(v), 1) for k, v in floor_sf.items()},
        "wall_lf": {k: round(sum((w.hi-w.lo)/PPF for w in walls if w.kind == k), 2)
                    for k in ["EXT", "P1", "P2", "P3"]},
        "doors": doors,
        "fixtures": {k: len(fixtures[k]) for k in ["gondola_sections_4ft", "gondola_sections_3ft",
                     "endcaps", "checkouts", "self_checkouts", "cooler_doors", "produce_tables"]}}
    # Do not leave a partially written takeoff if recognition fails.
    destination = Path(args.output)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.name+".", suffix=".tmp", delete=False) as f:
            temporary = f.name
            json.dump(result, f, indent=2)
            f.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    if out is not None:
        cv2.imwrite(str(out / "walls.png"), draw_walls(a, walls))
        report = dict(calibration=meta, walls=[w.record() for w in walls],
                      floor_regions=floor_regions, doors=door_info, fixtures=fixtures)
        (out / "measurements.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
