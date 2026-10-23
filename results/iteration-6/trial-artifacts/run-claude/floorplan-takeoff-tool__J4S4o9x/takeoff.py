#!/usr/bin/env python3
"""Quantity takeoff from a scanned A101 fixture & finish plan.

usage: python3 takeoff.py <scan.pdf> <takeoff.json>

Pipeline
  1. canon:    deskew, scale to 25 px/ft from the graphic scale bar, normalise levels
  2. walls:    find wall bands (paired outlines + fill type), group into runs,
               measure each run between the centerlines of the walls it meets
  3. floors:   building interior minus walls (openings closed), split by finish
               transition lines, classify each region by its hatch pattern
  4. doors:    openings in wall runs, classified by leaves / sliding panels /
               coil-box dashes
  5. fixtures: rectangles from fixture linework, classified by size and content;
               gondola runs split into sections at their cross lines
"""
import json
import os
import sys
import traceback

sys.dont_write_bytecode = True     # the tool directory may be read-only
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from canon import canonical, PPF  # noqa: E402
import params  # noqa: E402
from walls import find_walls, name_runs, raster_runs  # noqa: E402
from floors import segment, region_features, dot_density, noise_sigma, classify, FINISHES  # noqa: E402
from doors import count_doors, extend_free_ends  # noqa: E402
from fixtures import count_fixtures  # noqa: E402

WALL_TYPES = ['EXT', 'P1', 'P2', 'P3']
DOOR_TYPES = ['single', 'pair', 'overhead_coiling', 'auto_sliding']
FIXTURE_TYPES = ['gondola_sections_4ft', 'gondola_sections_3ft', 'endcaps', 'checkouts',
                 'self_checkouts', 'cooler_doors', 'produce_tables']


def empty_result():
    return {
        'floor_sf': {k: 0.0 for k in FINISHES},
        'wall_lf': {k: 0.0 for k in WALL_TYPES},
        'doors': {k: 0 for k in DOOR_TYPES},
        'fixtures': {k: 0 for k in FIXTURE_TYPES},
    }


def floor_areas(g, runs, outside, bbox, debug=None):
    x0, y0, x1, y1 = bbox
    m = int(2 * PPF)
    H, W = g.shape
    cx0, cy0 = max(int(x0) - m, 0), max(int(y0) - m, 0)
    cx1, cy1 = min(int(x1) + m, W), min(int(y1) + m, H)
    gc = g[cy0:cy1, cx0:cx1]
    wm = raster_runs(runs, (cy1 - cy0, cx1 - cx0), origin=(cx0, cy0))
    interior = (~outside[cy0:cy1, cx0:cx1]) & (wm == 0)
    lab = segment(gc, interior)
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    dots = dot_density(gc, lab, sn=noise_sigma(g))
    areas = {k: 0.0 for k in FINISHES}
    regions = {}
    for i, c in zip(ids, cnt):
        a = c / PPF ** 2
        A = region_features(gc, lab == i)
        cls, sc = classify(A, 100.0 * dots[i] / a)
        regions[int(i)] = dict(id=int(i), area=a, cls=cls)
    # a small blank region surrounded by one hatched region (not by walls) is
    # the inside of a fixture outline: the floor there is the surrounding finish
    for i, rg in regions.items():
        if rg['cls'] not in ('PC-1', 'SC-1') or rg['area'] > 80:
            continue
        m = (lab == i).astype(np.uint8)
        ring = cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(bool) & (m == 0)
        nb = lab[ring]
        if nb.size == 0 or (nb > 0).mean() < 0.7:
            continue
        vals, cts = np.unique(nb[nb > 0], return_counts=True)
        j = int(vals[np.argmax(cts)])
        if cts.max() >= 0.9 * cts.sum() and regions[j]['cls'] not in ('PC-1', 'SC-1'):
            rg['cls'] = regions[j]['cls']
    for rg in regions.values():
        areas[rg['cls']] += rg['area']
    regions = list(regions.values())
    if debug is not None:
        debug['regions'] = regions
        debug['labels'] = (lab, (cx0, cy0))
    return areas


def run(pdf_path, debug=None):
    res = empty_result()
    g, info = canonical(pdf_path)
    params.K = info.get('k', 1.0)
    if debug is not None:
        debug['canon'] = info
        debug['image'] = g
    runs, bbox = find_walls(g)
    outside = name_runs(runs, g.shape, bbox)
    try:
        if extend_free_ends(g, runs):
            outside = name_runs(runs, g.shape, bbox)
    except Exception:
        traceback.print_exc()
    lf = {k: 0.0 for k in WALL_TYPES}
    for r in runs:
        if r.name in lf:
            lf[r.name] += r.length()
    res['wall_lf'] = {k: round(v, 2) for k, v in lf.items()}
    if debug is not None:
        debug['runs'] = runs
        debug['bbox'] = bbox

    try:
        areas = floor_areas(g, runs, outside, bbox, debug)
        res['floor_sf'] = {k: round(v, 1) for k, v in areas.items()}
    except Exception:
        traceback.print_exc()

    try:
        dc, ops = count_doors(g, runs)
        res['doors'] = {k: int(dc[k]) for k in DOOR_TYPES}
        if debug is not None:
            debug['openings'] = ops
    except Exception:
        traceback.print_exc()

    try:
        # work on a crop around the building (counts only; saves memory)
        H_, W_ = g.shape
        cx0, cy0 = max(int(bbox[0]) - 30, 0), max(int(bbox[1]) - 30, 0)
        cx1, cy1 = min(int(bbox[2]) + 30, W_), min(int(bbox[3]) + 30, H_)
        gcrop = np.ascontiguousarray(g[cy0:cy1, cx0:cx1])
        roi = [int(bbox[0]) + 10 - cx0, int(bbox[1]) + 10 - cy0, int(bbox[2]) - 10 - cx0, int(bbox[3]) - 10 - cy0]
        fx, finfo = count_fixtures(gcrop, roi)
        res['fixtures'] = {k: int(fx[k]) for k in FIXTURE_TYPES}
        if debug is not None:
            debug['fixtures'] = finfo
    except Exception:
        traceback.print_exc()
    return res


def main(argv):
    if len(argv) != 3:
        sys.stderr.write('usage: takeoff.py <scan.pdf> <takeoff.json>\n')
        return 2
    pdf_path, out_path = argv[1], argv[2]
    try:
        res = run(pdf_path)
        code = 0
    except Exception:
        traceback.print_exc()
        res = empty_result()
        code = 1
    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=2)
        fh.write('\n')
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv))
