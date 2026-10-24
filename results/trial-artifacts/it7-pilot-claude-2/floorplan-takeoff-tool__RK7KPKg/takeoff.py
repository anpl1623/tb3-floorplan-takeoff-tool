#!/usr/bin/env python3
"""Quantity takeoff from a scanned A101 fixture & finish plan.

usage: python3 takeoff.py <scan.pdf> <takeoff.json>
"""
import os
import sys
import json
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

cv2.setNumThreads(2)

import prep  # noqa: E402
import layout  # noqa: E402
import walls  # noqa: E402
import doors  # noqa: E402
import floors  # noqa: E402
import fixtures  # noqa: E402

FLOOR_KEYS = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']
WALL_KEYS = ['EXT', 'P1', 'P2', 'P3']
DOOR_KEYS = ['single', 'pair', 'overhead_coiling', 'auto_sliding']
FIX_KEYS = ['gondola_sections_4ft', 'gondola_sections_3ft', 'endcaps', 'checkouts',
            'self_checkouts', 'cooler_doors', 'produce_tables']

VERBOSE = os.environ.get('TAKEOFF_VERBOSE', '') not in ('', '0')


def log(*a):
    if VERBOSE:
        print(*a, file=sys.stderr)


def empty_result():
    return {
        'floor_sf': {k: 0.0 for k in FLOOR_KEYS},
        'wall_lf': {k: 0.0 for k in WALL_KEYS},
        'doors': {k: 0 for k in DOOR_KEYS},
        'fixtures': {k: 0 for k in FIX_KEYS},
    }


def run(pdf_path):
    t0 = time.time()
    res = empty_result()
    img = prep.load_image(pdf_path)
    img, angle = prep.deskew(img)
    ink = prep.normalize(img)
    sigma = prep.noise_sigma(img, ink)
    tink = prep.texture_ink(ink, sigma)
    log('prep %.1fs angle %.3f sigma %.4f' % (time.time() - t0, angle, sigma))

    ppf, sb = layout.find_scale_bar(ink)
    leg = layout.find_legend(ink)
    if leg is None:
        raise RuntimeError('legend not found')
    ppf_leg = layout.scale_from_legend(leg)
    if ppf is None or (ppf_leg is not None and abs(ppf / ppf_leg - 1) > 0.08):
        log('scale bar %s inconsistent/missing, legend scale %s' % (ppf, ppf_leg))
        ppf = ppf_leg if ppf_leg is not None else 200.0 / 8.0
    log('scale %.4f px/ft  legend at %s' % (ppf, leg['floor'][0]))

    H, W = ink.shape
    x1 = max(1000, leg['floor'][0][0] - 60)
    roi = (int(0.01 * W), int(0.01 * H), x1, int(0.99 * H))
    fr, lines, pm = walls.analyze_walls(ink, ppf, leg, roi)
    walls.build_runs(ink, fr, lines)
    lines = walls.filter_lines(lines)
    walls.build_runs(ink, fr, lines)
    walls.classify_pieces(fr, lines, pm)
    wl = walls.wall_lengths(lines)
    for k in WALL_KEYS:
        res['wall_lf'][k] = round(float(wl.get(k, 0.0)), 2)
    log('walls %.1fs %s' % (time.time() - t0, res['wall_lf']))

    try:
        dres, _ = doors.detect_doors(ink, fr, lines, pm)
        for k in DOOR_KEYS:
            res['doors'][k] = int(dres.get(k, 0))
    except Exception:
        log(traceback.format_exc())
    log('doors %.1fs %s' % (time.time() - t0, res['doors']))

    tau = walls.calibrate_tau(ink, leg['parts'])
    occ = None
    cv = floors.Canvas(fr, shape=ink.shape)
    try:
        Wm, lab, st, inside = floors.rooms(fr, lines, cv)
        fx, occ = fixtures.detect_multi(cv.crop(tink), inside[lab], fr.px)
        for k in FIX_KEYS:
            res['fixtures'][k] = int(fx.get(k, 0))
    except Exception:
        log(traceback.format_exc())
    try:
        cases = fixtures.cooler_cases(tink, fr, lines)
        if cases:
            res['fixtures']['cooler_doors'] = int(sum(c[5] for c in cases))
            cm = fixtures.case_rects_mask((cv.h, cv.w), cv.x0, cv.y0, fr, cases)
            occ = cm if occ is None else (occ | cm)
        log('cooler cases: %s' % [(round(c[2], 1), round(c[3], 1), c[5]) for c in cases])
    except Exception:
        log(traceback.format_exc())
    log('fixtures %.1fs %s' % (time.time() - t0, res['fixtures']))

    try:
        areas = floors.finish_areas(ink, tink, fr, lines, leg, tau, occluder=occ)
        for i, k in enumerate(FLOOR_KEYS):
            res['floor_sf'][k] = round(float(areas.get(i, 0.0)), 1)
    except Exception:
        log(traceback.format_exc())
    log('floors %.1fs %s' % (time.time() - t0, res['floor_sf']))
    return res


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    pdf, out = sys.argv[1], sys.argv[2]
    try:
        res = run(pdf)
    except Exception:
        log(traceback.format_exc())
        res = empty_result()
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
