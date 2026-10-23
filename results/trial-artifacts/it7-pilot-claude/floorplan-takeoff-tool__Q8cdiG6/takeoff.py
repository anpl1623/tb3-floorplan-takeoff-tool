#!/usr/bin/env python3
"""Quantity takeoff from a scanned A101 fixture & finish plan.

usage: python3 takeoff.py <scan.pdf> <takeoff.json>
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import cv2

cv2.setNumThreads(2)

from prep import load_page, estimate_skew, deskew, normalise
from layout import find_scale_bar, find_legend
from walls import legend_wall_feats, find_walls, mark_exterior
from doors import find_swings, group_pairs, find_special
from fixtures import detect_all, count_fixtures, fixture_mask
from floors import wall_footprint, segment_cells, swatch_descriptors, classify_cells, assign_areas, set_blob_threshold

FLOOR_CODES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']
WALL_CODES = ['EXT', 'P3', 'P2', 'P1']   # legend order: ext, 8" CMU int, 6" mtl stud, 3-5/8" mtl stud

DEBUG = os.environ.get('TAKEOFF_DEBUG')


def log(*a):
    if DEBUG:
        print(*a, file=sys.stderr, flush=True)


def ink_threshold(D):
    """Darkness above which a pixel is considered ink (noise floor of blank paper + margin)."""
    small = cv2.resize(D, (D.shape[1] // 2, D.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Db = cv2.blur(small, (9, 9))
    m = Db < np.percentile(Db, 40)
    vals = small[m]
    if vals.size < 1000:
        return 0.15
    nf = float(np.percentile(vals, 99))
    return float(min(0.3, max(0.12, nf + 0.05)))


def building_roi(D, ppf):
    H, W = D.shape
    B = (D > 0.6).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(B, connectivity=8)
    cands = [i for i in range(1, n) if st[i][2] < 0.8 * W and st[i][0] < 0.7 * W and st[i][3] < 0.9 * H]
    bi = max(cands, key=lambda i: st[i][4])
    x, y, w, h = st[bi][:4]
    m = int(2 * ppf)
    return (max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m))


def empty_result():
    return {
        'floor_sf': {c: 0.0 for c in FLOOR_CODES},
        'wall_lf': {'EXT': 0.0, 'P1': 0.0, 'P2': 0.0, 'P3': 0.0},
        'doors': {'single': 0, 'pair': 0, 'overhead_coiling': 0, 'auto_sliding': 0},
        'fixtures': {'gondola_sections_4ft': 0, 'gondola_sections_3ft': 0, 'endcaps': 0, 'checkouts': 0,
                     'self_checkouts': 0, 'cooler_doors': 0, 'produce_tables': 0},
    }


def _stage(name, fn, *args):
    try:
        return fn(*args)
    except Exception:
        import traceback
        print('takeoff: stage %s failed' % name, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def run(pdf_path, out):
    t0 = time.time()
    gray, dpi = load_page(pdf_path)
    ang = _stage('skew', estimate_skew, gray) or 0.0
    gray = deskew(gray, ang)
    D, paper, ink = normalise(gray)
    del gray
    log('prep %.1fs angle %.3f' % (time.time() - t0, ang))

    sb = _stage('scale', find_scale_bar, D)
    ppf = float(sb['px_per_ft']) if sb else dpi / 8.0
    lg = _stage('legend', find_legend, D) or dict(floor=None, wall=None)
    log('scale ppf %.3f' % ppf)

    roi = building_roi(D, ppf)
    x0, y0, x1, y1 = roi
    sub = np.ascontiguousarray(D[y0:y1, x0:x1])

    # ---------------- walls
    lw = None
    if lg.get('wall'):
        lw = _stage('legend walls', legend_wall_feats, D, lg['wall'])
    if not lw:
        lw = [dict(t=0.65 * ppf, feat=None, peak=1.0)] * 4
    segs, peakref, Dh, Dv = find_walls(D, roi, ppf, lw)
    del Dh, Dv
    mark_exterior(segs, sub.shape, ppf)
    wall_lf = {'EXT': 0.0, 'P1': 0.0, 'P2': 0.0, 'P3': 0.0}
    for q in segs:
        L = (q['eb'] - q['ea']) / ppf
        if q['ext']:
            name = 'EXT'
        else:
            name = WALL_CODES[q['cls']]
            if name == 'EXT':
                name = 'P3'
        q['name'] = name
        wall_lf[name] += L
    out['wall_lf'] = {k: round(wall_lf[k], 2) for k in ('EXT', 'P1', 'P2', 'P3')}
    log('walls %.1fs' % (time.time() - t0), wall_lf)

    # ---------------- floor geometry
    F = wall_footprint(sub.shape, segs)
    bld, interior, Hl, cells, excl = segment_cells(sub, F, ppf, 0.55 * peakref)

    # ---------------- doors
    def do_doors():
        G = cv2.dilate(sub, np.ones((3, 3), np.uint8))
        _, slide0 = find_special(sub, segs, ppf, [], [])
        sw = find_swings(G, segs, ppf, D=sub, exclude=slide0)
        singles, pairs = group_pairs(sw, ppf)
        ohd, slide = find_special(sub, segs, ppf, singles, pairs)
        return {'single': len(singles), 'pair': len(pairs), 'overhead_coiling': len(ohd), 'auto_sliding': len(slide)}
    doors = _stage('doors', do_doors)
    if doors:
        out['doors'] = doors
    log('doors %.1fs' % (time.time() - t0), doors)

    # ---------------- fixtures
    def do_fixtures():
        dets = detect_all(sub, interior, ppf)
        return count_fixtures(dets, ppf)
    r = _stage('fixtures', do_fixtures)
    accepted = {}
    if r:
        out['fixtures'], accepted = r
    log('fixtures %.1fs' % (time.time() - t0), out['fixtures'])

    # ---------------- floor finishes
    def do_floors():
        thr = ink_threshold(D)
        set_blob_threshold(D)
        sw_desc = swatch_descriptors(D, lg['floor'], ppf, thr)
        fm = fixture_mask(sub.shape, accepted, ppf) if accepted else np.zeros(sub.shape, bool)
        lab, labels, info = classify_cells(sub, cells, excl, fm, ppf, sw_desc, thr)
        counts, full = assign_areas(lab, labels, interior, len(FLOOR_CODES))
        for i, inf in sorted(info.items(), key=lambda kv: -kv[1]['area']):
            sc = inf['scores']
            o = np.argsort(sc)
            log('  cell %d %.0f sf -> %s (%.2f) 2nd %s (%.2f)' % (i, inf['area'] / ppf ** 2, FLOOR_CODES[o[0]], sc[o[0]], FLOOR_CODES[o[1]], sc[o[1]]))
        return {c: float(counts[i] / ppf ** 2) for i, c in enumerate(FLOOR_CODES)}, full
    r = _stage('floors', do_floors)
    if r:
        floor_sf, full = r
        out['floor_sf'] = {k: round(v, 1) for k, v in floor_sf.items()}
        log('floors %.1fs' % (time.time() - t0), floor_sf)
        if DEBUG:
            try:
                import pickle
                dbg = dict(roi=roi, ppf=ppf, segs=segs, angle=ang)
                pickle.dump(dict(dbg=dbg, full=full.astype(np.int8)), open(DEBUG, 'wb'))
            except Exception:
                pass
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    out = empty_result()
    try:
        run(sys.argv[1], out)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
    with open(sys.argv[2], 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
