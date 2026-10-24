#!/usr/bin/env python3
"""Quantity takeoff from a scanned A101 fixture & finish plan.

Usage: python3 takeoff.py <scan.pdf> <takeoff.json>
"""
import os
import sys
import json
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

cv2.setNumThreads(2)

from common import prepare, CAL, hollow_dark_fills  # noqa: E402
from scalebar import measure_scale  # noqa: E402
from legend import find_swatches, FLOOR_CODES, WALL_CODES  # noqa: E402
import walls as W  # noqa: E402
import floors as F  # noqa: E402
import doors as D  # noqa: E402
import fixtures as X  # noqa: E402

DEBUG = bool(os.environ.get('TAKEOFF_DEBUG'))


def log(*a):
    if DEBUG:
        print(*a, file=sys.stderr)


def empty_result():
    return {
        'floor_sf': {c: 0.0 for c in FLOOR_CODES},
        'wall_lf': {c: 0.0 for c in ['EXT', 'P1', 'P2', 'P3']},
        'doors': {'single': 0, 'pair': 0, 'overhead_coiling': 0, 'auto_sliding': 0},
        'fixtures': {'gondola_sections_4ft': 0, 'gondola_sections_3ft': 0, 'endcaps': 0,
                     'checkouts': 0, 'self_checkouts': 0, 'cooler_doors': 0, 'produce_tables': 0},
    }


def orient_and_calibrate(dk):
    """Find scale bar and legend; retry with rotated sheet if they are not found."""
    h, w = dk.shape
    tries = [None, cv2.ROTATE_180] if w >= h else [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
    last = None
    for rot in tries:
        img = dk if rot is None else np.ascontiguousarray(cv2.rotate(dk, rot))
        try:
            ppf, sbinfo = measure_scale(img)
            sw = find_swatches(img, ppf)
            if sw['floor'][0][0] < 0.5 * img.shape[1]:
                raise RuntimeError('legend not on the right: sheet upside down?')
            return img, ppf, sbinfo, sw
        except Exception as e:  # noqa: BLE001
            last = e
            log('orientation', rot, 'failed:', e)
    raise last


def run(pdf_path):
    res = empty_result()
    dk, ang = prepare(pdf_path)
    log('skew', ang)
    dk, ppf, sbinfo, sw = orient_and_calibrate(dk)
    log('ppf', ppf, sbinfo.get('ests'))
    # sanity check against the legend swatch width (7.29 ft at plan scale on every example sheet)
    sw_w = float(np.median([f[2] for f in sw['floor']]))
    ppf_leg = sw_w / 7.29
    if abs(ppf_leg / ppf - 1.0) > 0.07:
        log('scale bar estimate %.3f inconsistent with legend (%.3f); using legend' % (ppf, ppf_leg))
        ppf = ppf_leg
        sw = find_swatches(dk, ppf)
    x_limit = int(min(f[0] for f in sw['floor']) - 2 * ppf)
    dkw = hollow_dark_fills(dk)  # for wall/door analysis: solid poche -> outlined band
    lw = W.legend_wall_profiles(dkw, sw['wall'], ppf)
    pk = float(np.median([d['peak'] for d in lw]))
    CAL['k'] = float(np.clip(pk / 0.85, 0.35, 1.2))
    CAL['pk'] = float(np.clip(pk, 0.3, 1.0))
    log('calibration k', CAL['k'])

    # ------------------------------------------------------------------ walls
    walls = []
    try:
        walls, raw = W.detect_walls(dkw, ppf, lw, x_limit)
        W.mark_exterior(walls, dk.shape, ppf)
        W.retype_interior(walls, dkw, ppf, lw)
        W.centreline_lengths(walls, ppf)
        tot = {c: 0.0 for c in ['EXT', 'P1', 'P2', 'P3']}
        for wl in walls:
            k = 'EXT' if wl.get('ext') else wl['type']
            tot[k] += max(0.0, wl['len'])
        res['wall_lf'] = {k: round(v, 2) for k, v in tot.items()}
    except Exception:
        traceback.print_exc()
    log('walls', res['wall_lf'])

    # ------------------------------------------------------------------ floors
    WM = F.wall_mask(dk.shape, walls, ppf)
    IN = F.building_interior(dk.shape, walls, ppf)
    if IN.sum() < 1000 * ppf ** 2:
        # no closed building outline: fall back to the plan area left of the legend
        log('building interior not found; using plan area')
        IN = np.zeros(dk.shape, bool)
        IN[int(0.05 * dk.shape[0]):int(0.9 * dk.shape[0]), int(0.03 * dk.shape[1]):x_limit] = True
    floor = IN & (WM == 0)
    lm = D.line_map(dk)
    fixture_boxes = []
    try:
        res['fixtures'], fixture_boxes = fixture_takeoff(lm, IN, ppf, wallmask=WM)
    except Exception:
        traceback.print_exc()
    log('fixtures', res['fixtures'])
    try:
        res['floor_sf'] = floor_takeoff(dk, lm, floor, sw, ppf, walls, fixture_boxes)
    except Exception:
        traceback.print_exc()
    log('floors', res['floor_sf'])

    # ------------------------------------------------------------------ doors
    try:
        lmw = D.line_map(dkw)
        ops = D.find_openings2(dk, walls, ppf, lm=lm, dk_alt=dkw, lm_alt=lmw)
        lm2 = D.line_map(dk, k=11)
        cnt = {'single': 0, 'pair': 0, 'overhead_coiling': 0, 'auto_sliding': 0}
        classified = []
        for op in ops:
            kind, sc = D.classify_opening(dk, lm, op, ppf, lm2=lm2)
            log('opening', op['wall']['o'], round(op['wall']['c']), op['u0'], op['u1'], kind,
                {k: round(v, 2) for k, v in sc.items()})
            classified.append([op, kind])
        # sliding doors whose panel overlap was mistaken for a jamb: merge adjacent halves
        classified.sort(key=lambda t: (id(t[0]['wall']), t[0]['u0']))
        merged = []
        for item in classified:
            if merged:
                prev = merged[-1]
                same = prev[0]['wall'] is item[0]['wall']
                gap = item[0]['u0'] - prev[0]['u1']
                if same and gap <= 0.5 * ppf and 'auto_sliding' in (prev[1], item[1]):
                    prev[0] = dict(prev[0], u1=item[0]['u1'])
                    prev[1] = 'auto_sliding'
                    continue
            merged.append(item)
        for op, kind in merged:
            if kind in cnt:
                cnt[kind] += 1
        res['doors'] = cnt
    except Exception:
        traceback.print_exc()
    log('doors', res['doors'])

    return res


def floor_takeoff(dk, lm, floor, sw, ppf, walls, fixture_boxes=()):
    from scipy import ndimage
    ys, xs = np.nonzero(floor)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fl = floor[y0:y1, x0:x1]
    sdk = dk[y0:y1, x0:x1]
    slm = lm[y0:y1, x0:x1]
    sm = cv2.GaussianBlur(sdk, (0, 0), 0.6)
    # texture descriptors are computed on a denoised copy (scanner/photocopy noise is white)
    F.set_patch_scale(ppf)
    tdk = cv2.GaussianBlur(dk, (0, 0), 1.0)
    sdesc = F.swatch_descriptors(tdk, sw['floor'], ppf)
    stdk = tdk[y0:y1, x0:x1]
    # exclusion mask for texture sampling: text (heavy components) and detected fixtures
    r = max(2, int(round(0.5 * ppf)))
    hb = (sdk > 0.55 * CAL['k']).astype(np.uint8)
    nh, hl, hst, _ = cv2.connectedComponentsWithStats(hb, connectivity=8)
    small = np.zeros(nh, bool)
    small[1:] = (hst[1:, 4] < 20) & (np.maximum(hst[1:, 2], hst[1:, 3]) < 7)
    hb[small[hl]] = 0
    hv = cv2.dilate(hb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    for (bx, by, bw, bh) in fixture_boxes:
        bx -= x0; by -= y0
        hv[max(0, by - 3):max(0, by + bh + 3), max(0, bx - 3):max(0, bx + bw + 3)] = 1
    tdk_full = cv2.GaussianBlur(dk, (0, 0), 1.0)
    blank_r = F.swatch_blank_radius(lm, sw['floor'], dk=tdk_full)
    log('swatch blank radius', np.round(blank_r, 1))
    sid = F.ink_distance(slm, dk=stdk)
    rng = np.random.default_rng(12345)
    min_seed = 3.0 * ppf ** 2

    def segment(mask, thr):
        bar = cv2.dilate((sm > thr).astype(np.uint8), np.ones((3, 3), np.uint8))
        free = (mask & (bar == 0)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
        return [(lab == i) for i in range(1, n) if st[i, 4] > min_seed]

    def classify(M):
        ok = (M & (hv == 0)).astype(np.uint8)
        ii = cv2.integral(ok)
        votes, used, per = F.classify_zone(stdk, M, ii, sdesc, rng, inkdist=sid, blank_r=blank_r)
        return votes, used, per

    seeds = []  # (mask, class index)
    thrs = [0.62 * CAL['pk'], 0.48 * CAL['pk'], 0.38 * CAL['pk']]

    def process(mask, level):
        for M in segment(mask, thrs[level]):
            votes, used, per = classify(M)
            area = M.sum() / ppf ** 2
            if votes is None:
                log('  ' * level + 'zone area %.1f unclassified' % area)
                continue
            share = votes.max() / max(votes.sum(), 1e-9)
            best = int(np.argmax(votes))
            log('  ' * level + 'zone area %.1f share %.2f' % (area, share),
                {FLOOR_CODES[i]: round(float(v), 2) for i, v in enumerate(votes) if v > 0.01})
            need = 0.92 if area > 1500 else 0.8
            if share < need and area > 40 and level + 1 < len(thrs):
                # mixed textures: try to split along fainter lines
                before = len(seeds)
                process(M, level + 1)
                if len(seeds) > before:
                    continue
            seeds.append((M, best))

    process(fl, 0)
    lab = np.zeros(fl.shape, np.int32)
    cls = [0]
    for i, (M, c) in enumerate(seeds):
        lab[M] = i + 1
        cls.append(c)
    cls = np.array(cls)
    out = {c: 0.0 for c in FLOOR_CODES}
    if not seeds:
        return out
    _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    full = lab[iy, ix]
    full[~fl] = 0
    cnt = np.bincount(full.ravel(), minlength=len(cls))
    for i in range(1, len(cls)):
        out[FLOOR_CODES[cls[i]]] += cnt[i] / ppf ** 2
    return {k: round(v, 1) for k, v in out.items()}


def cooler_rows(dets, ppf):
    """Keep cooler-door detections that belong to a row (neighbour at the door pitch)."""
    keep = []
    for i, d in enumerate(dets):
        sb = d[6]
        horiz = (sb[2] - sb[0]) > (sb[3] - sb[1])  # strip runs along x
        cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
        ok = False
        for j, e in enumerate(dets):
            eb = e[6]
            if i == j or ((eb[2] - eb[0]) > (eb[3] - eb[1])) != horiz:
                continue
            ex, ey = (eb[0] + eb[2]) / 2, (eb[1] + eb[3]) / 2
            along, across = (abs(ex - cx), abs(ey - cy)) if horiz else (abs(ey - cy), abs(ex - cx))
            if 1.9 * ppf <= along <= 3.2 * ppf and across <= 0.3 * ppf:
                ok = True
                break
        if ok:
            keep.append(d)
    return keep


def fixture_takeoff(lm, inside, ppf, wallmask=None):
    ys, xs = np.nonzero(inside)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = X.match_image(lm[y0:y1, x0:x1])
    ins = inside[y0:y1, x0:x1].copy()
    if wallmask is not None:
        wm = wallmask[y0:y1, x0:x1]
        ins &= (wm == 0)
        wii = cv2.integral(wm.astype(np.uint8))
    else:
        wii = None

    def count(name, thr, pad, wall_frac=0.25):
        dets = X.detect(img, name, ppf, min(thr, 0.5), pad_ft=pad)
        good = []
        for d in dets:
            if d[0] < thr:
                continue
            cx, cy = int(d[1] + d[3] / 2), int(d[2] + d[4] / 2)
            if not (0 <= cy < ins.shape[0] and 0 <= cx < ins.shape[1] and ins[cy, cx]):
                continue
            if wii is not None:
                a, b, c_, e = d[2], d[1], d[2] + d[4], d[1] + d[3]
                c_ = min(c_, wii.shape[0] - 1); e = min(e, wii.shape[1] - 1)
                inwall = wii[c_, e] - wii[a, e] - wii[c_, b] + wii[a, b]
                if inwall > wall_frac * d[3] * d[4]:
                    continue
            good.append(d)
        return good, dets

    out = {}
    boxes = []
    boxes_all = []
    for key, nm, thr, pad in (('produce_tables', 'produce', 0.7, 0.25), ('self_checkouts', 'kiosk', 0.7, 0.25),
                              ('checkouts', 'checkout', 0.7, 0.25), ('endcaps', 'endcap', 0.72, 0.1)):
        g = count(nm, thr, pad)[0]
        out[key] = len(g)
        boxes_all += g
    g = count('cooler2', 0.66, 0.08, wall_frac=0.05)[0]
    g = cooler_rows(g, ppf)
    sc = sorted([d[0] for d in g], reverse=True)
    thr = 0.74
    best_gap = 0.0
    for a, b in zip(sc[:-1], sc[1:]):
        if b < 0.66 or a > 0.93:
            continue
        if a - b > best_gap:
            best_gap, cut = a - b, a
    if best_gap >= 0.06:
        thr = max(0.7, min(cut, 0.86))
    g = cooler_rows([d for d in g if d[0] >= thr - 1e-9], ppf)
    log('cooler threshold', round(thr, 3), 'gap', round(best_gap, 3))
    out['cooler_doors'] = len(g)
    boxes_all += g
    # gondola sections: 3' and 4' templates compete; the better fit wins at each location
    c4 = [(d, 4) for d in count('sec4', 0.72, 0.08)[0]]
    c3 = [(d, 3) for d in count('sec3', 0.72, 0.08)[0]]
    allc = sorted(c4 + c3, key=lambda t: -t[0][0])
    acc = []
    for d, L in allc:
        clash = False
        for e, _ in acc:
            ox = min(d[1] + d[3], e[1] + e[3]) - max(d[1], e[1])
            oy = min(d[2] + d[4], e[2] + e[4]) - max(d[2], e[2])
            if ox > 0 and oy > 0 and ox * oy > 0.3 * min(d[3] * d[4], e[3] * e[4]):
                clash = True
                break
        if not clash:
            acc.append((d, L))
    s4 = [d for d, L in acc if L == 4 and d[0] >= 0.82]
    s3 = [d for d, L in acc if L == 3 and d[0] >= 0.82]
    out['gondola_sections_4ft'] = len(s4)
    out['gondola_sections_3ft'] = len(s3)
    for d in boxes_all + s4 + s3:
        boxes.append((int(d[1]) + x0, int(d[2]) + y0, int(d[3]), int(d[4])))
    order = ['gondola_sections_4ft', 'gondola_sections_3ft', 'endcaps', 'checkouts', 'self_checkouts',
             'cooler_doors', 'produce_tables']
    return {k: int(out[k]) for k in order}, boxes


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    pdf, out = sys.argv[1], sys.argv[2]
    try:
        res = run(pdf)
    except Exception:
        traceback.print_exc()
        res = empty_result()
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
