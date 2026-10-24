#!/usr/bin/env python3
"""Quantity takeoff from a scanned A101 fixture & finish plan.

usage: python3 takeoff.py <scan.pdf> <takeoff.json>

Pipeline (all measurements in feet, via the sheet's own graphic scale):
  core      load the scanned page, flatten paper/contrast, deskew, measure the
            graphic scale bar, find the exterior wall rectangle.
  legend    locate the 8 floor-finish swatches and 4 partition swatches (the
            chain's standard order: LVT-1 PC-1 SC-1 EP-1 VCT-1 CT-1 CPT-1 WOM-1;
            EXT P3 P2 P1).
  walls     pair parallel wall-face lines, type each wall by matching its poche
            against the legend swatches (P1 is also recognisable by thickness),
            snap centrelines to the whole-foot grid (also used to calibrate the
            building size), extend ends to the centreline of the wall met.
  fixtures  normalised template matching of the chain's standard blocks
            (gondola sections, endcaps, checkouts, self-checkouts, cooler doors
            with their case, produce tables) on the sales floor.
  floors    room polygons from the wall model (faces of walls, openings
            excluded); hatch identified by matching texture signatures
            (autocorrelation, ink density, long-line content) against the
            legend swatches; rooms with several finishes are split by a marker
            watershed along the finish transition lines.
  doors     swing doors from leaf+arc templates (mirrored pairs -> pair doors);
            wall openings from poche discontinuities; sliding doors from offset
            panels / outward arrows; remaining wide exterior openings are
            overhead coiling doors.
Set TAKEOFF_DEBUG=1 to log progress to stderr.
"""
import os
import sys
import json
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import cv2

cv2.setNumThreads(2)

import core
import legend as legmod
import walls as wallmod
import floor as floormod
import fixtures as fixmod
import doors as doormod

FLOOR_CODES = legmod.FLOOR_CODES
WALL_CODES = ['EXT', 'P1', 'P2', 'P3']


def empty_result():
    return {
        'floor_sf': {c: 0.0 for c in FLOOR_CODES},
        'wall_lf': {c: 0.0 for c in WALL_CODES},
        'doors': {'single': 0, 'pair': 0, 'overhead_coiling': 0, 'auto_sliding': 0},
        'fixtures': {'gondola_sections_4ft': 0, 'gondola_sections_3ft': 0, 'endcaps': 0,
                     'checkouts': 0, 'self_checkouts': 0, 'cooler_doors': 0, 'produce_tables': 0},
    }


class Ctx(object):
    pass


def floor_stage(c, res, rot, ppf, bldg, runs, ref, W, H, sx, sy, S, sref, leg, masked_boxes, note):
    ox0, oy0, ox1, oy1 = bldg['outer']
    Rh, Rv = wallmod.ridge_maps(rot)
    gxs, gys = floormod.detect_grid_lines(Rh, Rv, bldg, ppf, ref)
    rooms = floormod.room_polygons(runs, W, H)
    tf = lambda x, y: (ox0 + x * sx, oy0 + y * sy)
    lab = floormod.rasterize_rooms(rooms, rot.shape, tf)
    # strong-line threshold: above the strongest hatch drawn in the legend
    hp = []
    for (x0_, y0_, x1_, y1_) in leg['floor']:
        sub = S[int(round(y0_)) + 5:int(round(y1_)) - 4, int(round(x0_)) + 5:int(round(x1_)) - 4]
        if sub.size:
            hp.append(float(np.percentile(sub, 99)) / sref)
    thr = 0.55
    if hp:
        thr = min(max(0.55, 1.15 * max(hp)), 0.7)
    c.strong_thr = thr
    B = floormod.strong_line_mask(S, sref, thr)
    G = np.zeros_like(B)
    for x in gxs:
        G[:, max(int(round(x)) - 3, 0):int(round(x)) + 4] = True
    for y in gys:
        G[max(int(round(y)) - 3, 0):int(round(y)) + 4, :] = True
    # solid black blobs (columns, bold text) carry no hatch information
    solid = ((1.0 - rot) > 0.8).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    G |= cv2.dilate(solid, np.ones((7, 7), np.uint8)).astype(bool)
    del solid
    pad = int(0.15 * ppf)
    for d in masked_boxes:
        x0_ = max(int(d['x']) - pad, 0); y0_ = max(int(d['y']) - pad, 0)
        x1_ = int(d['x'] + d['w']) + pad; y1_ = int(d['y'] + d['h']) + pad
        G[y0_:y1_, x0_:x1_] = True
    Btex = cv2.dilate((B | G).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    Bseg = cv2.dilate(B.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    del G
    tink = floormod.texture_ink(rot)
    sw = floormod.swatch_signatures(tink, leg['floor'], Btex, lines=(Rh, Rv))
    note('floor prep, %d rooms' % len(rooms))
    c.floor_debug = [] if getattr(c, 'want_floor_debug', False) else None
    fres = floormod.finish_labels(tink, ppf, rooms, lab, sw, S, sref, Bseg, Btex, lines=(Rh, Rv),
                                  debug=c.floor_debug)
    del tink, Rh, Rv
    for poly, rr in zip(rooms, fres):
        for k, fr in rr['fractions'].items():
            res['floor_sf'][FLOOR_CODES[k]] += poly.area * fr
    note('floors done')
    c.floor_results = (rooms, fres)


def analyse(pdf_path, log=None, stages=('fixtures', 'doors'), floor_debug=False):
    t0 = time.time()
    def note(msg):
        if log is not None:
            log.write('[%6.1fs] %s\n' % (time.time() - t0, msg))
            log.flush()
    c = Ctx()
    c.want_floor_debug = floor_debug
    gray = core.load_scan(pdf_path)
    norm = core.normalise(gray)
    rot, angle = core.deskew(norm)
    c.rot = rot
    note('deskew %.4f deg' % angle)
    h, w = rot.shape
    ppf = None
    try:
        sb = core.find_scale_bar(rot)
        ppf = sb['ppf']
        if not (8.0 < ppf < 60.0):
            ppf = None
    except Exception:
        ppf = None
    if ppf is None:
        # fallback: legend swatches are 7.2 ft wide at plan scale
        nominal = h / 24.0 / 8.0  # 1/8" = 1'-0" on a 24" tall sheet
        lg = legmod.find_legend(rot, nominal, int(w * 0.5))
        ws = [b[2] - b[0] for b in lg['floor']]
        ppf = float(np.median(ws)) / 7.2
        note('scale bar not found; legend-based ppf')
    c.ppf = ppf
    note('scale bar ppf %.4f' % ppf)
    bldg = core.find_building(rot, ppf, xmax=int(w * 0.72))
    ox0, oy0, ox1, oy1 = bldg['outer']
    Wf = (ox1 - ox0) / ppf
    Hf = (oy1 - oy0) / ppf
    W = float(round(Wf)) if abs(Wf - round(Wf)) < 0.3 else round(Wf * 2) / 2.0
    H = float(round(Hf)) if abs(Hf - round(Hf)) < 0.3 else round(Hf * 2) / 2.0
    c.W, c.H, c.bldg = W, H, bldg
    sx = (ox1 - ox0) / W
    sy = (oy1 - oy0) / H
    c.sx, c.sy = sx, sy
    note('building %.2f x %.2f ft -> %g x %g' % (Wf, Hf, W, H))
    res = empty_result()
    text = 2.0 / 3.0
    res['wall_lf']['EXT'] = 2 * (W + H) - 4 * text
    leg = None
    try:
        leg = legmod.find_legend(rot, ppf, int(max(ox1 + 2 * ppf, w * 0.5)))
        note('legend found')
    except Exception:
        traceback.print_exc()
    c.legend = leg
    runs, ref = [], None
    if leg is not None:
        try:
            runs, ref, (W2, H2) = wallmod.detect_walls(rot, ppf, bldg, leg['wall'], W, H)
            if (W2, H2) != (W, H):
                note('calibrated building %g x %g -> %g x %g' % (W, H, W2, H2))
                W, H = W2, H2
                c.W, c.H = W, H
                sx = (ox1 - ox0) / W
                sy = (oy1 - oy0) / H
                c.sx, c.sy = sx, sy
                res['wall_lf']['EXT'] = 2 * (W + H) - 4 * text
        except Exception:
            traceback.print_exc()
            runs = []
    if ref is None:
        Rh, Rv = wallmod.ridge_maps(rot)
        ref = wallmod.face_reference(Rh, Rv, bldg)
        del Rh, Rv
    c.runs, c.ref = runs, ref
    note('walls: %d runs' % len(runs))
    for r in runs:
        res['wall_lf'][r['type']] += r['hi'] - r['lo']
    # line strength (used by floors, fixtures, doors)
    S = floormod.line_strength(rot)
    sref = floormod.strength_reference(S, bldg)
    c.S, c.sref = S, sref
    k3 = np.ones((3, 3), np.uint8)
    # fixtures (their footprints are excluded from hatch analysis)
    fixture_boxes = []
    if 'fixtures' in stages:
        region = (int(ox0) + 10, int(oy0) + 10, int(ox1) - 10, int(oy1) - 10)
        try:
            # store fixtures stand on the sales floor (the largest room)
            floor_mask = None
            try:
                rooms0 = floormod.room_polygons(runs, W, H)
                if rooms0:
                    big = max(rooms0, key=lambda p: p.area)
                    if big.area > 0.3 * sum(p.area for p in rooms0):
                        tf0 = lambda x, y: (ox0 + x * sx, oy0 + y * sy)
                        fm = floormod.rasterize_rooms([big.buffer(1.0)], rot.shape, tf0) > 0
                        floor_mask = fm
            except Exception:
                floor_mask = None
            fdbg = {}
            cnt = fixmod.detect_fixtures(rot, S, sref, region, sx, sy, floor_mask=floor_mask, debug=fdbg)
            for k, v in cnt.items():
                res['fixtures'][k] = v
            for k, dets in fdbg.items():
                fixture_boxes.extend(dets)
        except Exception:
            traceback.print_exc()
        note('fixtures %s' % res['fixtures'])
    # swing doors (leaf + arc areas are excluded from hatch analysis too)
    swings = []
    try:
        inkb = cv2.GaussianBlur((1.0 - rot).astype(np.float32), (0, 0), 0.7)
        dregion = (max(int(ox0 - 4 * sx), 0), max(int(oy0 - 4 * sy), 0),
                   min(int(ox1 + 4 * sx), w), min(int(oy1 + 4 * sy), h))
        swings = doormod.detect_swings(inkb, dregion, sx, sy)
        del inkb
    except Exception:
        traceback.print_exc()
    note('swings %d' % len(swings))
    c.swings = swings
    # floors
    try:
        if leg is None:
            raise RuntimeError('no legend: floor finishes cannot be identified')
        floor_stage(c, res, rot, ppf, bldg, runs, ref, W, H, sx, sy, S, sref, leg,
                    fixture_boxes + swings, note)
    except Exception:
        traceback.print_exc()
    if 'doors' not in stages:
        return res, c
    # doors
    try:
        if leg is None:
            singles, pairs = doormod.group_swings(swings, sx, sy)
            res['doors']['single'] = len(singles)
            res['doors']['pair'] = len(pairs)
            raise RuntimeError('no legend: only swing doors counted')
        lf = wallmod.legend_wall_features(rot, leg['wall'])
        dbg = {}
        dc = doormod.count_doors(rot, S, sref, ref, runs, W, H, bldg, sx, sy, lf, debug=dbg, swings=swings)
        res['doors'].update(dc)
        c.openings = dbg.get('openings', [])
        c.swings = dbg.get('swings', [])
        c.arrow_pairs = dbg.get('arrow_pairs', [])
    except Exception:
        traceback.print_exc()
    note('doors %s' % res['doors'])
    return res, c


def finalise(res):
    out = empty_result()
    for k in out['floor_sf']:
        out['floor_sf'][k] = round(float(res['floor_sf'].get(k, 0.0)), 1)
    for k in out['wall_lf']:
        out['wall_lf'][k] = round(float(res['wall_lf'].get(k, 0.0)), 2)
    for grp in ('doors', 'fixtures'):
        for k in out[grp]:
            out[grp][k] = int(res[grp].get(k, 0))
    return out


def main(argv):
    if len(argv) != 3:
        sys.stderr.write('usage: takeoff.py <scan.pdf> <takeoff.json>\n')
        return 2
    pdf, outp = argv[1], argv[2]
    log = sys.stderr if os.environ.get('TAKEOFF_DEBUG') else None
    try:
        res, _ = analyse(pdf, log)
    except Exception:
        traceback.print_exc()
        res = empty_result()
    out = finalise(res)
    with open(outp, 'w') as fh:
        json.dump(out, fh, indent=2)
        fh.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
