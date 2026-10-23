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

import numpy as np
import cv2

from cvm_takeoff import imgio
from cvm_takeoff import walls as W
from cvm_takeoff import floors as F
from cvm_takeoff import doors as DR
from cvm_takeoff import fixtures as FX

FLOOR_CODES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']
WALL_CODES = ['EXT', 'P1', 'P2', 'P3']
DOOR_KEYS = ['single', 'pair', 'overhead_coiling', 'auto_sliding']
FIXTURE_KEYS = ['gondola_sections_4ft', 'gondola_sections_3ft', 'endcaps', 'checkouts', 'self_checkouts',
                'cooler_doors', 'produce_tables']

# the detectors are tuned for sheets scanned so that one foot spans about this many pixels
CANON_PPF = 24.5


def log(ctx, *a):
    if ctx.get('verbose'):
        print('[%6.1fs]' % (time.time() - ctx['t0']), *a, file=sys.stderr)


def paper_map(d, bg):
    """Smooth map of the paper tone (absorbs uneven illumination of the scan).

    Local maxima of a coarse copy follow the paper between the lines; the offset between
    that envelope and the global paper level is removed so a uniform sheet maps to `bg`.
    """
    f = 4
    small = cv2.resize(d, (d.shape[1] // f, d.shape[0] // f), interpolation=cv2.INTER_AREA).astype(np.float32)
    env = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    env = cv2.GaussianBlur(env, (0, 0), 10)
    env -= float(np.median(env)) - bg
    return cv2.resize(env, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)


def prepare(pdf_path, ctx):
    img = imgio.load_sheet(pdf_path)
    ang = imgio.estimate_skew(img)
    d = imgio.deskew(img, ang)
    del img

    def scale_of(img):
        try:
            return imgio.find_scale(img)
        except Exception:
            # graphic scale unreadable: fall back on the size of the sheet border
            p = imgio.scale_from_border(img)
            if p is None:
                raise
            return p, dict(method='border')

    ppf, sinfo = scale_of(d)
    if not (19.5 <= ppf <= 31.0):
        # unusual scan resolution / print reduction: resample to the canonical scale
        s = CANON_PPF / ppf
        d = cv2.resize(d, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        ppf, sinfo = scale_of(d)
    bg = float(np.percentile(d, 75))
    bgmap = paper_map(d, bg)
    # solid ink level: the black scale bar segments (fallback: darkest scan tones)
    ink = float(np.percentile(d[::2, ::2], 0.05))
    try:
        x, y, h, seg = sinfo['zero_x'], sinfo['y'], sinfo['h'], sinfo['seg']
        patch = d[y + h // 4:y + h - h // 4, x + seg // 5:x + seg - seg // 5]
        if patch.size:
            ink = float(np.median(patch))
    except Exception:
        pass
    gain = 225.0 / max(30.0, bg - ink)
    D = np.clip((bgmap - d.astype(np.float32)) * gain, 0, None)
    del bgmap
    ctx.update(gray=d, D=D, ppf=ppf, angle=ang, scale_info=sinfo, bg=bg, ink=ink, gain=gain)
    log(ctx, 'angle %.4f ppf %.4f bg %.1f ink %.1f' % (ang, ppf, bg, ink))


def do_walls(ctx):
    D, ppf = ctx['D'], ctx['ppf']
    cands, heavy_segs = W.detect_walls(D, ppf)
    runs = W.merge_runs(cands, ppf, heavy_segs)
    runs = [r for r in runs if r['b'] - r['a'] >= 1.0 * ppf]
    W.snap_runs(runs, ppf)
    idxs = W.main_structure(runs, ppf)
    ext = W.classify_exterior(D.shape, runs, idxs, ppf)
    idxs = W.add_free_standing(runs, idxs, ext, ppf)
    tot = W.wall_lengths(runs, idxs, ppf)
    ctx.update(cands=cands, heavy_segs=heavy_segs, runs=runs, widx=idxs, exterior=ext, wall_lf=tot)
    log(ctx, 'walls', {k: round(v, 2) for k, v in tot.items()})


def do_floors(ctx):
    D, ppf, ext = ctx['D'], ctx['ppf'], ctx['exterior']
    runs, idxs = ctx['runs'], ctx['widx']
    wm = W.render_walls(D.shape, runs, idxs)
    ys, xs = np.nonzero(~ext)
    pad = 40
    y0 = max(0, ys.min() - pad); y1 = min(D.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad); x1 = min(D.shape[1], xs.max() + 1 + pad)
    Dc = D[y0:y1, x0:x1]
    extc = ext[y0:y1, x0:x1]
    wmc = wm[y0:y1, x0:x1]
    dog = F.dog_map(Dc, wmc)
    ref = F.calibrate_heavy(dog, runs, idxs, (x0, y0), ppf)
    heavy_thr = F.HEAVY_DOG if ref is None else float(np.clip(0.55 * ref, 30.0, 60.0))
    # straight heavy lines (transitions) are drawn into the barrier explicitly, extended so
    # that they close against walls even where the line response fades near junctions
    segs_c = {}
    for axis, lst in ctx['heavy_segs'].items():
        cl = []
        for s in lst:
            if s['b'] - s['a'] < 1.0 * ppf:
                continue
            s2 = dict(s)
            if axis == 0:
                s2['pos'] = s['pos'] - y0; s2['a'] = s['a'] - x0; s2['b'] = s['b'] - x0
            else:
                s2['pos'] = s['pos'] - x0; s2['a'] = s['a'] - y0; s2['b'] = s['b'] - y0
            cl.append(s2)
        segs_c[axis] = cl
    extra = F.render_segments(Dc.shape, segs_c)
    lab, interior = F.segment(Dc, dog, wmc, extc, ppf, heavy_thr=heavy_thr, extra_barrier=extra)
    heavy = ((dog > heavy_thr) | (extra > 0)).astype(np.uint8)
    del dog, extra
    valid = interior & ~(cv2.dilate(heavy, np.ones((9, 9), np.uint8)) > 0)
    del heavy
    maps, sw, support = F.local_amplitudes(Dc, ppf, valid)
    amps = F.region_amplitudes(lab, maps, sw, valid, support)
    specks = F.speck_density(Dc, lab, ppf, valid)
    lab, remap = F.merge_islands(lab, wmc, amps, ppf, specks=specks)
    amps = F.region_amplitudes(lab, maps, sw, valid, support)
    del maps
    specks = F.speck_density(Dc, lab, ppf, valid)
    areas = np.bincount(lab.ravel(), minlength=int(lab.max()) + 1) / (ppf * ppf)
    regions = []
    tot = {c: 0.0 for c in FLOOR_CODES}
    for r in range(1, int(lab.max()) + 1):
        if amps[r] is None:
            continue
        code, scores = F.classify_amplitudes(amps[r], specks[r])
        tot[code] = tot.get(code, 0.0) + float(areas[r])
        regions.append(dict(id=r, code=code, area=float(areas[r]), scores=scores, specks=float(specks[r])))
    ctx.update(floor_sf=tot, regions=regions, floor_lab=lab, floor_off=(x0, y0), wall_mask=wm, heavy_thr=heavy_thr)
    log(ctx, 'floors', {k: round(v, 1) for k, v in tot.items()})


def building_crop(ctx, pad_ft=1.0):
    D, ext, ppf = ctx['D'], ctx['exterior'], ctx['ppf']
    ys, xs = np.nonzero(~ext)
    pad = int(pad_ft * ppf)
    y0 = max(0, ys.min() - pad); y1 = min(D.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad); x1 = min(D.shape[1], xs.max() + 1 + pad)
    return x0, y0, x1, y1


def do_line_scale(ctx):
    """Response of thin lines on this scan relative to the reference scans (blur/scale)."""
    x0, y0, x1, y1 = building_crop(ctx)
    ev = FX.LineEvidence(ctx['D'][y0:y1, x0:x1])
    ctx['line_k'] = FX.line_scale(ev, ctx['runs'], ctx['widx'], (x0, y0))
    ctx['fx_ev'] = (ev, (x0, y0, x1, y1))
    log(ctx, 'line scale %.3f' % ctx['line_k'])


def do_doors(ctx):
    counts, details = DR.classify_doors(ctx['D'], ctx['runs'], ctx['widx'], ctx['ppf'], k=ctx.get('line_k', 1.0))
    ctx.update(doors=counts, door_details=details)
    log(ctx, 'doors', counts)


def do_fixtures(ctx):
    D, ppf, ext = ctx['D'], ctx['ppf'], ctx['exterior']
    if 'fx_ev' in ctx:
        ev, (x0, y0, x1, y1) = ctx.pop('fx_ev')
    else:
        x0, y0, x1, y1 = building_crop(ctx)
        ev = FX.LineEvidence(D[y0:y1, x0:x1])
    ev.k = ctx.get('line_k', 1.0)
    Dc = D[y0:y1, x0:x1]
    inside = ~ext[y0:y1, x0:x1]
    segs = FX.fixture_segments(Dc, ppf, ev)

    def ins(x, y):
        xi, yi = int(round(x)), int(round(y))
        return 0 <= yi < inside.shape[0] and 0 <= xi < inside.shape[1] and bool(inside[yi, xi])

    runs = [r for r in FX.find_gondolas(Dc, ppf, segs, ev)
            if ins(*FX.to_xy(r['axis'], 0.5 * (r['a'] + r['b']), 0.5 * (r['c1'] + r['c2'])))]
    n4, n3, nend = FX.count_gondolas(runs, ppf)
    strips = [s for s in FX.find_cooler_doors(Dc, ppf, segs, ev)
              if ins(*FX.to_xy(s['axis'], 0.5 * (s['a'] + s['b']), 0.5 * (s['c1'] + s['c2'])))]
    ncool = sum(len(s['doors']) for s in strips)
    pts = [c for c in FX.find_produce_tables(ev, segs, ppf) if ins(c['cx'], c['cy'])]
    scos = [c for c in FX.find_self_checkouts(ev, segs, ppf) if ins(c['cx'], c['cy'])]
    cos = [c for c in FX.find_checkouts(ev, segs, ppf) if ins(c['cx'], c['cy'])]
    fx = dict(gondola_sections_4ft=n4, gondola_sections_3ft=n3, endcaps=nend, checkouts=len(cos),
              self_checkouts=len(scos), cooler_doors=ncool, produce_tables=len(pts))
    ctx.update(fixtures=fx, fixture_detail=dict(runs=runs, strips=strips, produce=pts, sco=scos, checkouts=cos,
                                                 offset=(x0, y0)))
    log(ctx, 'fixtures', fx, 'k %.2f' % ev.k)


def run(pdf_path, verbose=False):
    ctx = dict(t0=time.time(), verbose=verbose, errors=[])
    # later stages need the sheet and the walls; the others are independent of each other
    for stage, essential in ((prepare, True), (do_walls, True), (do_line_scale, False),
                             (do_floors, False), (do_doors, False), (do_fixtures, False)):
        try:
            if stage is prepare:
                stage(pdf_path, ctx)
            else:
                stage(ctx)
        except Exception:
            ctx['errors'].append(traceback.format_exc())
            print('takeoff: stage %s failed:\n%s' % (stage.__name__, ctx['errors'][-1]), file=sys.stderr)
            if stage is do_line_scale:
                ctx['line_k'] = 1.0
            if essential:
                break
    out = dict(
        floor_sf={c: round(float(ctx.get('floor_sf', {}).get(c, 0.0)), 1) for c in FLOOR_CODES},
        wall_lf={c: round(float(ctx.get('wall_lf', {}).get(c, 0.0)), 2) for c in WALL_CODES},
        doors={k: int(ctx.get('doors', {}).get(k, 0)) for k in DOOR_KEYS},
        fixtures={k: int(ctx.get('fixtures', {}).get(k, 0)) for k in FIXTURE_KEYS},
    )
    ctx['result'] = out
    return out, ctx


def main(argv):
    if len(argv) != 3:
        print('usage: takeoff.py <scan.pdf> <takeoff.json>', file=sys.stderr)
        return 2
    out, ctx = run(argv[1], verbose=bool(os.environ.get('TAKEOFF_VERBOSE')))
    outdir = os.path.dirname(os.path.abspath(argv[2]))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    with open(argv[2], 'w') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
