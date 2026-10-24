#!/usr/bin/env python3
"""Quantity takeoff from a scanned A101 fixture & finish plan.

usage: python3 takeoff.py <scan.pdf> <takeoff.json>
"""
import os
import sys
import json
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import prep      # noqa: E402
import layout    # noqa: E402
import walls     # noqa: E402
import floors    # noqa: E402
import doors     # noqa: E402
import fixtures  # noqa: E402

DEBUG = bool(os.environ.get('TAKEOFF_DEBUG'))
CANON_PPF = 24.5
REF_OUTLINE = 0.875
if os.environ.get('TAKEOFF_THREADS'):
    cv2.setNumThreads(int(os.environ['TAKEOFF_THREADS']))


def log(*a):
    if DEBUG:
        print(*a, file=sys.stderr, flush=True)


def prepare(pdf_path):
    """Load, deskew, normalise, measure scale, resample, find legend."""
    t0 = time.time()
    gray, colour = prep.load_scan(pdf_path)
    k = prep.ink_map(gray)
    del gray, colour
    ang = prep.estimate_skew(k)
    if abs(ang) > 1e-4:
        k = prep.rotate(k, ang)
    log('skew %.4f deg  (%.1fs)' % (ang, time.time() - t0))
    # contrast normalisation: provisional, then from the scale bar's solid fill
    L0 = float(np.percentile(k, 99.7))
    sb = prep.find_scale_bar(k / max(L0, 1e-3))
    if sb is not None:
        x, y, w, h = sb['x'], sb['y'], sb['w'], sb['h']
        seg = k[y + h // 3:y + 2 * h // 3 + 1, x + int(0.02 * w):x + int(0.1 * w)]
        L = float(np.median(seg)) if seg.size else L0
    else:
        L = L0
    kn = k / max(L, 1e-3)
    del k
    sb = prep.find_scale_bar(kn) or sb
    if sb is None:
        # fallback: nominal 1/8" scale at the scan resolution, refined below
        # from the legend swatch width (7.3 ft in every office's legend)
        ppf = 25.0 * kn.shape[1] / 7200.0
        lg0 = layout.find_legend(kn, ppf)
        if lg0 is not None:
            ppf = float(np.median([q[2] for q in lg0['swatches']])) / 7.3
        sb = dict(x=0, y=0, w=0, h=0, ppf=ppf)
        log('scale bar not found; fallback ppf %.3f' % ppf)
    ppf = sb['ppf']
    log('scale %.4f px/ft  (%.1fs)' % (ppf, time.time() - t0))
    # resample to the canonical working resolution
    f = CANON_PPF / ppf
    if abs(f - 1.0) > 0.03:
        interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
        kn = cv2.resize(kn, None, fx=f, fy=f, interpolation=interp)
        np.clip(kn, 0, 2.0, out=kn)
        ppf *= f
        log('resampled x%.3f -> %.3f px/ft' % (f, ppf))
    lg = layout.find_legend(kn, ppf)
    if lg is None:
        raise RuntimeError('legend not found')
    # calibrate stroke darkness: heavy outlines of the legend bars -> REF_OUTLINE
    R = layout.bar_outline_peak(kn, lg['bars'])
    if R is not None and 0.2 < R < 2.0:
        kn *= REF_OUTLINE / R
        log('outline calibration R=%.3f' % R)
    return dict(kn=kn, ppf=ppf, lg=lg, sb=sb, angle=ang, t0=t0)


def empty_result():
    return {
        'floor_sf': {c: 0.0 for c in layout.FINISH_CODES},
        'wall_lf': {c: 0.0 for c in ('EXT', 'P1', 'P2', 'P3')},
        'doors': {c: 0 for c in ('single', 'pair', 'overhead_coiling', 'auto_sliding')},
        'fixtures': {c: 0 for c in ('gondola_sections_4ft', 'gondola_sections_3ft', 'endcaps',
                                    'checkouts', 'self_checkouts', 'cooler_doors', 'produce_tables')},
    }


def run(pdf_path):
    out = empty_result()
    P = prepare(pdf_path)
    kn, ppf, lg, t0 = P['kn'], P['ppf'], P['lg'], P['t0']
    hb, _ = walls.detect_bands_multi(kn, ppf, horizontal=True)
    vb, _ = walls.detect_bands_multi(kn, ppf, horizontal=False)
    wres = walls.assemble(kn, ppf, lg, hb, vb)
    try:
        nb = doors.bridge_gaps(kn, ppf, wres)
        if nb:
            log('bridged %d door gaps' % nb)
    except Exception as e:  # pragma: no cover - defensive
        log('bridge failed', repr(e))
    log('walls', {k_: round(v, 2) for k_, v in wres['lengths'].items()}, '(%.1fs)' % (time.time() - t0))
    out['wall_lf'] = {c: round(float(wres['lengths'].get(c, 0.0)), 2) for c in ('EXT', 'P1', 'P2', 'P3')}
    try:
        fl = floors.floor_areas(kn, ppf, lg, wres)
        log('floors', {k_: round(v, 1) for k_, v in fl.items()}, '(%.1fs)' % (time.time() - t0))
        out['floor_sf'] = {c: round(float(fl.get(c, 0.0)), 1) for c in layout.FINISH_CODES}
    except Exception as e:
        log('floors failed', repr(e))
    try:
        dcounts, _ = doors.detect_doors(kn, ppf, wres)
        log('doors', dcounts, '(%.1fs)' % (time.time() - t0))
        out['doors'] = {c: int(dcounts.get(c, 0)) for c in ('single', 'pair', 'overhead_coiling', 'auto_sliding')}
    except Exception as e:
        log('doors failed', repr(e))
    try:
        top, bot, lef, rig = wres['ext_box']
        region = (int(lef), int(top), int(rig), int(bot))
        dets = fixtures.match_blocks(kn, ppf, region, thr=0.5)
        fcounts, _ = fixtures.joint_nms(dets, ppf=ppf)
        log('fixtures', fcounts, '(%.1fs)' % (time.time() - t0))
        out['fixtures'] = {c: int(fcounts.get(c, 0)) for c in out['fixtures']}
    except Exception as e:
        log('fixtures failed', repr(e))
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    try:
        out = run(sys.argv[1])
    except Exception as e:
        import traceback
        traceback.print_exc()
        out = empty_result()
    with open(sys.argv[2], 'w') as f:
        json.dump(out, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
