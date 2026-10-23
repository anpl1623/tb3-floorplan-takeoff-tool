"""End-to-end quantity takeoff from a scanned A101 fixture & finish plan."""
import time
import numpy as np
import cv2

from . import image, walls as W, doors as D, floors as F, fixtures as X, legend as LG

FINISH_CODES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']
WALL_TAGS = ['EXT', 'P1', 'P2', 'P3']
# wall appearance class -> partition tag (per the sheet legend)
WALL_MAP = {'ext': 'EXT', 'hatch': 'P3', 'gray': 'P2', 'light': 'P1'}
REF_PPF = 24.5


def detect_walls(G, s, bg):
    ps = W.detect_pieces(G, s, bg, 400)
    for p in ps:
        p['cls'] = W.piece_class(p, s)
    ps = [p for p in ps if p['cls']]
    ws = W.merge_pieces(ps, s)
    ws = W.bridge_gaps(ws, G, bg, s)
    ws = [w for w in ws if (w['a1'] - w['a0']) >= 1.0 * s]
    ws = W.remove_nested(ws, s)
    all_ws = list(ws)
    fp = W.footprint(ws, s, G.shape)
    if fp is not None:
        def mid_inside(w):
            if w['o'] == 'H':
                return W.inside(fp, (w['a0'] + w['a1']) / 2.0, w['c'])
            return W.inside(fp, w['c'], (w['a0'] + w['a1']) / 2.0)
        ws = [w for w in ws if mid_inside(w)]
    W.refine_faces(G, ws, s)
    ws = W.drop_implausible(ws, s)
    if fp is not None:
        W.classify_exterior(ws, fp, s)
    else:
        for w in ws:
            w['ext'] = False
    W.snap_ends(ws, s)
    if W.extend_free_ends(ws, G, bg, s):
        W.snap_ends(ws, s)
    return ws, fp, all_ws


def wall_lengths(ws, s, wall_map=None):
    wall_map = wall_map or WALL_MAP
    out = {k: 0.0 for k in WALL_TAGS}
    for w in ws:
        L = abs(w['e1'] - w['e0']) / s
        tag = wall_map['ext'] if w.get('ext') else wall_map[w['cls']]
        out[tag] = out.get(tag, 0.0) + L
    return out


def count_doors(G, s, bg, ws):
    cnt = {'single': 0, 'pair': 0, 'overhead_coiling': 0, 'auto_sliding': 0}
    found = []
    nonswing = ('auto_sliding', 'overhead_coiling', 'unknown')
    typical = W.typical_fills(G, ws)
    for w in ws:
        ops = W.find_openings(G, w, s, bg, typical=typical)
        merged = []
        for o in ops:
            if merged and o[0] - merged[-1][1] < 0.6 * s:
                merged[-1] = (merged[-1][0], o[1])
            else:
                merged.append(o)
        items = []
        for (g0, g1) in merged:
            typ, feats = D.classify_opening(G, bg, w, g0, g1, s)
            items.append([g0, g1, typ, feats])
        # fragments of one sliding / overhead door split by internal panels
        changed = True
        while changed:
            changed = False
            for i in range(len(items) - 1):
                a, b = items[i], items[i + 1]
                if a[2] in nonswing and b[2] in nonswing and b[0] - a[1] < 1.5 * s:
                    g0, g1 = a[0], b[1]
                    typ, feats = D.classify_opening(G, bg, w, g0, g1, s)
                    if typ not in nonswing and 'unknown' not in (a[2], b[2]):
                        continue
                    items[i] = [g0, g1, typ, feats]
                    del items[i + 1]
                    changed = True
                    break
        for g0, g1, typ, feats in items:
            found.append(dict(wall=w, g0=g0, g1=g1, type=typ, feats=feats))
            if typ in cnt:
                cnt[typ] += 1
    return cnt, found


def count_fixtures(G, s, bg, ws):
    xs = [w['a0'] for w in ws if w['o'] == 'H'] + [w['a1'] for w in ws if w['o'] == 'H']
    ys = [w['a0'] for w in ws if w['o'] == 'V'] + [w['a1'] for w in ws if w['o'] == 'V']
    H, Wd = G.shape
    if xs and ys:
        x0 = max(0, int(min(xs)) - 20); x1 = min(Wd, int(max(xs)) + 20)
        y0 = max(0, int(min(ys)) - 20); y1 = min(H, int(max(ys)) + 20)
    else:
        x0, y0, x1, y1 = 0, 0, Wd, H
    sub = np.ascontiguousarray(G[y0:y1, x0:x1])
    f = X.C / s
    Gc = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    sym, dets = X.count_symbols(Gc, bg)
    n4 = n3 = ne = 0
    for tr in (False, True):
        img = np.ascontiguousarray(Gc.T) if tr else Gc
        Dk = np.clip(bg - img, 0, None)
        runs = X.gondola_runs(Gc, bg, tr)
        for r in runs:
            X.classify_segments(r, Dk)
        for r in X.resolve_runs(runs):
            n4 += r['n4']; n3 += r['n3']; ne += r['ne']
    return {
        'gondola_sections_4ft': int(n4),
        'gondola_sections_3ft': int(n3),
        'endcaps': int(ne),
        'checkouts': int(sym.get('checkout', 0)),
        'self_checkouts': int(sym.get('sco', 0)),
        'cooler_doors': int(sym.get('cooler', 0)),
        'produce_tables': int(sym.get('produce', 0)),
    }


def _guard(name, fn, default, errors):
    try:
        return fn()
    except Exception as e:  # keep going: a partial takeoff beats none
        import traceback
        errors.append('%s: %s' % (name, traceback.format_exc()))
        return default


def run(pdf_path, debug=False):
    t0 = time.time()
    errors = []
    g = image.load_pdf_gray(pdf_path)
    if g.shape[0] > g.shape[1]:
        g = np.ascontiguousarray(np.rot90(g, -1))   # landscape sheet scanned as portrait
    r, ang = _guard('deskew', lambda: image.deskew(g), (g, 0.0), errors)
    del g
    sc = _guard('scale', lambda: image.find_scale(r), None, errors)
    if not sc or sc.get('fallback'):
        # maybe scanned upside down: the scale bar only reads correctly one way up
        r2 = np.ascontiguousarray(r[::-1, ::-1])
        sc2 = _guard('scale180', lambda: image.find_scale(r2), None, errors)
        if sc2 and not sc2.get('fallback'):
            r, sc = r2, sc2
            ang += 180.0
        else:
            del r2
    s = float(sc['ppf']) if sc else 200.0 / 8.0
    G, norm = image.normalize(r, sc)
    del r
    # the detectors are tuned at the archive's ~200 dpi (about 24.5 px/ft at 1/8" = 1'-0");
    # bring scans at other resolutions to that pixel scale
    f = REF_PPF / s
    if abs(f - 1.0) > 0.06 and 0.4 < f < 1.45:
        G = cv2.resize(G, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        s = s * f
    return run_image(G, s, dict(angle=ang, norm=norm, resample=f), errors, t0, debug)


def empty_result():
    return {'floor_sf': {c: 0.0 for c in FINISH_CODES},
            'wall_lf': {k: 0.0 for k in WALL_TAGS},
            'doors': {'single': 0, 'pair': 0, 'overhead_coiling': 0, 'auto_sliding': 0},
            'fixtures': {k: 0 for k in ['gondola_sections_4ft', 'gondola_sections_3ft', 'endcaps',
                                        'checkouts', 'self_checkouts', 'cooler_doors', 'produce_tables']}}


def run_image(G, s, info=None, errors=None, t0=None, debug=False):
    t0 = time.time() if t0 is None else t0
    errors = [] if errors is None else errors
    info = {} if info is None else info
    result = empty_result()
    bg = float(np.percentile(G[::7, ::7], 75))
    info.update(ppf=s, bg=bg)

    ws, fp, all_ws = _guard('walls', lambda: detect_walls(G, s, bg), ([], None, []), errors)
    wall_map, tag_rows = _guard('legend_walls', lambda: LG.read_wall_map(G, s, all_ws), (WALL_MAP, None), errors)
    codes, code_rows = _guard('legend_codes', lambda: LG.read_finish_codes(G, s, all_ws, FINISH_CODES),
                              (list(FINISH_CODES), None), errors)
    info.update(wall_map=wall_map, codes=codes)
    wl = _guard('wall_lengths', lambda: wall_lengths(ws, s, wall_map), {}, errors)
    tags = [t for t in WALL_TAGS if t in wl] + [t for t in wl if t not in WALL_TAGS]
    result['wall_lf'] = {k: round(float(wl.get(k, 0.0)), 2) for k in tags}

    dc, found = _guard('doors', lambda: count_doors(G, s, bg, ws), (result['doors'], []), errors)
    result['doors'] = {k: int(v) for k, v in dc.items()}

    swings = _guard('swings', lambda: D.swing_mask(G.shape, found), None, errors)
    areas, regions, lab = _guard('floors', lambda: F.floor_areas(G, s, bg, ws, codes, swings), ({}, [], None), errors)
    result['floor_sf'] = {c: round(float(areas.get(c, 0.0)), 1) for c in codes}

    fx = _guard('fixtures', lambda: count_fixtures(G, s, bg, ws), None, errors)
    if fx:
        result['fixtures'] = fx
    info['time'] = time.time() - t0
    info['errors'] = errors
    if debug:
        return result, dict(info=info, walls=ws, doors=found, regions=regions)
    if errors:
        import sys
        for e in errors:
            sys.stderr.write(e + '\n')
    return result
