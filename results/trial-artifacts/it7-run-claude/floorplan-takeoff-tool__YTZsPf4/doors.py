"""Door detection: swing arcs (single / pair), sliding and overhead coiling openings."""
import numpy as np
import cv2
from scipy import ndimage as ndi

from fixtures import variants, match_all
from walls import THICK

DOOR_OFF = 0.25


def swing(R, lw=0.16, off=DOOR_OFF):
    """Leaf + quarter arc, hinge at (0, -off): the wall face lies just outside the box."""
    sh = [('rect', 0, 0, lw, R - off)]
    pts = []
    for t in np.linspace(0, np.pi / 2, 40):
        x = R * np.sin(t); y = R * np.cos(t) - off
        if y >= 0:
            pts.append((x, y))
    for a, b in zip(pts[:-1], pts[1:]):
        sh.append(('line', a[0], a[1], b[0], b[1]))
    return sh, R, R - off


def _hinge_geometry(d, sx, sy):
    """Return hinge point (px), wall orientation ('h'/'v'), opening direction sign,
    and leaf side for a swing detection."""
    name = d['name']
    k = int(name.split('_r')[1][0])
    m = name.endswith('m')
    x, y, w, h = d['x'], d['y'], d['w'], d['h']
    offx = DOOR_OFF * sx; offy = DOOR_OFF * sy
    # canonical k=0, no mirror: wall along top edge (y - off), hinge at top-left,
    # arc sweeps to the right.  Mirror flips x.  Rotation is clockwise.
    # Compute the 4 box corners and pick per orientation.
    if k == 0:
        wall = ('h', y - offy); hinge = (x + w, y - offy) if m else (x, y - offy); sweep = (-1 if m else 1)
    elif k == 1:
        wall = ('v', x + w + offx); hinge = (x + w + offx, y + h) if m else (x + w + offx, y); sweep = (-1 if m else 1)
    elif k == 2:
        wall = ('h', y + h + offy); hinge = (x, y + h + offy) if m else (x + w, y + h + offy); sweep = (1 if m else -1)
    else:
        wall = ('v', x - offx); hinge = (x - offx, y) if m else (x - offx, y + h); sweep = (1 if m else -1)
    return wall, hinge, sweep


def swing_arc(R, off=DOOR_OFF):
    """Arc only (no leaf) in the same frame as swing()."""
    pts = []
    for t in np.linspace(0.12, np.pi / 2 - 0.12, 30):
        x = R * np.sin(t); y = R * np.cos(t) - off
        if y >= 0:
            pts.append((x, y))
    sh = [('line', a[0], a[1], b[0], b[1]) for a, b in zip(pts[:-1], pts[1:])]
    return sh, R, R - off


def arc_score(ink, d, sx, sy):
    """NCC of the arc alone at a swing detection (rejects leaf-only look-alikes)."""
    from fixtures import render
    R = float(d['name'][2:5])
    k = int(d['name'].split('_r')[1][0])
    m = d['name'].endswith('m')
    shapes, W, H = swing_arc(R)
    tpl, _, (mx, my) = render(shapes, W, H, sx, sy, k=k, mirror=m, margin=0.08)
    th, tw = tpl.shape
    x0 = int(round(d['x'] - mx)) - 2; y0 = int(round(d['y'] - my)) - 2
    if x0 < 0 or y0 < 0 or y0 + th + 4 > ink.shape[0] or x0 + tw + 4 > ink.shape[1]:
        return 0.0
    win = ink[y0:y0 + th + 4, x0:x0 + tw + 4]
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    return float(r.max())


def detect_swings(ink, region, sx, sy, thr=0.74, arc_thr=0.18):
    tp = []
    for R in (2.5, 3.0, 3.5, 4.0):
        tp += variants(swing(R), sx, sy, (0, 1, 2, 3), (False, True), 'sw%.1f' % R, margin=0.08)
    dets = match_all(ink, region, tp, thr, nms_frac=0.3)
    out = []
    for d in dets:
        R = float(d['name'][2:5])
        d['arc'] = arc_score(ink, d, sx, sy)
        if d['arc'] < arc_thr:
            continue
        wall, hinge, sweep = _hinge_geometry(d, sx, sy)
        d.update({'R': R, 'wall': wall, 'hinge': hinge, 'sweep': sweep})
        out.append(d)
    return out


def group_swings(dets, sx, sy):
    """Group arcs into single and pair doors. Returns (singles, pairs)."""
    used = set()
    pairs = []
    for i, a in enumerate(dets):
        if i in used:
            continue
        best = None
        for j, b in enumerate(dets):
            if j <= i or j in used:
                continue
            if a['wall'][0] != b['wall'][0]:
                continue
            s = sx if a['wall'][0] == 'h' else sy
            if abs(a['wall'][1] - b['wall'][1]) > 0.4 * s:
                continue
            # along-wall coordinate of hinges
            ia = 0 if a['wall'][0] == 'h' else 1
            pa = a['hinge'][ia]; pb = b['hinge'][ia]
            dist = abs(pa - pb) / s
            # arcs must sweep towards each other
            if pa < pb:
                toward = a['sweep'] > 0 and b['sweep'] < 0
            else:
                toward = a['sweep'] < 0 and b['sweep'] > 0
            if not toward:
                continue
            if abs(dist - (a['R'] + b['R'])) > 0.8:
                continue
            # same side of the wall (both open into the same room)
            ca = a['y'] + a['h'] / 2 if a['wall'][0] == 'h' else a['x'] + a['w'] / 2
            cb = b['y'] + b['h'] / 2 if b['wall'][0] == 'h' else b['x'] + b['w'] / 2
            if (ca - a['wall'][1]) * (cb - b['wall'][1]) <= 0:
                continue
            if best is None or abs(dist - (a['R'] + b['R'])) < best[0]:
                best = (abs(dist - (a['R'] + b['R'])), j)
        if best is not None:
            used.add(i); used.add(best[1])
            pairs.append((a, dets[best[1]]))
    singles = [d for k, d in enumerate(dets) if k not in used]
    return singles, pairs


def wall_lines(runs, W, H):
    t = THICK['EXT']
    out = [
        {'h': True, 'c': t / 2, 'lo': 0.0, 'hi': W, 'type': 'EXT'},
        {'h': True, 'c': H - t / 2, 'lo': 0.0, 'hi': W, 'type': 'EXT'},
        {'h': False, 'c': t / 2, 'lo': 0.0, 'hi': H, 'type': 'EXT'},
        {'h': False, 'c': W - t / 2, 'lo': 0.0, 'hi': H, 'type': 'EXT'},
    ]
    for r in runs:
        out.append({'h': r['h'], 'c': r['c'], 'lo': r['lo'], 'hi': r['hi'], 'type': r['type']})
    return out


def find_openings(rot, walls, ox0, oy0, sx, sy, legend_feats, minw=1.5):
    """Openings = stretches along a wall whose poche does not match the wall's
    own poche: either far from the legend swatch, or a strong outlier against
    the wall's own median texture (robust to noisy scans)."""
    from walls import fill_features_at, fill_distance
    idx = {'EXT': 0, 'P3': 1, 'P2': 2, 'P1': 3}
    keys = ('m', 'g', 'p45', 'p135', 'diag')
    floors = {'m': 0.02, 'g': 0.06, 'p45': 0.07, 'p135': 0.07, 'diag': 0.05}
    out = []
    for wl in walls:
        t = THICK[wl['type']]
        ref = legend_feats[idx[wl['type']]]
        alt = legend_feats[0] if wl['type'] == 'P3' else (legend_feats[1] if wl['type'] == 'EXT' else None)
        if alt is not None and fill_distance(legend_feats[0], legend_feats[1]) >= 1.0:
            alt = None
        if wl['h']:
            c = oy0 + wl['c'] * sy; s = sx; base = ox0
            f0 = c - 0.5 * t * sy; f1 = c + 0.5 * t * sy
        else:
            c = ox0 + wl['c'] * sx; s = sy; base = oy0
            f0 = c - 0.5 * t * sx; f1 = c + 0.5 * t * sx
        lo = base + wl['lo'] * s; hi = base + wl['hi'] * s
        f0, f1 = _refine_faces(rot, wl['h'], f0, f1, lo + 0.8 * s, hi - 0.8 * s)
        win = 1.0 * s; step = 0.25 * s
        pos, dl, feats = [], [], []
        a = lo
        while a + win <= hi:
            ft = fill_features_at(rot, f0, f1, a, a + win, wl['h'])
            if ft is None:
                a += step
                continue
            d = fill_distance(ft, ref)
            if alt is not None:
                d = min(d, fill_distance(ft, alt))
            pos.append(a + win / 2)
            dl.append(d)
            ft = dict(ft); ft['diag'] = ft['p45'] + ft['p135']
            feats.append([ft[k] for k in keys])
            a += step
        if len(pos) < 3:
            continue
        pos = np.array(pos); dl = np.array(dl); F = np.array(feats)
        med = np.median(F, axis=0)
        mad = np.median(np.abs(F - med), axis=0) * 1.4826
        spread = np.maximum(mad, [floors[k] for k in keys])
        z = (np.abs(F - med) / spread).max(axis=1)
        L_ft = (hi - lo) / s
        bad = dl > 2.5
        if L_ft >= 8.0:
            bad |= ((z > 4.5) & (dl > 0.6)) | ((z > 2.5) & (dl > 1.6))
        # bridge flicker of up to 0.75 ft inside an opening
        bad = ndi.binary_closing(bad, structure=np.ones(4, bool)) | bad
        near_end = (pos - lo < 0.9 * s) | (hi - pos < 0.9 * s)
        bad[near_end] = False
        # junctions with perpendicular walls are not openings
        for o in walls:
            if o['h'] == wl['h'] or o is wl:
                continue
            tc = THICK[o['type']]
            if not (o['lo'] - 0.5 * t - 0.3 <= wl['c'] <= o['hi'] + 0.5 * t + 0.3):
                continue
            jp = base + o['c'] * s
            bad[np.abs(pos - jp) < (0.5 * tc + 0.6) * s] = False
        lab, n = ndi.label(bad)
        for i, sl in enumerate(ndi.find_objects(lab)):
            p0 = pos[sl[0].start] - win / 2 + step / 2
            p1 = pos[sl[0].stop - 1] + win / 2 - step / 2
            Lo = (p1 - p0) / s
            if Lo < minw:
                continue
            out.append({'wall': wl, 'lo': (p0 - base) / s, 'hi': (p1 - base) / s, 'w': Lo})
    return out


def merge_openings(ops, gap=1.5):
    ops = sorted(ops, key=lambda o: (id(o['wall']), o['lo']))
    out = []
    for o in ops:
        if out and out[-1]['wall'] is o['wall'] and o['lo'] - out[-1]['hi'] <= gap:
            out[-1]['hi'] = max(out[-1]['hi'], o['hi'])
            out[-1]['w'] = out[-1]['hi'] - out[-1]['lo']
        else:
            out.append(dict(o))
    return out


def classify_openings(rot, ops, swings, ox0, oy0, sx, sy, Rh, Rv, ref):
    """Label openings: 'swing' (has arc), 'sliding' (panels), 'overhead' (wide, empty)."""
    ink = 1.0 - rot
    # darkness of wall face lines as reference for 'solid dark' bars
    face_ink = []
    for o in ops:
        wl = o['wall']
        if wl['type'] != 'EXT':
            continue
    ink_dark = ink > 0.5 * _face_ink(ink, ops, ox0, oy0, sx, sy)
    for o in ops:
        wl = o['wall']
        t = THICK[wl['type']]
        s = sx if wl['h'] else sy
        # swing arcs hinged at this opening
        o['arcs'] = 0
        for d in swings:
            if d['wall'][0] != ('h' if wl['h'] else 'v'):
                continue
            wc = (oy0 + wl['c'] * sy) if wl['h'] else (ox0 + wl['c'] * sx)
            if abs(d['wall'][1] - wc) > (0.5 * t + 0.6) * s:
                continue
            hp = d['hinge'][0] if wl['h'] else d['hinge'][1]
            base = ox0 if wl['h'] else oy0
            hp_ft = (hp - base) / s
            if o['lo'] - 1.2 <= hp_ft <= o['hi'] + 1.2:
                o['arcs'] += 1
        # sliding panels: long *solid* dark bars inside the wall band (raw ink,
        # so fine dashes of an overhead door's centre line do not merge)
        if wl['h']:
            c = oy0 + wl['c'] * sy
            a = int(round(c - 0.32 * t * sy)); b = int(round(c + 0.32 * t * sy)) + 1
            x0 = int(round(ox0 + o['lo'] * sx)); x1 = int(round(ox0 + o['hi'] * sx))
            band = ink_dark[a:b, x0:x1]
        else:
            c = ox0 + wl['c'] * sx
            a = int(round(c - 0.32 * t * sx)); b = int(round(c + 0.32 * t * sx)) + 1
            y0 = int(round(oy0 + o['lo'] * sy)); y1 = int(round(oy0 + o['hi'] * sy))
            band = ink_dark[y0:y1, a:b].T
        runs = _longest_run(band)
        o['panel_ft'] = runs / s
        if o['arcs'] > 0:
            o['kind'] = 'swing'
        elif 1.5 <= o['panel_ft'] <= 0.8 * o['w']:
            o['kind'] = 'sliding'
        elif o['w'] >= 5.0:
            o['kind'] = 'overhead'
        else:
            o['kind'] = 'unknown'
    return ops


def _longest_run(band):
    """Longest horizontal run of True in any row of a 2D boolean array."""
    best = 0
    for row in band:
        if not row.any():
            continue
        r = row.astype(np.int8)
        d = np.diff(np.r_[0, r, 0])
        st = np.nonzero(d == 1)[0]; en = np.nonzero(d == -1)[0]
        if len(st):
            best = max(best, int((en - st).max()))
    return best


def _face_ink(ink, ops, ox0, oy0, sx, sy):
    """Typical peak ink of the exterior wall face lines."""
    vals = []
    for o in ops:
        wl = o['wall']
        if wl['type'] != 'EXT':
            continue
        t = THICK['EXT']
        if wl['h']:
            c = oy0 + wl['c'] * sy
            f = int(round(c - 0.5 * t * sy))
            xs = np.linspace(ox0 + 2 * sx, ox0 + 40 * sx, 30).astype(int)
            for x in xs:
                col = ink[max(f - 3, 0):f + 4, x]
                if col.size:
                    vals.append(col.max())
        break
    if not vals:
        return 0.7
    return float(np.median(vals))


def arrow():
    """Arrowhead pointing -x with a short shaft (sliding-door symbol)."""
    return [('line', 0, 0.22, 0.33, 0.0), ('line', 0, 0.22, 0.33, 0.44),
            ('line', 0, 0.22, 0.5, 0.22)], 0.5, 0.44


def detect_arrows(rot, region, sx, sy, thr=0.5):
    """Arrowhead candidates (sliding-door symbol) with tip position and direction."""
    ink = cv2.GaussianBlur((1.0 - rot).astype(np.float32), (0, 0), 0.7)
    tp = variants(arrow(), sx, sy, (0, 1, 2, 3), (False,), 'arrow', margin=0.1)
    cands = match_all(ink, region, tp, thr, nms_frac=0.3)
    arrows = []
    for d in cands:
        k = int(d['name'].split('_r')[1][0])
        cx = d['x'] + d['w'] / 2; cy = d['y'] + d['h'] / 2
        if k == 0:
            tip = (d['x'], cy); dirv = (-1, 0)
        elif k == 1:
            tip = (cx, d['y']); dirv = (0, -1)
        elif k == 2:
            tip = (d['x'] + d['w'], cy); dirv = (1, 0)
        else:
            tip = (cx, d['y'] + d['h']); dirv = (0, 1)
        arrows.append({'tip': tip, 'dir': dirv, 'score': d['score']})
    return arrows


def opening_has_arrows(o, arrows, ox0, oy0, sx, sy):
    """Outward-pointing arrows at both ends of the opening, on one side of the wall."""
    w_ = o['wall']
    if w_['h']:
        neg, pos_, along, across, s, sa, base, abase = (-1, 0), (1, 0), 0, 1, sx, sy, ox0, oy0
    else:
        neg, pos_, along, across, s, sa, base, abase = (0, -1), (0, 1), 1, 0, sy, sx, oy0, ox0
    c = abase + w_['c'] * sa
    lo = base + o['lo'] * s; hi = base + o['hi'] * s
    best = None
    for side in (-1, 1):
        # (the detected opening may be only part of the real one)
        L = [a for a in arrows if a['dir'] == neg and lo - 3.0 * s < a['tip'][along] < lo + 1.2 * s
             and 0.3 * sa < side * (a['tip'][across] - c) < 1.4 * sa]
        R = [a for a in arrows if a['dir'] == pos_ and hi - 1.2 * s < a['tip'][along] < hi + 3.0 * s
             and 0.3 * sa < side * (a['tip'][across] - c) < 1.4 * sa]
        for a in L:
            for b in R:
                if abs(a['tip'][across] - b['tip'][across]) > 0.15 * sa:
                    continue
                sep = (b['tip'][along] - a['tip'][along]) / s
                if not (3.5 <= sep <= 10.0):
                    continue
                sc = min(a['score'], b['score'])
                if best is None or sc > best:
                    best = sc
    return best


def count_doors(rot, S, sref, ref, runs, W, H, bldg, sx, sy, legend_wall_feats, debug=None, swings=None):
    """Full door count. Returns dict with single, pair, overhead_coiling, auto_sliding."""
    ox0, oy0, ox1, oy1 = bldg['outer']
    hh, ww = rot.shape
    if swings is None:
        inkb = cv2.GaussianBlur((1.0 - rot).astype(np.float32), (0, 0), 0.7)
        region = (max(int(ox0 - 4 * sx), 0), max(int(oy0 - 4 * sy), 0),
                  min(int(ox1 + 4 * sx), ww), min(int(oy1 + 4 * sy), hh))
        swings = detect_swings(inkb, region, sx, sy)
    singles, pairs = group_swings(swings, sx, sy)
    n_oh = n_sliding = 0
    try:
        n_oh, n_sliding = _classify_non_swing(rot, S, sref, ref, runs, W, H, bldg, sx, sy,
                                              legend_wall_feats, swings, debug)
    except Exception:
        import traceback
        traceback.print_exc()
    if debug is not None:
        debug['swings'] = swings
    return {'single': len(singles), 'pair': len(pairs),
            'overhead_coiling': n_oh, 'auto_sliding': n_sliding}


def _classify_non_swing(rot, S, sref, ref, runs, W, H, bldg, sx, sy, legend_wall_feats, swings, debug):
    ox0, oy0, ox1, oy1 = bldg['outer']
    hh, ww = rot.shape
    wl = wall_lines(runs, W, H)
    ops = merge_openings(find_openings(rot, wl, ox0, oy0, sx, sy, legend_wall_feats))
    # arcs per opening
    for o in ops:
        w_ = o['wall']
        t = THICK[w_['type']]
        s = sx if w_['h'] else sy
        wc = (oy0 + w_['c'] * sy) if w_['h'] else (ox0 + w_['c'] * sx)
        base = ox0 if w_['h'] else oy0
        o['arcs'] = 0
        for d in swings:
            if d['wall'][0] != ('h' if w_['h'] else 'v'):
                continue
            if abs(d['wall'][1] - wc) > (0.5 * t + 0.6) * s:
                continue
            hp = (d['hinge'][0] if w_['h'] else d['hinge'][1])
            hp_ft = (hp - base) / s
            if o['lo'] - 1.2 <= hp_ft <= o['hi'] + 1.2:
                o['arcs'] += 1
        o['kind'] = 'swing' if o['arcs'] else None
    # sliding: two panels drawn at different offsets inside the wall, or
    # outward arrows at both ends of a (not too wide) arc-free opening
    aregion = (max(int(ox0 - 2 * sx), 0), max(int(oy0 - 2 * sy), 0),
               min(int(ox1 + 2 * sx), ww), min(int(oy1 + 2 * sy), hh))
    arrows = detect_arrows(rot, aregion, sx, sy, thr=0.5)
    ink = 1.0 - rot
    face = _face_ink(ink, ops, ox0, oy0, sx, sy)
    n_sliding = 0
    for o in ops:
        if o['kind'] is not None or o['w'] < 3.0:
            continue
        sc = opening_has_arrows(o, arrows, ox0, oy0, sx, sy)
        o['arrow_score'] = sc
        po = panel_offset(ink, o, ox0, oy0, sx, sy, face)
        o['panel_off'] = po
        if po is not None and po >= 0.06:
            o['kind'] = 'sliding'
        elif sc is not None and sc >= 0.6 and o['w'] < 9.0:
            o['kind'] = 'sliding'
        if o['kind'] == 'sliding':
            n_sliding += 1
    apairs = []
    n_oh = 0
    for o in ops:
        if o['kind'] is None:
            if o['wall']['type'] == 'EXT' and o['w'] >= 7.0:
                o['kind'] = 'overhead'
                n_oh += 1
            else:
                o['kind'] = 'unknown'
    if debug is not None:
        debug['swings'] = swings; debug['openings'] = ops; debug['arrow_pairs'] = apairs
    return n_oh, n_sliding


def panel_offset(ink, o, ox0, oy0, sx, sy, face):
    """Across-wall offset (ft) between the dark bars in the left and right thirds
    of an opening (sliding panels are drawn side by side at different offsets);
    None if either third lacks a solid bar."""
    w_ = o['wall']
    t = THICK[w_['type']]
    s = sx if w_['h'] else sy
    sa = sy if w_['h'] else sx
    base = ox0 if w_['h'] else oy0
    abase = oy0 if w_['h'] else ox0
    c = abase + w_['c'] * sa
    a = int(round(c - 0.35 * t * sa)); b = int(round(c + 0.35 * t * sa)) + 1
    L = o['w'] - 1.0
    if L < 2.0:
        return None
    thirds = [(o['lo'] + 0.5, o['lo'] + 0.5 + L / 3.0), (o['hi'] - 0.5 - L / 3.0, o['hi'] - 0.5)]
    cents = []
    for (p0, p1) in thirds:
        i0 = int(round(base + p0 * s)); i1 = int(round(base + p1 * s))
        if w_['h']:
            band = ink[a:b, i0:i1]
        else:
            band = ink[i0:i1, a:b].T
        if band.size == 0:
            return None
        dark = band > 0.5 * face
        cover = dark.mean(axis=1)          # per across-wall row
        k = int(np.argmax(cover))
        if cover[k] < 0.7:
            return None
        wgt = np.clip(band - 0.3 * face, 0, None).sum(axis=1)
        rows = np.arange(len(wgt))
        sel = (rows >= k - 3) & (rows <= k + 3)
        cents.append(float((wgt[sel] * rows[sel]).sum() / max(wgt[sel].sum(), 1e-6)))
    return abs(cents[0] - cents[1]) / sa


def _refine_faces(rot, horiz, f0, f1, lo, hi):
    """Locate the two face lines of a wall from its median cross-profile."""
    from core import _line_center
    a = int(np.floor(min(f0, f1))) - 6; b = int(np.ceil(max(f0, f1))) + 7
    lo = int(lo); hi = int(hi)
    if hi - lo < 10 or a < 0:
        return f0, f1
    if horiz:
        band = rot[a:b, lo:hi]
        prof = np.median(band, axis=1)
    else:
        band = rot[lo:hi, a:b]
        prof = np.median(band, axis=0)
    n0 = _line_center(prof, f0 - a, 2) + a
    n1 = _line_center(prof, f1 - a, 2) + a
    if abs(n0 - f0) > 3 or abs(n1 - f1) > 3 or n1 - n0 < 0.6 * (f1 - f0):
        return f0, f1
    return n0, n1
