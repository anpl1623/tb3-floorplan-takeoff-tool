"""Door detection and classification at wall openings."""
import numpy as np
import cv2


def polar_profile(G, hx, hy, ux, uy, nx, ny, r0, r1, th0=10, th1=80, nth=48):
    ths = np.radians(np.linspace(th0, th1, nth))
    rs = np.arange(int(r0), int(r1) + 1)
    cx = np.cos(ths)[None, :] * ux + np.sin(ths)[None, :] * nx
    cy = np.cos(ths)[None, :] * uy + np.sin(ths)[None, :] * ny
    X = hx + rs[:, None] * cx
    Y = hy + rs[:, None] * cy
    xs = np.clip(np.round(X).astype(int), 0, G.shape[1] - 1)
    ys = np.clip(np.round(Y).astype(int), 0, G.shape[0] - 1)
    V = G[ys, xs].astype(np.float32)
    return rs, V


def arc_dip(G, hx, hy, ux, uy, nx, ny, r_exp, lo=0.8, hi=1.2):
    """Depth of the dip in the theta-averaged polar profile near r_exp, and the
    fraction of angles that are dark at the dip radius."""
    rs, V = polar_profile(G, hx, hy, ux, uy, nx, ny, 0.45 * r_exp, 1.55 * r_exp)
    A = V.mean(1)
    sel = (rs >= lo * r_exp) & (rs <= hi * r_exp)
    if not sel.any():
        return 0.0, 0.0, r_exp
    i = np.argmin(np.where(sel, A, 1e9))
    # allow the arc to straddle neighbouring radii
    j0, j1 = max(0, i - 2), min(len(rs), i + 3)
    vmin = V[j0:j1].min(0)
    ref = np.median(A)
    dip = ref - A[i]
    frac = float((vmin < ref - 30).mean())
    return float(dip), frac, float(rs[i])


def swing_scores(G, w, g0, g1):
    """Arc evidence for single (radius = width) and pair (radius = width/2) leaves,
    for both hinge ends and both sides of the wall."""
    width = float(g1 - g0)
    out = []
    for side in (-1, 1):
        face = w['t0'] if side < 0 else w['t1']
        nvec = (0.0, float(side)) if w['o'] == 'H' else (float(side), 0.0)
        def pt(a, c):
            return (a, c) if w['o'] == 'H' else (c, a)
        for hinge, u in ((g0, 1.0), (g1, -1.0)):
            uvec = (u, 0.0) if w['o'] == 'H' else (0.0, u)
            d1 = (0.0, 0.0, 0.0); d2 = (0.0, 0.0, 0.0)
            for da in (-4.0, 0.0, 4.0):
                hx, hy = pt(hinge + da * u, face)
                e1 = arc_dip(G, hx, hy, uvec[0], uvec[1], nvec[0], nvec[1], width)
                e2 = arc_dip(G, hx, hy, uvec[0], uvec[1], nvec[0], nvec[1], width / 2.0)
                if (e1[1], e1[0]) > (d1[1], d1[0]):
                    d1 = e1
                if (e2[1], e2[0]) > (d2[1], d2[0]):
                    d2 = e2
            out.append(dict(side=side, hinge=hinge, single=d1, half=d2))
    return out


def band_patch(G, w, g0, g1, pad):
    lo = int(np.floor(w['t0'])) - pad
    hi = int(np.ceil(w['t1'])) + pad + 1
    a0, a1 = int(g0), int(g1) + 1
    if w['o'] == 'H':
        patch = G[lo:hi, a0:a1]
    else:
        patch = G[a0:a1, lo:hi].T
    return patch.astype(np.float32), lo


def _max_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def panel_score(G, bg, w, g0, g1):
    """Longest solid heavy line parallel to the wall inside the wall band, as a
    fraction of the opening width (sliding door panels)."""
    patch, lo = band_patch(G, w, g0, g1, 0)
    rows = np.arange(lo, lo + patch.shape[0])
    sel = (rows >= w['t0'] + 2.5) & (rows <= w['t1'] - 2.5)
    if not sel.any() or patch.shape[1] < 5:
        return 0.0
    sub = cv2.blur(patch[sel], (3, 1))
    best = 0
    for row in sub:
        best = max(best, _max_run(row < bg - 110))
    return best / float(patch.shape[1])


def offset_dashed_line(G, bg, w, g0, g1, s):
    """Strongest line parallel to the wall, 0.5-3 ft off either face, spanning the opening."""
    pad = int(3.2 * s)
    patch, lo = band_patch(G, w, g0, g1, pad)
    rows = np.arange(lo, lo + patch.shape[0])
    prof = patch.mean(1)
    best = 0.0
    for side in (-1, 1):
        if side < 0:
            sel = (rows <= w['t0'] - 0.5 * s) & (rows >= w['t0'] - 3.0 * s)
        else:
            sel = (rows >= w['t1'] + 0.5 * s) & (rows <= w['t1'] + 3.0 * s)
        if not sel.any():
            continue
        pr = prof[sel]
        # local background from the median of the profile
        dark = np.median(pr) - pr.min()
        best = max(best, float(dark))
    return best


def classify_opening(G, bg, w, g0, g1, s):
    res = swing_scores(G, w, g0, g1)
    single = max(res, key=lambda r: (r['single'][1], r['single'][0]))
    pair_by_side = []
    for side in (-1, 1):
        rr = [r for r in res if r['side'] == side]
        pair_by_side.append((min(r['half'][1] for r in rr), min(r['half'][0] for r in rr), side))
    pair = max(pair_by_side)
    feats = dict(width_ft=(g1 - g0) / s, single_frac=single['single'][1], single_dip=single['single'][0],
                 pair_frac=pair[0], pair_dip=pair[1])
    if pair[0] >= 0.7 and pair[1] >= 20 and (pair[0] >= single['single'][1] or (g1 - g0) / s > 4.5):
        feats['swings'] = [(g0, pair[2], (g1 - g0) / 2.0, 1.0), (g1, pair[2], (g1 - g0) / 2.0, -1.0)]
        return 'pair', feats
    if single['single'][1] >= 0.7 and single['single'][0] >= 20:
        u = 1.0 if single['hinge'] == g0 else -1.0
        feats['swings'] = [(single['hinge'], single['side'], float(g1 - g0), u)]
        return 'single', feats
    ps = panel_score(G, bg, w, g0, g1)
    od = offset_dashed_line(G, bg, w, g0, g1, s)
    feats.update(panel=ps, offset_line=od)
    if ps >= 0.3:
        return 'auto_sliding', feats
    if od >= 25 or (g1 - g0) / s >= 8.0:
        return 'overhead_coiling', feats
    # weak evidence: fall back on the opening width together with partial evidence
    wft = (g1 - g0) / s
    if 2.4 <= wft <= 4.2 and single['single'][1] >= 0.5 and single['single'][0] >= 12:
        u = 1.0 if single['hinge'] == g0 else -1.0
        feats['swings'] = [(single['hinge'], single['side'], float(g1 - g0), u)]
        return 'single', feats
    if 4.6 <= wft <= 8.0 and pair[0] >= 0.5 and pair[1] >= 12:
        feats['swings'] = [(g0, pair[2], (g1 - g0) / 2.0, 1.0), (g1, pair[2], (g1 - g0) / 2.0, -1.0)]
        return 'pair', feats
    if 4.6 <= wft <= 8.0 and ps >= 0.15:
        return 'auto_sliding', feats
    return 'unknown', feats


def swing_mask(shape, doors, pad=3):
    """Quarter-disc masks of door swings (hinge, side, radius, direction along wall)."""
    m = np.zeros(shape, np.uint8)
    for d in doors:
        w = d['wall']
        for (hinge, side, r, u) in d['feats'].get('swings', []):
            face = w['t0'] if side < 0 else w['t1']
            if w['o'] == 'H':
                cx, cy = hinge, face
                # angles in image coords: along wall = +x (u>0) or -x; away from wall = side*y
                a_along = 0.0 if u > 0 else 180.0
                a_away = 90.0 if side > 0 else 270.0
            else:
                cx, cy = face, hinge
                a_along = 90.0 if u > 0 else 270.0
                a_away = 0.0 if side > 0 else 180.0
            a0, a1 = sorted([a_along, a_away])
            if a1 - a0 > 180:
                a0, a1 = a1, a0 + 360
            cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(r + pad), int(r + pad)), 0, a0, a1, 1, -1)
    return m > 0
