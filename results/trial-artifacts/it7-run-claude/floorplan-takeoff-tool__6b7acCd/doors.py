"""Door detection: openings in walls and door-type classification."""
import numpy as np
import cv2
from common import CAL


def wall_band(dk, w, pad=0):
    """Return (band, r0, lo): band rows run across the wall, columns along it starting at lo."""
    c, t = w['c'], w['t']
    r0 = int(np.floor(c - t / 2 - pad))
    r1 = int(np.ceil(c + t / 2 + pad)) + 1
    lo = int(np.floor(min(w['lo'], w.get('a', w['lo']))))
    hi = int(np.ceil(max(w['hi'], w.get('b', w['hi']))))
    if w['o'] == 'H':
        band = dk[r0:r1, lo:hi]
    else:
        band = dk[lo:hi, r0:r1].T
    return band, r0, lo


def _peaks(v, th):
    out = []
    n = len(v)
    i = 0
    while i < n:
        if v[i] > th:
            j = i
            while j < n and v[j] > th:
                j += 1
            seg = v[i:j]
            out.append(i + int(np.argmax(seg)))
            i = j
        else:
            i += 1
    return out


def find_openings2(dk, walls, ppf, min_ft=2.0, max_ft=18.0, lm=None, dk_alt=None, lm_alt=None):
    """Openings = intervals between consecutive thin jamb caps across the wall band.

    For walls with solid dark poche the alternative (hollowed) images are used."""
    out = []
    for w in walls:
        band, r0, lo = wall_band(dk, w)
        lband = wall_band(lm, w)[0] if lm is not None else None
        if dk_alt is not None:
            c_, t_ = w['c'], w['t']
            ii0 = int(np.ceil(c_ - t_ / 2 - r0 + 2.5)); ii1 = int(np.floor(c_ + t_ / 2 - r0 - 2.5)) + 1
            if ii1 > ii0 and np.median(band[ii0:ii1]) > 0.55 * CAL['k']:
                band, r0, lo = wall_band(dk_alt, w)
                lband = wall_band(lm_alt, w)[0] if lm_alt is not None else None
        c, t = w['c'], w['t']
        ya = c - t / 2 - r0
        yb = c + t / 2 - r0
        i0 = int(np.ceil(ya + 2.5)); i1 = int(np.floor(yb - 2.5)) + 1
        if i1 - i0 < 1:
            continue
        inner = band[i0:i1]
        if lband is not None:
            cap = ((inner > 0.45 * CAL['k']) & (lband[i0:i1] > 0.2 * CAL['k'])).mean(0)
        else:
            cap = (inner > 0.45 * CAL['k']).mean(0)
        jambs = []
        n = len(cap)
        i = 0
        while i < n:
            if cap[i] > 0.6:
                j = i
                while j < n and cap[j] > 0.6:
                    j += 1
                if j - i <= max(6, int(0.25 * ppf)) and (j - i >= 2 or cap[i:j].max() >= 0.9):
                    seg = cap[i:j]
                    jambs.append(i + int(np.argmax(seg)))
                i = j
            else:
                i += 1
        # ignore caps in the junction zones beyond the wall's own extent and inside crossing walls
        lo_ok = w['lo'] + 2 - lo
        hi_ok = w['hi'] - 2 - lo
        bad = []
        for ot in walls:
            if ot is w or ot['o'] == w['o']:
                continue
            if ot['lo'] - 3 <= c <= ot['hi'] + 3:
                bad.append((ot['c'] - ot['t'] / 2 - 3 - lo, ot['c'] + ot['t'] / 2 + 3 - lo))
        jambs = [j for j in jambs if lo_ok <= j <= hi_ok and not any(b0 <= j <= b1 for b0, b1 in bad)]
        # interior line coverage between jambs (sliding panels fill the band with parallel lines)
        j0 = int(np.ceil(ya + 1.8)); j1 = int(np.floor(yb - 1.8)) + 1
        if lband is not None:
            linner = lband[j0:j1] > 0.25 * CAL['k']
        else:
            linner = band[j0:j1] > 0.3 * CAL['k']

        def anycov(a, b):
            a2, b2 = a + 3, b - 2
            if b2 - a2 < 3:
                return 0.0
            return float(linner[:, a2:b2].any(0).mean())
        k = 0
        while k < len(jambs) - 1:
            a, b = jambs[k], jambs[k + 1]
            wid = (b - a) / ppf
            # a sliding door whose panel overlap looks like a jamb: merge the two halves
            if k + 2 < len(jambs):
                c2 = jambs[k + 2]
                if (c2 - a) / ppf <= 12.0 and anycov(a, b) > 0.8 and anycov(b, c2) > 0.8:
                    out.append(dict(wall=w, u0=lo + a, u1=lo + c2, width=(c2 - a) / ppf))
                    k += 3
                    continue
            if min_ft <= wid <= max_ft:
                out.append(dict(wall=w, u0=lo + a, u1=lo + b, width=wid))
                k += 2
            else:
                k += 1
    return out


def _sample(dk, pts, r=1):
    h, w = dk.shape
    vals = []
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        if r <= ix < w - r and r <= iy < h - r:
            vals.append(dk[iy - r:iy + r + 1, ix - r:ix + r + 1].max())
        else:
            vals.append(0.0)
    return np.array(vals)


def arc_score(dk, o, c_face, hinge, toward, side, R, th=0.18):
    """Coverage of a quarter arc centred at the hinge on the wall face.

    o: wall orientation; along-axis coordinate u, across-axis v. hinge at (u=hinge, v=c_face).
    toward: +1/-1 direction along the wall from hinge to the other jamb.
    side: +1/-1 direction across the wall (away from the wall)."""
    angs = np.linspace(np.radians(12), np.radians(78), 24)
    us = hinge + toward * R * np.cos(angs)
    vs = c_face + side * R * np.sin(angs)
    pts = [(u, v) if o == 'H' else (v, u) for u, v in zip(us, vs)]
    vals = _sample(dk, pts)
    return float((vals > th).mean())


def leaf_score(dk, o, c_face, hinge, side, R, th=0.3):
    """Coverage of a door leaf: line from hinge perpendicular to the wall, length R."""
    ss = np.linspace(0.15, 0.9, 16) * R
    pts = []
    for s_ in ss:
        u, v = hinge, c_face + side * s_
        pts.append((u, v) if o == 'H' else (v, u))
    vals = _sample(dk, pts, r=2)
    return float((vals > th).mean())


def line_map(dk, k=7):
    """Thin dark structures relative to local background (top-hat on darkness)."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return dk - cv2.morphologyEx(dk, cv2.MORPH_OPEN, se)


def panel_score(lm, op):
    """Sliding-door panels: two parallel lines inside the band, one covering the first half
    of the opening and another (different row) the second half."""
    w = op['wall']
    band, r0, lo = wall_band(lm, w)
    c, t = w['c'], w['t']
    ya = c - t / 2 - r0
    yb = c + t / 2 - r0
    i0 = int(np.ceil(ya + 2.0)); i1 = int(np.floor(yb - 2.0)) + 1
    W = op['u1'] - op['u0']
    a = op['u0'] - lo + 4
    b = op['u1'] - lo - 4
    if i1 <= i0 or b - a < 10:
        return 0.0, 0.0
    inner = band[i0:i1, a:b] > 0.25 * CAL['k']
    n = inner.shape[1]
    left = inner[:, int(0.05 * n):int(0.4 * n)].mean(1)
    right = inner[:, int(0.6 * n):int(0.95 * n)].mean(1)
    anycov = inner.any(0).mean()
    best = 0.0
    for r1 in range(len(left)):
        for r2 in range(len(right)):
            if abs(r1 - r2) >= 2:
                best = max(best, min(left[r1], right[r2]))
    return float(best), float(anycov)


def dashed_score(lm, op, ppf):
    """Dashed line parallel to the wall on either side, spanning the opening (coiling door hood)."""
    w = op['wall']
    o = w['o']
    W = op['u1'] - op['u0']
    best = 0.0
    for side in (-1, 1):
        face = w['c'] + side * w['t'] / 2
        for dft in np.arange(0.3, 3.01, 0.05):
            v = face + side * dft * ppf
            us = np.arange(op['u0'] + 0.05 * W, op['u1'] - 0.05 * W, 1.0)
            pts = [(u, v) for u in us] if o == 'H' else [(v, u) for u in us]
            vals = _sample(lm, pts, r=1)
            on = vals > 0.2 * CAL['k']
            if len(on) < 10:
                continue
            frac = on.mean()
            # count dash runs
            runs = np.count_nonzero(np.diff(on.astype(int)) == 1) + int(on[0])
            dashes_per_ft = runs / (len(on) / ppf)
            if 0.4 <= frac <= 0.92 and 0.6 <= dashes_per_ft <= 4.0:
                best = max(best, float(frac))
    return best


def classify_opening(dk, lm, op, ppf, lm2=None):
    if lm2 is None:
        lm2 = lm
    w = op['wall']
    u0, u1 = op['u0'], op['u1']
    W = u1 - u0
    sc = dict(single=0.0, pair=0.0)
    for side in (-1, 1):
        face = w['c'] + side * w['t'] / 2
        for hinge, tw in ((u0, 1), (u1, -1)):
            a = arc_score(lm, w['o'], face, hinge, tw, side, W, th=0.15 * CAL['k'])
            l = leaf_score(lm2, w['o'], face, hinge, side, W, th=0.2 * CAL['k'])
            sc['single'] = max(sc['single'], min(a, l))
        tk = CAL['k']
        a1 = arc_score(lm, w['o'], face, u0, 1, side, W / 2, th=0.15 * tk)
        a2 = arc_score(lm, w['o'], face, u1, -1, side, W / 2, th=0.15 * tk)
        l1 = leaf_score(lm2, w['o'], face, u0, side, W / 2, th=0.2 * tk)
        l2 = leaf_score(lm2, w['o'], face, u1, side, W / 2, th=0.2 * tk)
        sc['pair'] = max(sc['pair'], min(a1, a2, l1, l2))
    sc['panel'], sc['any'] = panel_score(lm, op)
    sc['dash'] = dashed_score(lm, op, ppf)
    if sc['panel'] >= 0.8:
        kind = 'auto_sliding'
    elif sc['pair'] >= 0.75 and sc['pair'] >= sc['single'] - 0.1:
        kind = 'pair'
    elif sc['single'] >= 0.75:
        kind = 'single'
    elif sc['dash'] > 0 and W >= 5 * ppf:
        kind = 'overhead_coiling'
    elif W >= 7 * ppf and w.get('ext') and sc['single'] < 0.5 and sc['pair'] < 0.5 and sc['panel'] < 0.5:
        # wide exterior opening without swing or sliding leaves
        kind = 'overhead_coiling'
    else:
        kind = None
    return kind, sc
