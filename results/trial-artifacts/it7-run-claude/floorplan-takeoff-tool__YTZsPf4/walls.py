"""Wall (partition) detection, typing and centreline lengths."""
import numpy as np
import cv2
from scipy import ndimage as ndi

# standard thicknesses (ft) by type
THICK = {'EXT': 8 / 12.0, 'P3': 8 / 12.0, 'P2': 7.25 / 12.0, 'P1': 4.875 / 12.0}


def ridge_maps(rot, k=3, L=9):
    D = 1.0 - rot
    Dx = cv2.blur(D, (L, 1))
    Rh = Dx - 0.5 * (np.roll(Dx, k, axis=0) + np.roll(Dx, -k, axis=0))
    Dy = cv2.blur(D, (1, L))
    Rv = Dy - 0.5 * (np.roll(Dy, k, axis=1) + np.roll(Dy, -k, axis=1))
    return Rh, Rv


def face_reference(Rh, Rv, bldg):
    ox0, oy0, ox1, oy1 = bldg['outer']
    vals = []
    for x in np.linspace(ox0 + 60, ox1 - 60, 150).astype(int):
        vals.append(Rh[int(oy0) - 3:int(oy0) + 4, x].max())
        vals.append(Rh[int(oy1) - 3:int(oy1) + 4, x].max())
    for y in np.linspace(oy0 + 60, oy1 - 60, 150).astype(int):
        vals.append(Rv[y, int(ox0) - 3:int(ox0) + 4].max())
        vals.append(Rv[y, int(ox1) - 3:int(ox1) + 4].max())
    return float(np.median(vals))


def face_segments(R, horiz, thr, minlen, box):
    """Line segments from a ridge map. Returns list of (pos, lo, hi, strength)."""
    X0, Y0, X1, Y1 = box
    sub = R[Y0:Y1, X0:X1]
    m = (sub > thr).astype(np.uint8)
    k = np.ones((1, minlen), np.uint8) if horiz else np.ones((minlen, 1), np.uint8)
    op = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    lab, n = ndi.label(op)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        ys, xs = sl
        if horiz:
            a0, a1 = xs.start, xs.stop
            r0 = max(ys.start - 2, 0); r1 = min(ys.stop + 2, sub.shape[0])
            w = np.clip(sub[r0:r1, a0:a1], 0, None) * (lab[r0:r1, a0:a1] == i + 1)
            prof = w.sum(axis=1)
            pos = (prof * np.arange(r0, r1)).sum() / max(prof.sum(), 1e-6)
            strength = float(np.median(sub[int(round(pos)), a0:a1]))
            out.append((pos + Y0, a0 + X0, a1 + X0, strength))
        else:
            a0, a1 = ys.start, ys.stop
            c0 = max(xs.start - 2, 0); c1 = min(xs.stop + 2, sub.shape[1])
            w = np.clip(sub[a0:a1, c0:c1], 0, None) * (lab[a0:a1, c0:c1] == i + 1)
            prof = w.sum(axis=0)
            pos = (prof * np.arange(c0, c1)).sum() / max(prof.sum(), 1e-6)
            strength = float(np.median(sub[a0:a1, int(round(pos))]))
            out.append((pos + X0, a0 + Y0, a1 + Y0, strength))
    return out


def pair_faces(segs, ppf, tmin=0.28, tmax=0.85, minov=0.4):
    segs = sorted(segs)
    out = []
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            d = b[0] - a[0]
            if d < tmin * ppf:
                continue
            if d > tmax * ppf:
                break
            lo = max(a[1], b[1]); hi = min(a[2], b[2])
            if hi - lo > minov * ppf:
                out.append({'pos': 0.5 * (a[0] + b[0]), 'lo': lo, 'hi': hi,
                            'f0': a[0], 'f1': b[0], 't': d / ppf,
                            's': min(a[3], b[3])})
    return out


# ---------------------------------------------------------------- fill features

def fill_features_at(rot, f0, f1, lo, hi, horiz):
    """Poche features for the strip between face lines f0 < f1 over [lo, hi].

    Statistics use only the middle 40% of the thickness so that the face lines
    (whose blurred tails reach 2-3 px) do not contaminate them.
    """
    t = f1 - f0
    a = int(np.floor(f0 + 1.0)); b = int(np.ceil(f1 - 1.0)) + 1
    lo = int(np.ceil(lo)); hi = int(np.floor(hi))
    if hi - lo < 8 or b - a < 3:
        return None
    if horiz:
        strip = rot[a:b, lo:hi]
    else:
        strip = rot[lo:hi, a:b].T
    ink = 1.0 - strip.astype(np.float32)
    sm = cv2.GaussianBlur(ink, (0, 0), 0.7)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    if not horiz:
        # back to sheet orientation: transposed strip has x<->y swapped
        gx, gy = gy, gx
    r0 = int(np.floor(f0 + 0.3 * t)) - a
    r1 = int(np.ceil(f1 - 0.3 * t)) - a + 1
    r0 = max(r0, 1); r1 = min(r1, ink.shape[0] - 1)
    if r1 <= r0:
        r0 = max(min(r0, ink.shape[0] - 2), 0); r1 = r0 + 1
    sl = (slice(r0, r1), slice(2, ink.shape[1] - 2))
    inkc = ink[sl]; gxc = gx[sl]; gyc = gy[sl]
    m = float(inkc.mean())
    e = gxc * gxc + gyc * gyc
    tot = float(e.sum()) + 1e-9
    g = float(np.sqrt(e.mean()))
    pxy = gxc * gyc
    p45 = float(np.clip(pxy, 0, None).sum() / (0.5 * tot))
    p135 = float(np.clip(-pxy, 0, None).sum() / (0.5 * tot))
    return {'m': m, 'g': g, 'p45': p45, 'p135': p135, 'n': int(inkc.size)}


def fill_distance(fa, fb):
    if fa is None or fb is None:
        return 9.0
    dm = (fa['m'] - fb['m']) / max(0.035, 0.2 * max(fa['m'], fb['m']))
    dg = (fa['g'] - fb['g']) / max(0.05, 0.3 * max(fa['g'], fb['g']))
    d = dm * dm + 0.7 * dg * dg
    # orientation only meaningful when textured
    if min(fa['g'], fb['g']) > 0.12:
        d += 6.0 * ((fa['p45'] - fb['p45']) ** 2 + (fa['p135'] - fb['p135']) ** 2)
    return d


def legend_wall_features(rot, boxes):
    feats = []
    for (x0, y0, x1, y1) in boxes:
        feats.append(fill_features_at(rot, y0, y1, x0 + 4, x1 - 4, True))
    return feats


# ---------------------------------------------------------------- assembly

def _snap(v, tol=0.2):
    r = round(v)
    if abs(v - r) <= tol:
        return float(r)
    h = round(v * 2) / 2.0
    if abs(v - h) <= 0.12:
        return float(h)
    return float(v)


def classify_piece(rot, p, horiz, lf, ppf, win_ft=1.5):
    """Classify a face-pair piece by windows. Returns (type, score, details)."""
    L = p['hi'] - p['lo']
    win = max(int(win_ft * ppf), 16)
    step = max(win // 2, 8)
    thin = p['t'] < 0.5
    cand = ['P1'] if thin else ['P3', 'P2']
    ext_like_p3 = fill_distance(lf[0], lf[1]) < 1.0
    idx = {'EXT': 0, 'P3': 1, 'P2': 2, 'P1': 3}
    votes = {c: 0.0 for c in cand}
    nwin = 0; good = 0
    a = p['lo'] + 3
    while a + win <= p['hi'] - 3 or (nwin == 0 and a < p['hi'] - 8):
        b = min(a + win, p['hi'] - 3)
        ft = fill_features_at(rot, p['f0'], p['f1'], a, b, horiz)
        nwin += 1
        if ft is not None:
            ds = {}
            for c in cand:
                if c == 'P3' and ext_like_p3:
                    ds[c] = min(fill_distance(ft, lf[0]), fill_distance(ft, lf[1]))
                else:
                    ds[c] = fill_distance(ft, lf[idx[c]])
            best = min(ds, key=ds.get)
            if ds[best] < 2.5:
                votes[best] += (b - a)
                good += (b - a)
        a += step
    if good <= 0:
        # thin face pair whose poche could not be matched (e.g. a few pixels of
        # blurred hatch): thickness alone identifies a 3-5/8" stud partition
        if thin and 0.3 <= p['t'] <= 0.5 and L >= 2.5 * ppf and p.get('s', 0) > 0:
            return 'P1', 0.36
        return None, 0.0
    best = max(votes, key=votes.get)
    return best, votes[best] / max(L, 1)


def calibrate_dims(pieces, bldg, W0, H0, ppf):
    """Choose building W, H (ft) near the scale-bar estimate that put the wall
    centrelines closest to whole feet."""
    ox0, oy0, ox1, oy1 = bldg['outer']
    out = []
    for horiz, span, origin, F0 in ((False, ox1 - ox0, ox0, W0), (True, oy1 - oy0, oy0, H0)):
        qs = [q for q in pieces if q['h'] == horiz and q['score'] >= 0.5]
        pos = np.array([q['px']['pos'] for q in qs], float)
        wts = np.array([q['px']['hi'] - q['px']['lo'] for q in qs], float)
        raw = span / ppf
        if len(pos) < 3 or wts.sum() < 30 * ppf:
            out.append(F0)
            continue
        best = None
        cands = set()
        for k in range(int(np.floor(raw)) - 1, int(np.ceil(raw)) + 2):
            cands.add(float(k)); cands.add(k + 0.5)
        for F in sorted(cands):
            s = span / F
            c = (pos - origin) / s
            mis = float((np.minimum(np.abs(c - np.round(c)), 0.3) * wts).sum() / wts.sum())
            pen = 4.0 * abs(s - ppf) / ppf + (0.04 if F != round(F) else 0.0)
            score = mis + pen
            if best is None or score < best[0]:
                best = (score, F)
        out.append(best[1])
    return out[0], out[1]


def detect_walls(rot, ppf, bldg, legend_wall_boxes, W_ft, H_ft):
    Rh, Rv = ridge_maps(rot)
    ref = face_reference(Rh, Rv, bldg)
    lf = legend_wall_features(rot, legend_wall_boxes)
    ox0, oy0, ox1, oy1 = bldg['outer']
    sx = (ox1 - ox0) / W_ft
    sy = (oy1 - oy0) / H_ft
    box = (int(ox0) - 20, int(oy0) - 20, int(ox1) + 20, int(oy1) + 20)
    text = 2.0 / 3.0
    pieces = []
    ext_t = []
    for horiz, R in ((True, Rh), (False, Rv)):
        segs = face_segments(R, horiz, 0.5 * ref, max(int(0.75 * ppf), 10), box)
        P = pair_faces(segs, ppf)
        for p in P:
            if horiz:
                c = (p['pos'] - oy0) / sy; lo = (p['lo'] - ox0) / sx; hi = (p['hi'] - ox0) / sx
                extent = H_ft
            else:
                c = (p['pos'] - ox0) / sx; lo = (p['lo'] - oy0) / sy; hi = (p['hi'] - oy0) / sy
                extent = W_ft
            # skip anything inside the exterior wall band
            if c < text + 0.25 or c > extent - text - 0.25:
                if (abs(c - text / 2) < 0.2 or abs(c - (extent - text / 2)) < 0.2) and hi - lo > 8:
                    ext_t.append(p['t'])
                continue
            typ, score = classify_piece(rot, p, horiz, lf, ppf)
            if typ is None:
                continue
            if score < 0.35:
                thin_ok = p['t'] < 0.5 and score >= 0.1 and hi - lo >= 2.5
                thick_ok = score >= 0.1 and p['s'] >= 0.6 * ref and hi - lo >= 3.0
                if not (thin_ok or thick_ok):
                    continue
            pieces.append({'h': horiz, 'c': c, 'lo': lo, 'hi': hi, 't': p['t'],
                           'type': typ, 'score': score, 's': p['s'], 'px': p})
    # calibrate the building dimensions so that wall centrelines fall on whole
    # feet (guards against a slightly-off graphic scale)
    W_ft, H_ft = calibrate_dims(pieces, bldg, W_ft, H_ft, ppf)
    sx = (ox1 - ox0) / W_ft
    sy = (oy1 - oy0) / H_ft
    for q in pieces:
        p = q['px']
        if q['h']:
            q['c'] = (p['pos'] - oy0) / sy; q['lo'] = (p['lo'] - ox0) / sx; q['hi'] = (p['hi'] - ox0) / sx
        else:
            q['c'] = (p['pos'] - ox0) / sx; q['lo'] = (p['lo'] - oy0) / sy; q['hi'] = (p['hi'] - oy0) / sy
    # P2 and P3 drawn with (nearly) the same poche: tell them apart by thickness
    # relative to the exterior wall (P3 = EXT = 8", P2 = 7.25")
    if ext_t and fill_distance(lf[1], lf[2]) < 1.0:
        t_ext = float(np.median(ext_t))
        for q in pieces:
            if q['type'] in ('P2', 'P3'):
                q['type'] = 'P2' if q['t'] < t_ext - 0.035 else 'P3'
    # resolve conflicts: overlapping pieces with close centrelines -> keep best
    pieces.sort(key=lambda q: -(q['score'] * (q['hi'] - q['lo'])))
    kept = []
    for q in pieces:
        bad = False
        for k in kept:
            if k['h'] != q['h']:
                continue
            if abs(k['c'] - q['c']) < 0.5 * (k['t'] + q['t']) + 0.05:
                ov = min(k['hi'], q['hi']) - max(k['lo'], q['lo'])
                if ov > 0.25:
                    bad = True; break
        if not bad:
            kept.append(q)
    # group into lines and merge
    runs = []
    for horiz in (True, False):
        qs = [q for q in kept if q['h'] == horiz]
        qs.sort(key=lambda q: (_snap(q['c']), q['type'], q['lo']))
        groups = {}
        for q in qs:
            key = (_snap(q['c']), q['type'])
            groups.setdefault(key, []).append(q)
        for (c, typ), g in groups.items():
            g.sort(key=lambda q: q['lo'])
            cur = None
            for q in g:
                if cur is None:
                    cur = {'h': horiz, 'c': c, 'lo': q['lo'], 'hi': q['hi'], 'type': typ,
                           'len_meas': q['hi'] - q['lo'], 'n': 1, 'px_last': q['px']}
                    continue
                gap = q['lo'] - cur['hi']
                if gap <= 1.2 or (gap <= 16.0 and _gap_has_door(rot, Rh if horiz else Rv, cur, q, horiz, ref, bldg, sx, sy)):
                    if q['hi'] > cur['hi']:
                        cur['hi'] = q['hi']; cur['px_last'] = q['px']
                    cur['len_meas'] += q['hi'] - q['lo']; cur['n'] += 1
                else:
                    runs.append(cur)
                    cur = {'h': horiz, 'c': c, 'lo': q['lo'], 'hi': q['hi'], 'type': typ,
                           'len_meas': q['hi'] - q['lo'], 'n': 1, 'px_last': q['px']}
            if cur is not None:
                runs.append(cur)
    # drop tiny runs
    runs = [r for r in runs if r['hi'] - r['lo'] >= 1.0]
    # junction extension
    ext_runs = [
        {'h': True, 'c': text / 2, 'lo': 0, 'hi': W_ft, 'type': 'EXT'},
        {'h': True, 'c': H_ft - text / 2, 'lo': 0, 'hi': W_ft, 'type': 'EXT'},
        {'h': False, 'c': text / 2, 'lo': 0, 'hi': H_ft, 'type': 'EXT'},
        {'h': False, 'c': W_ft - text / 2, 'lo': 0, 'hi': H_ft, 'type': 'EXT'},
    ]
    allr = runs + ext_runs
    for r in runs:
        tr = THICK[r['type']]
        for end in ('lo', 'hi'):
            e = r[end]
            best = None
            for o in allr:
                if o['h'] == r['h'] or o is r:
                    continue
                to = THICK[o['type']]
                if abs(e - o['c']) > to / 2 + 0.7:
                    continue
                if not (o['lo'] - tr / 2 - 0.7 <= r['c'] <= o['hi'] + tr / 2 + 0.7):
                    continue
                d = abs(e - o['c'])
                if best is None or d < best[0]:
                    best = (d, o['c'])
            r[end + '_ext'] = best[1] if best is not None else _snap(e, 0.15)
            r[end + '_met'] = best is not None
    for r in runs:
        r['lo'], r['hi'] = r['lo_ext'], r['hi_ext']
    # walls must connect to the wall network
    runs = [r for r in runs if (r['lo_met'] and r['hi_met'])
            or ((r['lo_met'] or r['hi_met']) and r['hi'] - r['lo'] >= 8.0)]
    return runs, ref, (W_ft, H_ft)


def _gap_has_door(rot, R, a, b, horiz, ref, bldg, sx, sy):
    """Weak face lines across the gap between two collinear pieces => opening."""
    ox0, oy0, ox1, oy1 = bldg['outer']
    pa = a.get('px_last'); pb = b.get('px')
    if pa is None or pb is None:
        return False
    f0 = 0.5 * (pa['f0'] + pb['f0']); f1 = 0.5 * (pa['f1'] + pb['f1'])
    g0 = pa['hi']; g1 = pb['lo']
    if g1 - g0 < 3:
        return True
    i0 = int(np.ceil(g0)); i1 = int(np.floor(g1))
    fr = []
    for f in (f0, f1):
        k = int(round(f))
        if horiz:
            band = R[k - 2:k + 3, i0:i1].max(axis=0)
        else:
            band = R[i0:i1, k - 2:k + 3].max(axis=1)
        fr.append(band > 0.25 * ref)
    both = fr[0] & fr[1]
    return float(both.mean()) > 0.6
