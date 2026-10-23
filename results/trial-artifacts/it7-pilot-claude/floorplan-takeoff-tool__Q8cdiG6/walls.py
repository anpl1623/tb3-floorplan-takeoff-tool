"""Wall detection: paired heavy face lines, classification against legend poche."""
import numpy as np
import cv2


def ridge_mask_rows(Dh, T, prom=0.12, w=4):
    """Local maxima along axis 0 above T with prominence `prom` on both sides (within w px)."""
    from scipy.ndimage import minimum_filter1d
    R = np.zeros(Dh.shape, bool)
    c = Dh[1:-1]
    R[1:-1] = (c > T) & (c >= Dh[:-2]) & (c > Dh[2:])
    mn = minimum_filter1d(Dh, size=w, axis=0, mode='nearest')
    # min over [y-w, y-1] is mn centred at y - w//2 - 1 ... use shifts
    up = np.full(Dh.shape, np.inf, np.float32)
    dn = np.full(Dh.shape, np.inf, np.float32)
    o = w // 2
    # window [y-w, y-1] has centre y - o - (1 if w even else 1)
    up[w:] = mn[o:Dh.shape[0] - w + o]
    dn[:-w] = mn[w + o - (w - 1):Dh.shape[0] - (w - 1) + o][:Dh.shape[0] - w] if False else mn[1 + o:Dh.shape[0] - w + 1 + o]
    R &= (Dh - up >= prom) & (Dh - dn >= prom)
    return R


def pair_bands(R, tmin, tmax):
    """Given ridge mask R (lines along axis 1), find pairs of ridges at vertical
    separation d in [tmin, tmax] with no ridge in between.
    Returns band mask (rows between paired ridges, inclusive) and
    arrays top/bottom sub-structures: top_d[y,x] = d if pair starts at y."""
    H, W = R.shape
    cs = np.cumsum(R, axis=0, dtype=np.int32)
    band = np.zeros((H, W), bool)
    top_d = np.zeros((H, W), np.uint8)
    for d in range(tmin, tmax + 1):
        a = R[:H - d]
        b = R[d:]
        # ridges strictly between y and y+d: cs[y+d-1]-cs[y]
        between = cs[d - 1:H - 1] - cs[:H - d]
        m = a & b & (between == 0)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        top_d[ys, xs] = np.where(top_d[ys, xs] == 0, d, np.minimum(top_d[ys, xs], d))
    ys, xs = np.nonzero(top_d)
    ds = top_d[ys, xs].astype(np.int32)
    # fill band rows
    for k in range(int(ds.max()) + 1 if len(ds) else 0):
        sel = ds >= k
        band[ys[sel] + k, xs[sel]] = True
    return band, top_d


def detect_bands(D, roi, ppf, T, tmin, tmax, blur_ft=0.5):
    """Detect horizontal and vertical wall bands in ROI.
    Returns (bandH, topH, bandV, topV) in ROI coordinates (V arrays transposed back)."""
    x0, y0, x1, y1 = roi
    sub = D[y0:y1, x0:x1]
    L = max(5, int(round(blur_ft * ppf)) | 1)
    Dh = cv2.blur(sub, (L, 1))
    Rh = ridge_mask_rows(Dh, T)
    bandH, topH = pair_bands(Rh, tmin, tmax)
    Dv = cv2.blur(sub, (1, L)).T.copy()
    Rv = ridge_mask_rows(Dv, T)
    bandV, topV = pair_bands(Rv, tmin, tmax)
    return Dh, Rh, bandH, topH, Dv.T, Rv.T, bandV.T, topV.T


def _peak_subpix(p, i):
    if 0 < i < len(p) - 1:
        a, b, c = p[i - 1], p[i], p[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def extract_pieces(band, Dsm, orient, ppf, min_len_px):
    """Band mask (lines along axis 1 in this orientation: rows = across, cols = along).
    For orient 'H' pass band as-is, for 'V' pass transposed arrays.
    Components are split into runs of consistent top/bottom faces.
    Returns list of dicts: c (center across), t (thickness), a, b (extent along), faces."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=4)
    pieces = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w < min_len_px:
            continue
        comp = lab[y:y + h, x:x + w] == i
        colany = comp.any(axis=0)
        tops = np.argmax(comp, axis=0)
        bots = h - 1 - np.argmax(comp[::-1], axis=0)
        cols_all = np.nonzero(colany)[0]
        if len(cols_all) == 0:
            continue
        tops = tops[cols_all]
        bots = bots[cols_all]
        # split into runs of consistent (top, bottom)
        runs = []
        start = 0
        for k in range(1, len(cols_all) + 1):
            brk = (k == len(cols_all) or cols_all[k] - cols_all[k - 1] > 2 or
                   abs(int(tops[k]) - int(np.median(tops[start:k]))) > 2 or
                   abs(int(bots[k]) - int(np.median(bots[start:k]))) > 2)
            if brk:
                runs.append((start, k))
                start = k
        for (r0_, r1_) in runs:
            if r1_ - r0_ < min_len_px:
                continue
            ytop = int(np.median(tops[r0_:r1_])) + y
            ybot = int(np.median(bots[r0_:r1_])) + y
            cols = cols_all[r0_:r1_] + x
            r0 = max(0, ytop - 4)
            r1 = min(Dsm.shape[0], ybot + 5)
            prof = Dsm[r0:r1, cols].mean(axis=1)
            it = ytop - r0
            ib = ybot - r0

            def refine(i0):
                lo = max(0, i0 - 2)
                hi = min(len(prof), i0 + 3)
                j = lo + int(np.argmax(prof[lo:hi]))
                return _peak_subpix(prof, j) + r0
            ft = refine(it)
            fb = refine(ib)
            if fb - ft < 2:
                continue
            i_t = int(np.clip(round(ft - r0), 0, len(prof) - 1))
            i_b = int(np.clip(round(fb - r0), 0, len(prof) - 1))
            pieces.append(dict(orient=orient, f1=ft, f2=fb, c=0.5 * (ft + fb), t=fb - ft,
                               a=float(cols.min()), b=float(cols.max() + 1), ncols=int(len(cols)),
                               peak=float(min(prof[i_t], prof[i_b]))))
    return pieces


def merge_pieces(pieces, ppf, gap_px):
    """Merge collinear pieces (same orientation, c, t) separated by small gaps."""
    out = []
    for orient in ('H', 'V'):
        ps = sorted([p for p in pieces if p['orient'] == orient], key=lambda p: (p['c'], p['a']))
        used = [False] * len(ps)
        for i, p in enumerate(ps):
            if used[i]:
                continue
            grp = [p]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, q in enumerate(ps):
                    if used[j]:
                        continue
                    for g in grp:
                        if abs(q['c'] - g['c']) <= 1.6 and abs(q['t'] - g['t']) <= 1.6:
                            gap = max(q['a'] - g['b'], g['a'] - q['b'])
                            if gap <= gap_px:
                                grp.append(q)
                                used[j] = True
                                changed = True
                                break
            w = np.array([g['b'] - g['a'] for g in grp])
            m = dict(orient=orient,
                     f1=float(np.average([g['f1'] for g in grp], weights=w)),
                     f2=float(np.average([g['f2'] for g in grp], weights=w)),
                     a=min(g['a'] for g in grp), b=max(g['b'] for g in grp),
                     peak=float(np.average([g['peak'] for g in grp], weights=w)),
                     pieces=[(g['a'], g['b']) for g in sorted(grp, key=lambda g: g['a'])])
            m['c'] = 0.5 * (m['f1'] + m['f2'])
            m['t'] = m['f2'] - m['f1']
            out.append(m)
    return out


def seg_rect(s):
    """Return (x0, y0, x1, y1) rectangle of the wall band in ROI coords."""
    if s['orient'] == 'H':
        return (s['a'], s['f1'], s['b'], s['f2'])
    return (s['f1'], s['a'], s['f2'], s['b'])


def strip_features(D, s, a, b, margin):
    """Fill features of the band interior of segment s between along-coords a..b."""
    f1 = int(np.ceil(s['f1'] + margin))
    f2 = int(np.floor(s['f2'] - margin)) + 1
    a = int(a)
    b = int(b)
    if f2 - f1 < 1 or b - a < 3:
        return None
    if s['orient'] == 'H':
        strip = D[f1:f2, a:b]
    else:
        strip = D[a:b, f1:f2].T
    return strip


def region_feats(D, x0, y0, x1, y1):
    """Fill features of D[y0:y1, x0:x1] in original orientation."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    pad = 2
    big = D[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    oy = y0 - max(0, y0 - pad)
    ox = x0 - max(0, x0 - pad)
    g = cv2.GaussianBlur(big, (0, 0), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
    r = D[y0:y1, x0:x1]
    exx = float((gx * gx).mean())
    eyy = float((gy * gy).mean())
    exy = float((gx * gy).mean())
    E = exx + eyy + 1e-6
    return np.array([float(r.mean()), E, exy / E, (exx - eyy) / E, float(r.std())], np.float64)


def extend_to_junctions(segs, tol=3.0, tolE=6.0):
    """Extend each segment end to the centerline of the perpendicular wall it runs into."""
    for s in segs:
        s['ea'] = s['a']
        s['eb'] = s['b']
        s['ja'] = None
        s['jb'] = None
    for s in segs:
        perp = [p for p in segs if p['orient'] != s['orient']]
        for end in ('a', 'b'):
            e = s[end]
            best = None
            for p in perp:
                if abs(e - p['c']) > p['t'] / 2.0 + tolE:
                    continue
                if s['f2'] < p['a'] - tol or s['f1'] > p['b'] + tol:
                    continue
                d = abs(e - p['c'])
                if best is None or d < best[0]:
                    best = (d, p)
            if best is not None:
                s['e' + end] = best[1]['c']
                s['j' + end] = best[1]
    return segs


def _face_evidence(Dsm, s, a, b, thr):
    """Fraction of columns in [a,b) (along) where both face rows have a line above thr.
    Dsm oriented with lines along axis 1 (rows across)."""
    a = int(np.floor(a))
    b = int(np.ceil(b))
    if b <= a:
        return 1.0
    res = []
    for f in (s['f1'], s['f2']):
        r = int(round(f))
        blk = Dsm[max(0, r - 1):r + 2, a:b]
        if blk.size == 0:
            return 0.0
        res.append(blk.max(axis=0) > thr)
    both = res[0] & res[1]
    return float(both.mean())


def bridge_gaps(segs, Dh, Dv, ppf, thr, max_gap_ft=16.0):
    """Merge collinear same-thickness segments across gaps that show continuing face lines."""
    changed = True
    while changed:
        changed = False
        for orient in ('H', 'V'):
            Dsm = Dh if orient == 'H' else Dv.T
            ss = sorted([s for s in segs if s['orient'] == orient], key=lambda s: s['a'])
            for i, s in enumerate(ss):
                for q in ss[i + 1:]:
                    if abs(q['c'] - s['c']) > 1.6 or abs(q['t'] - s['t']) > 1.6:
                        continue
                    gap = q['a'] - s['b']
                    if gap <= 0 or gap > max_gap_ft * ppf:
                        continue
                    # no perpendicular wall crossing the gap
                    blocked = False
                    for p in segs:
                        if p['orient'] == orient:
                            continue
                        if s['b'] - 2 < p['c'] < q['a'] + 2 and p['a'] - 3 <= s['f2'] and p['b'] + 3 >= s['f1']:
                            blocked = True
                            break
                    if blocked:
                        continue
                    ev = _face_evidence(Dsm, s, s['b'], q['a'], thr)
                    if ev < 0.75:
                        continue
                    w1 = s['b'] - s['a']
                    w2 = q['b'] - q['a']
                    s['f1'] = (s['f1'] * w1 + q['f1'] * w2) / (w1 + w2)
                    s['f2'] = (s['f2'] * w1 + q['f2'] * w2) / (w1 + w2)
                    s['c'] = 0.5 * (s['f1'] + s['f2'])
                    s['t'] = s['f2'] - s['f1']
                    s['b'] = q['b']
                    s['pieces'] = s.get('pieces', []) + q.get('pieces', [])
                    segs.remove(q)
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
    return segs


def remove_contained(segs):
    """Drop segments whose band lies mostly inside a longer segment's band."""
    keep = []
    for s in segs:
        r = seg_rect(s)
        area = max(1e-6, (r[2] - r[0]) * (r[3] - r[1]))
        inside = False
        for q in segs:
            if q is s or (q['b'] - q['a']) <= (s['b'] - s['a']):
                continue
            o = seg_rect(q)
            ix = max(0.0, min(r[2], o[2] + 2) - max(r[0], o[0] - 2))
            iy = max(0.0, min(r[3], o[3] + 2) - max(r[1], o[1] - 2))
            if ix * iy > 0.7 * area:
                inside = True
                break
        if not inside:
            keep.append(s)
    return keep


def legend_wall_feats(D, boxes):
    out = []
    for (x, y, w, h) in boxes:
        prof = D[y - 2:y + h + 2, x + 5:x + w - 5].mean(axis=1)
        n = len(prof)
        i1 = int(np.argmax(prof[:n // 2]))
        i2 = n // 2 + int(np.argmax(prof[n // 2:]))
        f1 = y - 2 + _peak_subpix(prof, i1)
        f2 = y - 2 + _peak_subpix(prof, i2)
        t = f2 - f1
        mg = max(2.5, 0.3 * t)
        ft = region_feats(D, x + 6, int(np.ceil(f1 + mg)), x + w - 6, int(np.floor(f2 - mg)) + 1)
        peak = float(min(prof[i1], prof[i2]))
        out.append(dict(t=t, feat=ft, peak=peak))
    return out


def find_walls(D, roi, ppf, legend_walls, T=None):
    """Full wall pipeline. Returns (segs, peakref, Dh, Dv)."""
    x0, y0, x1, y1 = roi
    sub = D[y0:y1, x0:x1]
    if T is None:
        T = 0.62
    tmin = int(np.floor(0.3 * ppf))
    tmax = int(np.ceil(0.85 * ppf))
    Dh, Rh, bandH, topH, Dv, Rv, bandV, topV = detect_bands(D, roi, ppf, T, tmin, tmax)
    pH = extract_pieces(bandH, Dh, 'H', ppf, int(0.3 * ppf))
    pV = extract_pieces(bandV.T, Dv.T, 'V', ppf, int(0.3 * ppf))
    segs = merge_pieces(pH + pV, ppf, gap_px=tmax + 4)
    longs = [q['peak'] for q in segs if q['b'] - q['a'] > 5 * ppf]
    peakref = float(np.median(longs)) if longs else 0.9
    segs = [q for q in segs if q['b'] - q['a'] >= 1.0 * ppf and q['peak'] >= 0.8 * peakref]
    segs = bridge_gaps(segs, Dh, Dv, ppf, 0.5 * peakref)
    segs = remove_contained(segs)
    segs = [q for q in segs if q['b'] - q['a'] >= 1.5 * ppf]
    # fill classification
    for q in segs:
        mg = max(2.5, 0.3 * q['t'])
        f1 = int(np.ceil(q['f1'] + mg))
        f2 = int(np.floor(q['f2'] - mg)) + 1
        feats = []
        step = max(4, int(ppf))
        for a in range(int(q['a']), int(q['b']) - step // 2, step):
            b = min(int(q['b']), a + step)
            if q['orient'] == 'H':
                ft = region_feats(sub, a, f1, b, f2)
            else:
                ft = region_feats(sub, f1, a, f2, b)
            if ft is not None:
                feats.append(ft)
        if not feats:
            q['feat'] = np.zeros(5)
            q['dists'] = [1e9] * len(legend_walls)
            q['cls'] = 3
            continue
        feats = np.array(feats)
        blank = (feats[:, 0] < 0.07) & (feats[:, 4] < 0.06)
        q['feat'] = np.median(feats[~blank], axis=0) if (~blank).sum() else np.median(feats, axis=0)
        ds = []
        for L in legend_walls:
            fl = L['feat']
            fq = q['feat']
            if fl is None:
                ds.append(1e9)
                continue
            d2 = ((fl[0] - fq[0]) / 0.06) ** 2 + ((fl[4] - fq[4]) / 0.05) ** 2 + \
                 ((fl[2] - fq[2]) / 0.25) ** 2 + ((L['t'] - q['t']) / (0.1 * ppf)) ** 2
            ds.append(d2)
        q['dists'] = ds
        q['cls'] = int(np.argmin(ds))
    extend_to_junctions(segs)
    # isolated short pieces (no wall at either end) are text / symbols, not partitions
    segs = [q for q in segs if not (q['ja'] is None and q['jb'] is None and (q['b'] - q['a']) < 8.0 * ppf)]
    extend_to_junctions(segs)
    return segs, peakref, Dh, Dv


def mark_exterior(segs, shape, ppf):
    from scipy import ndimage as ndi
    fp = np.zeros(shape, np.uint8)
    for q in segs:
        x0, y0, x1, y1 = seg_rect(q)
        cv2.rectangle(fp, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, -1)
    k = int(0.85 * ppf * 1.5)
    fpc = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bld = ndi.binary_fill_holes(fpc)
    H, W = shape
    for q in segs:
        if q['orient'] == 'H':
            xs = np.arange(int(q['a']) + 2, max(int(q['a']) + 3, int(q['b']) - 2), 3)
            xs = np.clip(xs, 0, W - 1)
            o1 = bld[max(int(q['f1']) - 4, 0), xs].mean()
            o2 = bld[min(int(q['f2']) + 5, H - 1), xs].mean()
        else:
            ys = np.arange(int(q['a']) + 2, max(int(q['a']) + 3, int(q['b']) - 2), 3)
            ys = np.clip(ys, 0, H - 1)
            o1 = bld[ys, max(int(q['f1']) - 4, 0)].mean()
            o2 = bld[ys, min(int(q['f2']) + 5, W - 1)].mean()
        q['ext'] = bool(min(o1, o2) < 0.5)
    return segs
