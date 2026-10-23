"""Floor finish regions: segmentation by walls and transition lines, texture classification."""
import numpy as np
import cv2

SHIFT = 4
SC = float(1 << SHIFT)


def _rect(mask, x0, y0, x1, y1, val=1):
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    cv2.fillPoly(mask, [np.round(pts * SC).astype(np.int32)], val, lineType=cv2.LINE_8, shift=SHIFT)


def wall_extent(w):
    lo = min(w['a0'], w['e0']) if w.get('j0') is not None else w['a0']
    hi = max(w['a1'], w['e1']) if w.get('j1') is not None else w['a1']
    return lo, hi


# Wall face lines read ~0.02 ft proud of the faces used by the takeoff (calibrated on
# the example sheets); the wall mask used for floor areas is trimmed by this much.
FACE_TRIM_FT = 0.0204


def draw_walls(shape, walls, only_ext=False, grow=0.0):
    m = np.zeros(shape, np.uint8)
    for w in walls:
        if only_ext and not w.get('ext'):
            continue
        lo, hi = wall_extent(w)
        t0 = w['t0'] - grow; t1 = w['t1'] + grow
        # pixel centres: pixel i covers [i-0.5, i+0.5]; fillPoly fills centres inside
        if w['o'] == 'H':
            _rect(m, lo - 0.5, t0 - 0.5, hi + 0.5, t1 - 0.5)
        else:
            _rect(m, t0 - 0.5, lo - 0.5, t1 - 0.5, hi + 0.5)
    return m


def interior_mask(shape, walls, grow=0.0):
    ext = draw_walls(shape, walls, only_ext=True, grow=grow)
    ff = ext.copy()
    pad = np.zeros((shape[0] + 2, shape[1] + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 2)
    inside = (ff == 0)
    # guard against a leaking exterior loop (a missed piece of exterior wall)
    exts = [w for w in walls if w.get('ext')]
    if exts:
        xs = []; ys = []
        for w in exts:
            lo, hi = wall_extent(w)
            if w['o'] == 'H':
                xs += [lo, hi]; ys += [w['t0'], w['t1']]
            else:
                ys += [lo, hi]; xs += [w['t0'], w['t1']]
        bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if inside.sum() < 0.4 * bbox_area:
            k = int(max(15, 0.05 * min(shape))) | 1
            closed = cv2.morphologyEx(ext, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            ff = closed.copy()
            pad = np.zeros((shape[0] + 2, shape[1] + 2), np.uint8)
            cv2.floodFill(ff, pad, (0, 0), 2)
            inside = (ff == 0)
    return inside


def transition_mask(Gb, thr, inside, wallmask, s):
    """Heavy-line pixels inside the building that are connected into long structures
    (finish transition lines, and the halo of wall face lines)."""
    m = ((Gb < thr) & inside).astype(np.uint8)
    # connect to walls so that lines ending at wall faces join the wall network
    near = cv2.dilate(wallmask, np.ones((3, 3), np.uint8))
    m2 = np.maximum(m, near)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m2, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) >= 2.6 * s:
            keep[i] = True
    return keep[lab] & (m > 0)


def _segments(mask, horizontal, min_len):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1) if horizontal else (1, min_len))
    op = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, st, cen = cv2.connectedComponentsWithStats(op, 8)
    segs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if horizontal and h <= 8:
            segs.append((x, x + w - 1, cen[i][1]))
        elif not horizontal and w <= 8:
            segs.append((y, y + h - 1, cen[i][0]))
    return segs


def bridge_hidden_transitions(tm, G, wm, bg, s, max_gap_ft=16.0):
    """Re-connect straight transition lines interrupted by a fixture drawn over them.
    A gap is bridged only if it is white (fixture fill) and a fixture edge crosses
    the line where it stops."""
    out = tm.copy()
    m8 = tm.astype(np.uint8)
    max_gap = int(max_gap_ft * s)
    for horizontal in (True, False):
        img = G if horizontal else G.T
        wmo = wm if horizontal else wm.T
        segs = _segments(m8 if horizontal else np.ascontiguousarray(m8.T), True, int(1.0 * s))
        segs.sort(key=lambda t: (round(t[2]), t[0]))
        H, W = img.shape

        def edge_at(x, yc, direction):
            # fixture edge (line across the transition) just beyond the end of the line
            ys = np.r_[np.arange(int(yc) - 9, int(yc) - 2), np.arange(int(yc) + 3, int(yc) + 10)]
            ys = ys[(ys >= 0) & (ys < H)]
            xs = np.arange(x, x + 6 * direction, direction)
            xs = xs[(xs >= 0) & (xs < W)]
            if len(ys) == 0 or len(xs) == 0:
                return False
            prof = img[np.ix_(ys, xs)].mean(0)
            return prof.min() < bg - 25

        def white_gap(xa, xb, yc):
            if xb - xa < 2:
                return True
            yy = slice(max(0, int(round(yc)) - 2), int(round(yc)) + 3)
            seg = img[yy, xa:xb]
            return seg.size > 0 and seg.mean() > bg - 25 and not wmo[yy, xa:xb].any()

        for i, (a0, a1, yc) in enumerate(segs):
            # look right of a1
            best = None
            for (b0, b1, yb) in segs:
                if abs(yb - yc) > 2.5 or b0 <= a1 + 2 or b0 - a1 > max_gap:
                    continue
                if best is None or b0 < best[0]:
                    best = (b0, yb)
            target = None
            if best is not None:
                target = best[0]
            else:
                # run into a wall?
                y = int(round(yc))
                if 0 <= y < H:
                    row = wmo[y, a1 + 1:min(W, a1 + 1 + max_gap)]
                    hit = np.nonzero(row)[0]
                    if len(hit):
                        target = a1 + 1 + int(hit[0])
            if target is None or target - a1 <= 3:
                continue
            if not edge_at(a1 + 1, yc, 1):
                continue
            if not white_gap(a1 + 3, target - 2, yc):
                continue
            y = int(round(yc))
            if horizontal:
                out[max(0, y - 1):y + 2, a1:target + 1] = True
            else:
                out[a1:target + 1, max(0, y - 1):y + 2] = True
        # and the mirror case: look left of a0 (into a wall or segment)
        for i, (a0, a1, yc) in enumerate(segs):
            best = None
            for (b0, b1, yb) in segs:
                if abs(yb - yc) > 2.5 or b1 >= a0 - 2 or a0 - b1 > max_gap:
                    continue
                if best is None or b1 > best[0]:
                    best = (b1, yb)
            target = None
            if best is not None:
                target = best[0]
            else:
                y = int(round(yc))
                if 0 <= y < H:
                    lo = max(0, a0 - max_gap)
                    row = wmo[y, lo:a0]
                    hit = np.nonzero(row)[0]
                    if len(hit):
                        target = lo + int(hit[-1])
            if target is None or a0 - target <= 3:
                continue
            if not edge_at(a0 - 1, yc, -1):
                continue
            if not white_gap(target + 2, a0 - 3, yc):
                continue
            y = int(round(yc))
            if horizontal:
                out[max(0, y - 1):y + 2, target:a0 + 1] = True
            else:
                out[target:a0 + 1, max(0, y - 1):y + 2] = True
    return out


def assign_barrier(lab, barrier):
    """Give barrier pixels the label of the nearest labelled floor pixel."""
    src = (lab == 0).astype(np.uint8)
    dist, idx = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    del dist
    # DIST_LABEL_PIXEL numbers the zero pixels of src (= labelled floor) 1..N in raster order
    lut = np.zeros(int(idx.max()) + 1, np.int32)
    vals = lab[src == 0]
    lut[1:len(vals) + 1] = vals
    del vals
    sel = barrier & (lab == 0)
    out = lab
    out[sel] = lut[idx[sel]]
    return out


# Prototype texture signatures: (energy, [vertical-line, diag '/', horizontal-line, diag '\\'])
# orientation bins are gradient directions 0/45/90/135 deg.
PROTOTYPES = [
    ('LVT-1', 32.3, (0.15, 0.05, 0.76, 0.05)),
    ('LVT-1', 32.3, (0.76, 0.05, 0.15, 0.05)),   # planks running the other way
    ('PC-1', 6.3, (0.53, 0.06, 0.36, 0.06)),
    ('PC-1', 6.3, (0.36, 0.06, 0.53, 0.06)),
    ('SC-1', 7.0, (0.27, 0.25, 0.25, 0.23)),
    ('EP-1', 17.6, (0.04, 0.88, 0.06, 0.02)),
    ('VCT-1', 32.0, (0.40, 0.09, 0.42, 0.09)),
    ('CT-1', 38.8, (0.14, 0.36, 0.15, 0.36)),
    ('CPT-1', 34.5, (0.05, 0.01, 0.05, 0.89)),
    ('WOM-1', 54.0, (0.02, 0.02, 0.94, 0.02)),
    ('WOM-1', 54.0, (0.94, 0.02, 0.02, 0.02)),
]


def gradient_bins(G, strip=600):
    """Thresholded gradient magnitude and 4-bin orientation, computed in strips."""
    H, W = G.shape
    magt = np.zeros((H, W), np.float32)
    bins = np.zeros((H, W), np.uint8)
    pad = 6
    for y0 in range(0, H, strip):
        a = max(0, y0 - pad); b = min(H, y0 + strip + pad)
        Gs = cv2.GaussianBlur(G[a:b], (0, 0), 1.0)
        gx = cv2.Sobel(Gs, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Gs, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
        bb = (((ang + 22.5) // 45).astype(np.int32) % 4).astype(np.uint8)
        y1 = min(H, y0 + strip)
        sl = slice(y0 - a, y1 - a)
        m = mag[sl]
        magt[y0:y1] = np.where(m > 20, m, 0)
        bins[y0:y1] = bb[sl]
    return magt, bins


def region_features(lab, n, magt, bins, heavy, erode_px=13):
    """Per-label texture energy and orientation histogram (eroded, heavy ink excluded)."""
    feats = {}
    k = np.ones((erode_px, erode_px), np.uint8)
    for i in range(1, n):
        m = (lab == i)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = m[y0:y1, x0:x1].astype(np.uint8)
        me = cv2.erode(sub, k) > 0
        me &= ~heavy[y0:y1, x0:x1]
        cnt = int(me.sum())
        if cnt < 200:
            me = sub > 0
            cnt = int(me.sum())
        mt = magt[y0:y1, x0:x1][me]
        bb = bins[y0:y1, x0:x1][me]
        e = np.array([mt[bb == b].sum() for b in range(4)], np.float64)
        tot = e.sum()
        ori = (e / tot) if tot > 0 else np.full(4, 0.25)
        feats[i] = dict(energy=float(tot / max(cnt, 1)), ori=tuple(ori), npx=cnt, box=(y0, y1, x0, x1), mask=me)
    return feats


def line_period(G, m, axis, maxlag):
    """Lag (px) of the first clear autocorrelation peak of the masked, high-passed
    texture along axis (perpendicular to the hatch lines)."""
    if m.sum() < 400 or maxlag < 6:
        return None
    x = G.astype(np.float32)
    x = (x - cv2.GaussianBlur(x, (0, 0), 4.0)).astype(np.float64)
    xm = np.where(m, x - x[m].mean(), 0.0)
    mm = m.astype(np.float64)
    n = x.shape[axis]
    N = 1
    while N < n + maxlag + 1:
        N *= 2
    Fx = np.fft.rfft(xm, N, axis=axis)
    Fm = np.fft.rfft(mm, N, axis=axis)
    A = np.fft.irfft(Fx * np.conj(Fx), N, axis=axis)
    Mm = np.fft.irfft(Fm * np.conj(Fm), N, axis=axis)
    sl = [slice(None), slice(None)]
    sl[axis] = slice(0, maxlag + 2)
    A = A[tuple(sl)].sum(axis=1 - axis)
    Mm = Mm[tuple(sl)].sum(axis=1 - axis)
    ac = A / np.maximum(Mm, 1.0)
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo = 3
    peaks = [i for i in range(lo, maxlag + 1) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]]
    if not peaks:
        return None
    top = max(ac[i] for i in peaks)
    if top < 0.08:
        return None
    for i in peaks:
        if ac[i] >= 0.5 * top:
            return i
    return None


def classify_texture(ft):
    E = max(ft['energy'], 0.5)
    if E < 2.5:
        return 'PC-1', 0.0
    best = None
    for code, Ep, op in PROTOTYPES:
        d = abs(np.log(E / Ep)) / np.log(1.6) + sum(abs(a - b) for a, b in zip(ft['ori'], op)) / 0.3
        if best is None or d < best[1]:
            best = (code, d)
    return best


def segment(G, s, bg, walls):
    """Return (label image, n) of floor regions bounded by walls and heavy transition lines."""
    shape = G.shape
    trim = -FACE_TRIM_FT * s
    wm = draw_walls(shape, walls, grow=trim)
    inside = interior_mask(shape, walls, grow=trim) & (wm == 0)
    Gb = cv2.GaussianBlur(G, (0, 0), 1.5)
    vals = []
    for w in walls:
        if not w.get('ext'):
            continue
        for a in np.linspace(w['a0'] + 20, w['a1'] - 20, 40).astype(int):
            for tf in (w['t0'], w['t1']):
                c = int(round(tf))
                if w['o'] == 'H':
                    vals.append(Gb[c - 2:c + 3, a].min())
                else:
                    vals.append(Gb[a, c - 2:c + 3].min())
    wd = float(np.median(vals)) if vals else 115.0
    thr = bg - 0.65 * (bg - wd)
    tm = transition_mask(Gb, thr, inside, wm, s)
    tm = bridge_hidden_transitions(tm, G, wm > 0, bg, s) & inside
    floor = (inside & ~tm).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(floor, 4)
    small = st[:, 4] < 0.5 * s * s
    small[0] = False
    lab[small[lab]] = 0
    lab = assign_barrier(lab, inside & (lab == 0))
    heavy = cv2.dilate((Gb < thr).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return lab, n, heavy, dict(thr=thr, wall_dark=wd)


def _ranked(ft):
    E = max(ft['energy'], 0.5)
    ds = {}
    for code, Ep, op in PROTOTYPES:
        d = abs(np.log(E / Ep)) / np.log(1.6) + sum(abs(a - b) for a, b in zip(ft['ori'], op)) / 0.3
        ds[code] = min(ds.get(code, 1e9), d)
    return sorted(ds.items(), key=lambda kv: kv[1])


def floor_areas(G, s, bg, walls, codes, exclude=None):
    lab, n, heavy, info = segment(G, s, bg, walls)
    if exclude is not None:
        heavy = heavy | exclude
    magt, bins = gradient_bins(G)
    feats = region_features(lab, n, magt, bins, heavy)
    areas = np.bincount(lab.ravel(), minlength=n).astype(np.float64) / (s * s)
    out = {c: 0.0 for c in codes}
    regions = []
    for i, ft in feats.items():
        code, dist = classify_texture(ft)
        if code == 'CT-1':
            # diagonal 9" grid: strong period of ~1.06 ft along the axes
            y0, y1, x0, x1 = ft['box']
            per = line_period(G[y0:y1, x0:x1], ft['mask'], 1, int(1.6 * s))
            ft['period_ft'] = (per / s) if per else None
            if not per or not (0.8 <= per / s <= 1.35):
                rk = [c for c in _ranked(ft) if c[0] != 'CT-1']
                code, dist = rk[0]
        if code in ('LVT-1', 'WOM-1'):
            # plank pitch (7") vs walk-off mat ribs: decide by line period, robust to blur
            y0, y1, x0, x1 = ft['box']
            axis = 0 if ft['ori'][2] >= ft['ori'][0] else 1
            per = line_period(G[y0:y1, x0:x1], ft['mask'], axis, int(0.8 * s))
            ft['period_ft'] = (per / s) if per else None
            if per:
                code = 'WOM-1' if per / s < 0.38 else 'LVT-1'
        ft.pop('mask', None)
        regions.append(dict(label=i, area=float(areas[i]), code=code, dist=float(dist), **ft))
        if code in out:
            out[code] += float(areas[i])
        else:
            out[code] = out.get(code, 0.0) + float(areas[i])
    return out, regions, lab
