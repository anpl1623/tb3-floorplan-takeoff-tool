"""Floor finish zones: segmentation by walls and transition lines, texture classification."""
import numpy as np
import cv2
from walls import wall_rect


def wall_mask(shape, walls, ppf, grow=0.0):
    M = np.zeros(shape, np.uint8)
    for w in walls:
        lo = min(w['lo'], w.get('a', w['lo']))
        hi = max(w['hi'], w.get('b', w['hi']))
        h = w['t'] / 2 + grow
        if w['o'] == 'H':
            x0, x1, y0, y1 = lo - h * 0, hi, w['c'] - h, w['c'] + h
            # extend along by half thickness at ends to cover corners
            x0 -= w['t'] / 2; x1 += w['t'] / 2
        else:
            y0, y1, x0, x1 = lo, hi, w['c'] - h, w['c'] + h
            y0 -= w['t'] / 2; y1 += w['t'] / 2
        # pixel centres inside [x0,x1]x[y0,y1]
        ix0, ix1 = int(np.ceil(x0 - 0.5)), int(np.floor(x1 - 0.5))
        iy0, iy1 = int(np.ceil(y0 - 0.5)), int(np.floor(y1 - 0.5))
        M[max(0, iy0):iy1 + 1, max(0, ix0):ix1 + 1] = 1
    return M


def building_interior(shape, walls, ppf=25.0):
    """Mask of pixels inside the exterior walls (flood fill from outside, small gaps closed)."""
    ext = [w for w in walls if w.get('ext')]
    f = 4
    h, w_ = shape[0] // f + 1, shape[1] // f + 1
    M = np.zeros((h, w_), np.uint8)
    for w in ext:
        lo = min(w['lo'], w.get('a', w['lo'])) - w['t'] / 2
        hi = max(w['hi'], w.get('b', w['hi'])) + w['t'] / 2
        if w['o'] == 'H':
            x0, x1, y0, y1 = lo, hi, w['c'] - w['t'] / 2, w['c'] + w['t'] / 2
        else:
            y0, y1, x0, x1 = lo, hi, w['c'] - w['t'] / 2, w['c'] + w['t'] / 2
        cv2.rectangle(M, (int(x0 / f), int(y0 / f)), (int(np.ceil(x1 / f)), int(np.ceil(y1 / f))), 1, -1)
    best = None
    for gap_ft in (1.0, 3.0, 8.0, 16.0):
        k = max(3, int(round(gap_ft * ppf / f)) | 1)
        Mc = cv2.morphologyEx(M, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        inv = (1 - Mc).astype(np.uint8)
        n, lab = cv2.connectedComponents(inv, connectivity=4)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        inside = ~np.isin(lab, list(border)) & (Mc == 0)
        if inside.sum() > 0.2 * (M.shape[0] * M.shape[1]) * 0.0 + 100:
            best = inside
            break
    if best is None:
        return np.zeros(shape, bool)
    big = cv2.resize(best.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    # refine at full resolution: inside = not exterior-wall and region connected to coarse interior
    Mf = np.zeros(shape, np.uint8)
    for w in ext:
        lo = min(w['lo'], w.get('a', w['lo'])) - w['t'] / 2
        hi = max(w['hi'], w.get('b', w['hi'])) + w['t'] / 2
        if w['o'] == 'H':
            x0, x1, y0, y1 = lo, hi, w['c'] - w['t'] / 2, w['c'] + w['t'] / 2
        else:
            y0, y1, x0, x1 = lo, hi, w['c'] - w['t'] / 2, w['c'] + w['t'] / 2
        ix0, ix1 = int(np.ceil(x0 - 0.5)), int(np.floor(x1 - 0.5))
        iy0, iy1 = int(np.ceil(y0 - 0.5)), int(np.floor(y1 - 0.5))
        Mf[max(0, iy0):iy1 + 1, max(0, ix0):ix1 + 1] = 1
    # grow the coarse interior up to the exterior wall faces
    grown = cv2.dilate(big.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & (Mf == 0)
    return grown | (big & (Mf == 0))


PER_EDGES = np.array([2.0, 3.0, 4.5, 6.5, 9.5, 14.0, 20.0, 30.0, 45.0, 70.0, 1e9])
NANG = 8
_GEOM_CACHE = {}


def _geom(h, w):
    key = (h, w)
    if key not in _GEOM_CACHE:
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        r = np.sqrt(fx ** 2 + fy ** 2)
        per = np.where(r > 0, 1.0 / np.maximum(r, 1e-9), 1e9)
        ang = np.arctan2(fy, fx) % np.pi
        rb = np.digitize(per, PER_EDGES) - 1
        ab = np.minimum((ang / np.pi * NANG + 0.5).astype(int) % NANG, NANG - 1)
        valid = (rb >= 0) & (rb < len(PER_EDGES) - 1) & (r > 0)
        idx = (rb * NANG + ab)
        idx = np.where(valid, idx, -1).ravel()
        win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        _GEOM_CACHE[key] = (idx, win)
    return _GEOM_CACHE[key]


def spec_desc(p):
    p = p.astype(np.float32)
    h, w = p.shape
    idx, win = _geom(h, w)
    m = float(p.mean())
    q = p - m
    F = np.abs(np.fft.fft2(q * win)) ** 2
    Fv = F.ravel()
    ok = idx >= 0
    H = np.bincount(idx[ok], weights=Fv[ok], minlength=(len(PER_EDGES) - 1) * NANG)
    tot = H.sum()
    var = float((q * q).mean())
    return dict(m=m, var=var, H=np.sqrt(H / max(tot, 1e-12)))


def spec_dist(a, b):
    dm = (a['m'] - b['m']) / 0.03
    dv = (np.log(a['var'] + 1e-4) - np.log(b['var'] + 1e-4)) / 0.7
    dh = np.sqrt(((a['H'] - b['H']) ** 2).sum()) / 0.25
    return dm * dm + dv * dv + dh * dh


BASE_PATCH_SIZES = [(70, 150), (60, 90), (48, 48), (32, 32)]
BASE_WEIGHTS = [1.0, 1.0, 0.7, 0.4]
PATCH_SIZES = list(BASE_PATCH_SIZES)
SIZE_WEIGHT = dict(zip(PATCH_SIZES, BASE_WEIGHTS))


def set_patch_scale(ppf):
    """Patch sizes scale with the drawing scale (nominal 24.5 px/ft)."""
    global PATCH_SIZES, SIZE_WEIGHT
    s = float(np.clip(ppf / 24.5, 0.5, 2.0))
    PATCH_SIZES = [(max(16, int(round(h * s))), max(16, int(round(w * s)))) for h, w in BASE_PATCH_SIZES]
    SIZE_WEIGHT = dict(zip(PATCH_SIZES, BASE_WEIGHTS))


def swatch_descriptors(dk, swatches, ppf):
    """Descriptors of sub-patches of each floor legend swatch, for each patch size."""
    out = {}
    for (ph, pw) in PATCH_SIZES:
        per = []
        for (x, y, w, h) in swatches:
            inner = dk[y + 5:y + h - 5, x + 5:x + w - 5]
            ds = []
            if inner.shape[0] < ph or inner.shape[1] < pw:
                per.append(ds)
                continue
            sy = max(3, (inner.shape[0] - ph) // 4)
            sx = max(3, (inner.shape[1] - pw) // 6)
            for oy in range(0, inner.shape[0] - ph + 1, sy):
                for ox in range(0, inner.shape[1] - pw + 1, sx):
                    ds.append(spec_desc(inner[oy:oy + ph, ox:ox + pw]))
            per.append(ds)
        out[(ph, pw)] = per
    return out


def ink_distance(lm, th=0.08, dk=None, fill_th=0.15):
    """Distance (px) to the nearest ink pixel (thin lines of the line map, or any fill)."""
    ink = (lm > th)
    if dk is not None:
        ink = ink | (dk > fill_th)
    ink = ink.astype(np.uint8)
    return cv2.distanceTransform((1 - ink).astype(np.uint8), cv2.DIST_L2, 5)


def swatch_blank_radius(lm, swatches, th=0.08, dk=None):
    out = []
    for (x, y, w, h) in swatches:
        sub = lm[y:y + h, x:x + w].copy()
        if dk is not None:
            sub = np.maximum(sub, np.where(dk[y:y + h, x:x + w] > 0.15, 1.0, 0.0).astype(sub.dtype))
        # treat the swatch frame as ink so distances are bounded by the frame
        sub[:5, :] = 1; sub[-5:, :] = 1; sub[:, :5] = 1; sub[:, -5:] = 1
        dt = ink_distance(sub, th)[5:-5, 5:-5]
        out.append(float(np.percentile(dt, 90)))
    return np.array(out)


def classify_zone(dk, zone_mask, okmask_ii, sdesc, rng, n_target=40, max_tries=1500,
                  inkdist=None, blank_r=None):
    """Weighted votes over random patches of several sizes fully inside the zone.

    Returns (weighted votes, n used, per-size votes)."""
    ys, xs = np.nonzero(zone_mask)
    if len(ys) == 0:
        return None, 0, None
    nclass = len(next(iter(sdesc.values())))
    total = np.zeros(nclass)
    used_all = 0
    per = {}
    for (ph, pw) in PATCH_SIZES:
        sd = sdesc[(ph, pw)]
        if any(len(s) == 0 for s in sd):
            continue
        votes = np.zeros(nclass)
        used = 0
        for _ in range(max_tries):
            k = rng.integers(len(ys))
            a, b = ys[k] - ph // 2, xs[k] - pw // 2
            if a < 0 or b < 0 or a + ph >= dk.shape[0] or b + pw >= dk.shape[1]:
                continue
            c = okmask_ii[a + ph, b + pw] - okmask_ii[a, b + pw] - okmask_ii[a + ph, b] + okmask_ii[a, b]
            if c < 0.97 * ph * pw:
                continue
            d = spec_desc(dk[a:a + ph, b:b + pw])
            dists = np.array([min(spec_dist(d, s) for s in ss) for ss in sd])
            if inkdist is not None and blank_r is not None:
                bmax = float(np.percentile(inkdist[a:a + ph:2, b:b + pw:2], 90))
                incompatible = 1.5 * blank_r + 2.0 < bmax
                if incompatible.all():
                    continue
                dists = np.where(incompatible, dists + 1e3, dists)
            votes[int(np.argmin(dists))] += 1
            used += 1
            if used >= n_target:
                break
        per[(ph, pw)] = votes
        if used >= 6:
            total += SIZE_WEIGHT[(ph, pw)] * votes / used * min(1.0, used / 20.0)
            used_all += used
    if used_all == 0:
        return None, 0, per
    return total, used_all, per
