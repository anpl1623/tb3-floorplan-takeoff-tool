"""Floor finish regions: segmentation by walls / transition lines and texture classification."""
import numpy as np
import cv2
from scipy import ndimage as ndi

HEAVY_DOG = 48.0


def dog_map(D, wall_mask=None):
    """Difference of Gaussians line map; the background term ignores wall pixels."""
    g1 = cv2.GaussianBlur(D, (0, 0), 1.5)
    if wall_mask is None:
        g2 = cv2.GaussianBlur(D, (0, 0), 6.0)
    else:
        nw = (wall_mask == 0).astype(np.float32)
        num = cv2.GaussianBlur(D * nw, (0, 0), 6.0)
        den = cv2.GaussianBlur(nw, (0, 0), 6.0)
        g2 = num / np.maximum(den, 1e-3)
    return g1 - g2


def segment(D, dog, wall_mask, exterior, ppf, min_region_sf=6.0, heavy_thr=HEAVY_DOG, extra_barrier=None):
    """Label floor regions inside the building.

    Returns label image (0 = not floor) and the interior mask.
    """
    interior = (~exterior) & (wall_mask == 0)
    heavy = (dog > heavy_thr).astype(np.uint8)
    if extra_barrier is not None:
        heavy |= extra_barrier.astype(np.uint8)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    heavy = cv2.morphologyEx(heavy, cv2.MORPH_CLOSE, disk) > 0
    wall_d = cv2.dilate(wall_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    free = interior & (~heavy) & (~wall_d)
    lab, n = ndi.label(free)
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    min_px = min_region_sf * ppf * ppf
    small = areas < min_px
    small[0] = True
    lab[small[lab]] = 0
    # relabel compactly
    keep = np.nonzero(~small)[0]
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    lab = remap[lab]
    # assign remaining interior pixels (lines, text, columns, tiny cells) to nearest region
    unl = interior & (lab == 0)
    if unl.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        filled = lab[iy, ix]
        lab = np.where(interior, filled, 0).astype(np.int32)
    return lab, interior


# Hatch patterns of the CAD standard: fundamental spatial frequencies in cycles per foot,
# (fx, fy) with y pointing down the sheet.  Patterns with two families must show both.
R2 = float(np.sqrt(0.5))
FREQS = {
    'LVT': (0.0, 12.0 / 7.0),             # 7" planks, horizontal joints
    'VCTx': (1.0, 0.0),                   # 12" x 12" grid
    'VCTy': (0.0, 1.0),
    'CT1': (R2 / 0.75, R2 / 0.75),        # 9" x 9" diagonal grid
    'CT2': (-R2 / 0.75, R2 / 0.75),
    'EP': (R2, R2),                       # "/" lines at 12"
    'CPT': (-R2 / 0.5, R2 / 0.5),         # "\" lines at 6"
    'WOM': (0.0, 4.0),                    # 3" ribs
}
PATTERNS = {
    'LVT-1': ('LVT',),
    'VCT-1': ('VCTx', 'VCTy'),
    'CT-1': ('CT1', 'CT2'),
    'EP-1': ('EP',),
    'CPT-1': ('CPT',),
    'WOM-1': ('WOM',),
}
# typical local amplitude (normalised darkness units) of each pattern family
REF_AMP = {'LVT-1': 12.3, 'VCT-1': 7.2, 'CT-1': 9.2, 'EP-1': 7.1, 'CPT-1': 13.7, 'WOM-1': 19.3}
PATTERN_MIN = 0.5
PATTERN_WEAK = 0.38
SPECK_MIN = 0.12          # dots per sf separating sealed (stippled) from polished concrete


def local_amplitudes(D, ppf, valid=None):
    """Coherent local amplitude of each hatch frequency (Gaussian window ~0.9 ft).

    valid: optional mask of pixels to use (normalised convolution ignores the rest).
    """
    sw = 0.9 * ppf
    if valid is None:
        D0 = D - cv2.GaussianBlur(D, (0, 0), 10)
        norm = None
    else:
        V = valid.astype(np.float32)
        vb = cv2.GaussianBlur(V, (0, 0), 10)
        mean = cv2.GaussianBlur(D * V, (0, 0), 10) / np.maximum(vb, 1e-3)
        D0 = (D - mean) * V
        del mean, vb
        norm = np.maximum(cv2.GaussianBlur(V, (0, 0), sw), 1e-3)
    H, W = D.shape
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    maps = {}
    for k, (fx, fy) in FREQS.items():
        fx /= ppf
        fy /= ppf
        ph = (2 * np.pi * fx) * xx + (2 * np.pi * fy) * yy
        c = np.cos(ph).astype(np.float32)
        s = np.sin(ph).astype(np.float32)
        del ph
        zr = cv2.GaussianBlur(D0 * c, (0, 0), sw)
        zi = cv2.GaussianBlur(D0 * s, (0, 0), sw)
        a = 2.0 * np.sqrt(zr * zr + zi * zi)
        if norm is not None:
            a /= norm
        maps[k] = a
        del c, s, zr, zi
    support = None if norm is None else norm
    return maps, sw, support


def region_amplitudes(lab, maps, sw, valid=None, support=None, erode_ft_px=None):
    n = int(lab.max())
    out = [None] * (n + 1)
    ex = int(0.6 * sw) if erode_ft_px is None else int(erode_ft_px)
    objs = ndi.find_objects(lab)
    for r in range(1, n + 1):
        sl = objs[r - 1]
        if sl is None:
            continue
        m = (lab[sl] == r).astype(np.uint8)
        er = cv2.erode(m, np.ones((2 * ex + 1, 2 * ex + 1), np.uint8)) > 0
        sel = er.copy()
        if valid is not None:
            sel &= valid[sl]
        if support is not None:
            sel &= support[sl] > 0.5
        if sel.sum() < 50:
            sel = er if er.sum() >= 50 else (m > 0)
        out[r] = {k: float(np.median(v[sl][sel])) for k, v in maps.items()}
    return out


def classify_amplitudes(amp, specks):
    """Finish code of a region from its hatch amplitudes (and stipple density if unhatched).

    A hatch is accepted when it reaches PATTERN_MIN of its reference amplitude, or a somewhat
    weaker response (heavier blur damps the fine patterns) that still clearly beats every
    other pattern.
    """
    scores = {}
    for name, fams in PATTERNS.items():
        scores[name] = min(amp[f] for f in fams) / REF_AMP[name]
    ranked = sorted(scores.values(), reverse=True)
    best = max(scores, key=scores.get)
    if scores[best] >= PATTERN_MIN or (scores[best] >= PATTERN_WEAK and ranked[0] >= 1.8 * ranked[1]):
        return best, scores
    return ('SC-1' if specks > SPECK_MIN else 'PC-1'), scores


def speck_density(D, lab, ppf, valid=None):
    """Isolated dots per square foot for each region (stipple of sealed concrete).

    Dots are local maxima of a small-scale blob response whose Hessian is curved in both
    directions (points on lines are rejected); the threshold follows the scan noise.
    """
    b = cv2.GaussianBlur(D, (0, 0), 1.0) - cv2.GaussianBlur(D, (0, 0), 3.0)
    inside = lab > 0
    if valid is not None:
        inside &= valid
    vals = b[inside]
    if vals.size < 100:
        return np.zeros(int(lab.max()) + 1)
    med = float(np.median(vals))
    noise = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-3
    thr = max(6.0 * noise, 5.0)
    mx = cv2.dilate(b, np.ones((5, 5), np.uint8))
    pk = (b >= mx) & (b > thr) & inside
    ys, xs = np.nonzero(pk)
    G = cv2.GaussianBlur(D, (0, 0), 1.5)
    dxx = cv2.Sobel(G, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(G, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(G, cv2.CV_32F, 1, 1, ksize=3)
    a = dxx[ys, xs]; c = dyy[ys, xs]; bb = dxy[ys, xs]
    tr = a + c
    det_ = np.sqrt(np.maximum((a - c) ** 2 + 4 * bb * bb, 0))
    l1 = 0.5 * (tr - det_)          # most negative curvature
    l2 = 0.5 * (tr + det_)
    blob = (l1 < 0) & (l2 < 0) & (np.abs(l2) > 0.3 * np.abs(l1))
    nreg = int(lab.max())
    cnt = np.bincount(lab[ys[blob], xs[blob]], minlength=nreg + 1).astype(np.float64)
    if valid is not None:
        area = np.bincount(lab[inside].ravel(), minlength=nreg + 1) / (ppf * ppf)
    else:
        area = np.bincount(lab.ravel(), minlength=nreg + 1) / (ppf * ppf)
    return cnt / np.maximum(area, 1e-6)


def calibrate_heavy(dog, runs, idxs, off, ppf):
    """DoG response of exterior wall outer faces (heavy pen reference)."""
    x0, y0 = off
    H, W = dog.shape
    vals = []
    for i in idxs:
        r = runs[i]
        if r.get('type') != 'EXT' or not r.get('ext_side'):
            continue
        face = r['p1'] if r['ext_side'] < 0 else r['p2']
        lo = int(max(r['a'], r['sa'])) + 20
        hi = int(min(r['b'], r['sb'])) - 20
        for q in range(lo, hi, 4):
            if r['axis'] == 0:
                yy, xx = int(round(face)) - y0, q - x0
                if 2 <= yy < H - 2 and 0 <= xx < W:
                    vals.append(float(dog[yy - 2:yy + 3, xx].max()))
            else:
                yy, xx = q - y0, int(round(face)) - x0
                if 0 <= yy < H and 2 <= xx < W - 2:
                    vals.append(float(dog[yy, xx - 2:xx + 3].max()))
    if len(vals) < 20:
        return None
    return float(np.median(vals))


def render_segments(shape, segs, ext=8, width=3):
    """Rasterise axis-aligned line segments (extended at both ends) into a mask."""
    m = np.zeros(shape, np.uint8)
    H, W = shape
    h = width // 2
    for axis, lst in segs.items():
        lim_across, lim_along = (H, W) if axis == 0 else (W, H)
        for s in lst:
            p = int(round(s['pos']))
            if p - h < 0 or p + h >= lim_across:
                continue
            a = max(0, int(s['a']) - ext)
            b = min(lim_along - 1, int(s['b']) + ext)
            if b < a:
                continue
            if axis == 0:
                m[p - h:p + h + 1, a:b + 1] = 1
            else:
                m[a:b + 1, p - h:p + h + 1] = 1
    return m


def merge_islands(lab, wall_mask, amps, ppf, max_sf=60.0, specks=None):
    """Merge small, blank regions enclosed by a single other region (fixtures drawn heavy).

    Islands showing a hatch or the stipple of sealed concrete are real finish areas and stay.
    """
    n = int(lab.max())
    areas = np.bincount(lab.ravel(), minlength=n + 1) / (ppf * ppf)
    objs = ndi.find_objects(lab)
    remap = np.arange(n + 1)
    wall_d = cv2.dilate(wall_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for r in range(1, n + 1):
        if areas[r] >= max_sf or amps[r] is None or objs[r - 1] is None:
            continue
        scores = [min(amps[r][f] for f in fams) / REF_AMP[name] for name, fams in PATTERNS.items()]
        if max(scores) >= PATTERN_MIN:
            continue
        if specks is not None and r < len(specks) and specks[r] > SPECK_MIN:
            continue
        sl = objs[r - 1]
        y0 = max(0, sl[0].start - 6); y1 = min(lab.shape[0], sl[0].stop + 6)
        x0 = max(0, sl[1].start - 6); x1 = min(lab.shape[1], sl[1].stop + 6)
        sub = lab[y0:y1, x0:x1]
        m = (sub == r).astype(np.uint8)
        ring = (cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0) & (m == 0)
        if (ring & wall_d[y0:y1, x0:x1]).any():
            continue
        nb = sub[ring]
        nb = nb[(nb != 0) & (nb != r)]
        if len(nb) == 0:
            continue
        vals, cnt = np.unique(nb, return_counts=True)
        if cnt.max() < 0.97 * cnt.sum():
            continue
        remap[r] = vals[np.argmax(cnt)]
    # resolve chains
    for r in range(1, n + 1):
        t = r
        seen = 0
        while remap[t] != t and seen < 10:
            t = remap[t]
            seen += 1
        remap[r] = t
    return remap[lab], remap
