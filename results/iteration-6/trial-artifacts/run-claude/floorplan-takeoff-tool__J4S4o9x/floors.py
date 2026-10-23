"""Floor finish regions: segmentation inside wall faces, texture classification."""
import numpy as np
import cv2
from scipy import ndimage as ndi

from canon import PPF
import params

FINISHES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']

# hatch signatures in canonical pixels: (direction, period)
#   h: horizontal lines (profile along y), v: vertical lines,
#   d1: '/' lines (x+y const), d2: '\' lines (x-y const)
SIGS = {
    'LVT-1': [('h', 14.6)],
    'WOM-1': [('h', 6.25)],
    'VCT-1': [('h', 25.0), ('v', 25.0)],
    'CT-1': [('d1', 26.5), ('d2', 26.5)],
    'EP-1': [('d1', 35.4)],
    'CPT-1': [('d2', 17.7)],
}


def segment(g, interior, line_thr=145, min_region_sf=15.0):
    lines = (g < params.thr(line_thr)) & interior
    lines_d = cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & interior
    free = (interior & ~lines_d).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(free, connectivity=4)
    small = st[:, 4] < min_region_sf * PPF ** 2
    small[0] = True
    lab[small[lab]] = 0
    # assign line pixels / specks to the nearest region
    _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    full = lab[iy, ix]
    full[~interior] = 0
    return full


def _amplitudes(K, ys, xs, mode, periods):
    """Amplitude (darkness units) of the periodic component of the projection
    of K along the given direction, for each period."""
    if mode == 'h':
        u = ys
    elif mode == 'v':
        u = xs
    elif mode == 'd1':
        u = xs + ys
    else:
        u = xs - ys
    u = u - u.min()
    n = int(u.max()) + 1
    s = np.bincount(u, weights=K, minlength=n)
    c = np.bincount(u, minlength=n).astype(np.float64)
    ok = c > 15
    idx = np.nonzero(ok)[0]
    if len(idx) < 48:
        return {p: 0.0 for p in periods}
    p = np.where(ok, s / np.maximum(c, 1), 0)
    p = np.interp(np.arange(idx[0], idx[-1] + 1), idx, p[idx])
    if len(p) > 162:
        p = p - ndi.uniform_filter1d(p, 81, mode='nearest')
    else:
        p = p - p.mean()
    N = len(p)
    w = np.hanning(N)
    pw = p * w
    nn = np.arange(N)
    out = {}
    for per in periods:
        best = 0.0
        for pp in np.linspace(per * 0.97, per * 1.03, 7):
            z = np.abs((pw * np.exp(-2j * np.pi * nn / pp)).sum()) * 2 / w.sum()
            best = max(best, z)
        out[per] = float(best)
    return out


TARGETS = {'h': [14.6, 6.25, 25.0], 'v': [25.0], 'd1': [26.5, 35.4], 'd2': [26.5, 17.7]}


def region_features(g, mask):
    m8 = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    if m8.sum() < 400:
        m8 = mask
    ys, xs = np.nonzero(m8)
    if len(ys) > 300000:
        sel = np.random.default_rng(0).choice(len(ys), 300000, replace=False)
        ys, xs = ys[sel], xs[sel]
    K = np.clip(248.0 - g[ys, xs].astype(np.float32), 0, 100)
    return {mode: _amplitudes(K, ys, xs, mode, per) for mode, per in TARGETS.items()}


def noise_sigma(g, sigma=1.5, tile=700):
    """Noise of the smoothed image, estimated on blank-ish tiles of the sheet
    (the least textured of a set of tiles spread over the sheet)."""
    H, W = g.shape
    ests = []
    for fy in (0.08, 0.3, 0.5, 0.7, 0.85):
        for fx in (0.05, 0.25, 0.45, 0.65, 0.85):
            y0, x0 = int(fy * (H - tile)), int(fx * (W - tile))
            t = g[y0:y0 + tile, x0:x0 + tile].astype(np.float32)
            gs = cv2.GaussianBlur(t, (0, 0), sigma)
            hp = gs - cv2.GaussianBlur(gs, (0, 0), 3.0)
            sub = hp[20:-20:3, 20:-20:3]
            ests.append(1.4826 * float(np.median(np.abs(sub - np.median(sub)))))
    ests.sort()
    return ests[len(ests) // 4]


def dot_density(g, labels, sn=None, sigma=1.5):
    """Count small isolated dark specks (SC-1 stipple) per region label.
    Threshold adapts to the scan's noise level (sn from the whole sheet)."""
    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma)
    if sn is None:
        sn = noise_sigma(g, sigma)
    bg = cv2.dilate(gs, np.ones((9, 9), np.uint8))
    con = bg - gs
    T = max(8.0, 5.0 * sn)
    m = (con > T).astype(np.uint8)
    n, cl, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    sel = (st[:, 4] >= 2) & (st[:, 4] <= 30) & (st[:, 2] <= 8) & (st[:, 3] <= 8)
    sel[0] = False
    cx = np.clip(cen[sel, 0].astype(int), 0, g.shape[1] - 1)
    cy = np.clip(cen[sel, 1].astype(int), 0, g.shape[0] - 1)
    b = labels[cy, cx]
    return np.bincount(b, minlength=int(labels.max()) + 1)


def class_scores(A):
    h, v, d1, d2 = A['h'], A['v'], A['d1'], A['d2']
    return {
        'LVT-1': h[14.6] - 0.5 * max(v[25.0], d1[26.5], d2[26.5]),
        'WOM-1': h[6.25] - h[25.0] - v[25.0],
        'VCT-1': min(h[25.0], v[25.0]),
        'CT-1': min(d1[26.5], d2[26.5]),
        'EP-1': d1[35.4] - d2[26.5] - d2[17.7],
        'CPT-1': d2[17.7] - d1[26.5] - d1[35.4],
    }


def classify(A, dots_per_100sf, min_amp=2.5):
    sc = class_scores(A)
    best = max(sc, key=sc.get)
    # dense stipple with only weak periodicity is sealed concrete
    if dots_per_100sf >= 25 and sc[best] < 4.0:
        return 'SC-1', sc
    if sc[best] >= min_amp * min(params.K, 1.0):
        return best, sc
    return ('SC-1' if dots_per_100sf >= 25 else 'PC-1'), sc
