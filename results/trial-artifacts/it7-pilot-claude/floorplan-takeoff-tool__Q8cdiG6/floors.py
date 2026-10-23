"""Floor finish regions: segmentation into cells and texture classification."""
import numpy as np
import cv2
from scipy import ndimage as ndi

from texture import describe, distance, acov_distance


def wall_footprint(shape, segs):
    F = np.zeros(shape, bool)
    for q in segs:
        a = min(q['ea'], q['a'])
        b = max(q['eb'], q['b'])
        lo = int(np.ceil(q['f1']))
        hi = int(np.floor(q['f2'])) + 1
        a0 = int(np.ceil(a))
        b0 = int(np.floor(b)) + 1
        if q['orient'] == 'H':
            F[max(0, lo):hi, max(0, a0):b0] = True
        else:
            F[max(0, a0):b0, max(0, lo):hi] = True
    return F


def segment_cells(sub, F, ppf, heavy_thr):
    """Split the building interior into cells bounded by walls and long heavy lines."""
    bld = ndi.binary_fill_holes(F)
    interior = bld & ~F
    g = cv2.GaussianBlur(sub, (0, 0), 0.7)
    H = ((g > heavy_thr) & interior).astype(np.uint8)
    # remove solid (filled) parts: anything surviving an opening wider than a line
    r = int(max(4, round(0.17 * ppf)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    solid = cv2.morphologyEx(H, cv2.MORPH_OPEN, ker)
    solid = cv2.dilate(solid, np.ones((3, 3), np.uint8))
    H = (H.astype(bool) & ~solid.astype(bool)).astype(np.uint8)
    ns, labs, sts, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    small_solid = np.zeros(ns, bool)
    small_solid[1:] = sts[1:, 4] < 3.0 * ppf * ppf
    small_solid = small_solid[labs]
    n, lab, st, _ = cv2.connectedComponentsWithStats(H, connectivity=8)
    thin = np.ones(n, bool)
    keep = np.zeros(n, bool)
    keep[1:] = (np.maximum(st[1:, 2], st[1:, 3]) >= 1.2 * ppf)
    Hl = keep[lab]
    Hl = cv2.dilate(Hl.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    cells = interior & ~Hl
    # text / blob exclusion mask for texture analysis
    bigh = np.zeros(n, bool)
    bigh[1:] = (np.maximum(st[1:, 2], st[1:, 3]) > 0.25 * ppf) & thin[1:]
    excl = cv2.dilate((bigh[lab] | small_solid).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return bld, interior, Hl, cells, excl


BLOB_THR = [None]


def dog(Dr):
    return cv2.GaussianBlur(Dr, (0, 0), 1.0) - cv2.GaussianBlur(Dr, (0, 0), 3.0)


def set_blob_threshold(D):
    """Noise-adaptive threshold for small dark blobs (DoG response)."""
    small = D[::2, ::2]
    Db = cv2.blur(small, (9, 9))
    m = Db < np.percentile(Db, 40)
    sub = D[:D.shape[0] // 2 * 2:2, :D.shape[1] // 2 * 2:2]
    g = dog(np.ascontiguousarray(D))[::2, ::2]
    v = g[m[:g.shape[0], :g.shape[1]]]
    if v.size < 1000:
        BLOB_THR[0] = 0.05
        return
    mad = float(np.median(np.abs(v - np.median(v)))) * 1.4826
    BLOB_THR[0] = float(max(0.03, 5.0 * mad))


def extra_feats(Dr, mask, ppf, thr):
    ink = Dr > thr
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    m = mask
    n = max(1, int(m.sum()))
    b05 = float((dist[m] > 0.5 * ppf).mean()) if m.any() else 0.0
    bt = BLOB_THR[0] if BLOB_THR[0] is not None else 0.05
    blobs = dog(Dr) > bt
    nl, lab, st, _ = cv2.connectedComponentsWithStats((blobs & m).astype(np.uint8), connectivity=8)
    areas = st[1:, 4]
    small = int(((areas >= 2) & (areas <= 0.06 * ppf * ppf)).sum())
    dens = small / (n / ppf ** 2)
    return dict(b05=b05, blob=dens)


def swatch_descriptors(D, boxes, ppf, thr):
    out = []
    for (x, y, w, h) in boxes:
        m = np.zeros((h + 4, w + 4), bool)
        m[7:h - 3, 7:w - 3] = True
        Dr = D[y - 2:y + h + 2, x - 2:x + w + 2]
        d = describe(Dr, m)
        d.update(extra_feats(Dr, m, ppf, thr))
        out.append(d)
    return out


def sheet_scales(sw):
    """Typical inter-swatch distances used to normalise each descriptor term."""
    da, dm, ds = [], [], []
    for i in range(len(sw)):
        for j in range(i + 1, len(sw)):
            da.append(acov_distance(sw[i], sw[j]))
            dm.append((sw[i]['mu'] - sw[j]['mu']) ** 2)
            ds.append((sw[i]['sd'] - sw[j]['sd']) ** 2)
    return dict(acov=max(float(np.median(da)), 1e-12), mu=max(float(np.median(dm)), 1e-6),
                sd=max(float(np.median(ds)), 1e-6))


def score_against(dc, sw, scales=None):
    scores = []
    for s in sw:
        dd = distance(dc, s)
        sc = dd['rho'] / 0.01 + dd['mu'] / 0.02 ** 2 + dd['sd'] / 0.02 ** 2
        db = max(0.0, abs(dc['b05'] - s['b05']) - 0.15)
        sc += (db / 0.1) ** 2
        sc += (np.log1p(dc['blob']) - np.log1p(s['blob'])) ** 2 / 0.1
        scores.append(sc)
    return np.array(scores)


def classify_cells(sub, cells, excl, fixmask, ppf, sw, thr, min_cell_sf=4.0):
    n, lab, st, _ = cv2.connectedComponentsWithStats(cells.astype(np.uint8), connectivity=4)
    scales = sheet_scales(sw)
    labels = np.full(n, -1, int)
    info = {}
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_cell_sf * ppf * ppf:
            continue
        m = lab[y:y + h, x:x + w] == i
        m = ndi.binary_fill_holes(m)
        m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        m &= ~excl[y:y + h, x:x + w]
        m2 = m & ~fixmask[y:y + h, x:x + w]
        if m2.sum() > 0.3 * ppf * ppf * 4:
            m = m2
        if m.sum() < 200:
            continue
        Dr = sub[y:y + h, x:x + w]
        d = describe(Dr, m)
        if d is None:
            continue
        d.update(extra_feats(Dr, m, ppf, thr))
        sc = score_against(d, sw, scales)
        labels[i] = int(np.argmin(sc))
        info[i] = dict(area=a, scores=sc, desc=d)
    return lab, labels, info


def assign_areas(lab, labels, interior, nclass):
    """Assign every interior pixel to the nearest classified cell; return pixel counts per class."""
    known = labels[lab] if lab.max() > 0 else np.full(lab.shape, -1)
    known = np.where(interior, known, -1)
    have = known >= 0
    if not have.any():
        return np.zeros(nclass), known
    # nearest labelled pixel for unlabelled interior pixels
    dist, (iy, ix) = ndi.distance_transform_edt(~have, return_indices=True)
    full = known[iy, ix]
    full = np.where(interior, full, -1)
    counts = np.array([(full == k).sum() for k in range(nclass)], float)
    return counts, full
