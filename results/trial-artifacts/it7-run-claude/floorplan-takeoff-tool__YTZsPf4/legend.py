"""Locate legend swatches (floor finishes, partition types)."""
import numpy as np
import cv2
from scipy import ndimage as ndi

FLOOR_CODES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']
WALL_CODES = ['EXT', 'P3', 'P2', 'P1']


def _runs(mask1d, minlen):
    runs = []
    n = len(mask1d)
    i = 0
    while i < n:
        if mask1d[i]:
            j = i
            while j < n and mask1d[j]:
                j += 1
            if j - i >= minlen:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def find_legend(rot, ppf, plan_xmax):
    """Return dict with 'floor': list of 8 (x0,y0,x1,y1) boxes, 'wall': list of 4."""
    h, w = rot.shape
    dark = (rot < 0.62).astype(np.uint8)
    lab, n = ndi.label(dark, structure=np.ones((3, 3)))
    objs = ndi.find_objects(lab)
    sw_w = 7.3 * ppf
    thin = []
    for i, sl in enumerate(objs):
        hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
        if sl[1].start < plan_xmax:
            continue
        if 0.6 * sw_w < ww < 1.6 * sw_w and 0.25 * ppf < hh < 1.3 * ppf:
            thin.append((sl[1].start, sl[0].start, ww, hh))
    # group of 4 stacked thin boxes with same x0/width
    best = None
    for a in thin:
        grp = [b for b in thin if abs(b[0] - a[0]) <= 4 and abs(b[2] - a[2]) <= 6]
        grp.sort(key=lambda b: b[1])
        if len(grp) < 4:
            continue
        for k in range(len(grp) - 3):
            g = grp[k:k + 4]
            gaps = np.diff([b[1] for b in g])
            if gaps.min() > 1.5 * ppf and gaps.max() < 8 * ppf and gaps.max() / gaps.min() < 1.4:
                if best is None or g[0][1] > best[0][1]:
                    best = g
    if best is None:
        raise RuntimeError('legend partition swatches not found')
    walls = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in best]
    x0 = int(np.median([b[0] for b in walls]))
    x1 = int(np.median([b[2] for b in walls]))
    ytop = walls[0][1]
    # floor swatches: vertical border runs at the left/right edge columns above
    # the wall swatches (retry more leniently for faint / broken borders)
    sm = cv2.blur(rot[:ytop], (1, 5))
    runs = None
    for thr, gap_ft in ((0.8, 0.4), (0.9, 0.4), (0.9, 0.8)):
        band = (sm[:, x0 - 2:x0 + 4].min(axis=1) < thr)
        bandr = (sm[:, x1 - 4:x1 + 2].min(axis=1) < thr)
        both = band & bandr
        gap = max(int(gap_ft * ppf), 1)
        both = ndi.binary_closing(both, structure=np.ones(gap)) | both
        rr = _runs(both.astype(bool), int(2.0 * ppf))
        rr = [r for r in rr if r[1] - r[0] < 6 * ppf]
        if len(rr) >= 8:
            rr = rr[-8:]
            hts = np.array([b - a for a, b in rr], float)
            pitch = np.diff([a for a, b in rr])
            if hts.std() < 0.25 * ppf and pitch.std() < 0.3 * ppf:
                runs = rr
                break
    if runs is None:
        raise RuntimeError('floor swatches not found')
    floors = []
    for (a, b) in runs:
        floors.append((x0, a, x1, b))
    # refine each floor swatch border with profile analysis
    floors = [refine_box(rot, bx) for bx in floors]
    walls = [refine_box(rot, bx, thin=True) for bx in walls]
    return {'floor': floors, 'wall': walls}


def refine_box(rot, box, thin=False):
    """Refine a rectangle (outline) to subpixel border line centres."""
    x0, y0, x1, y1 = box
    def center(prof, c, half=3):
        c = int(round(c))
        lo = max(c - half, 0); hi = min(c + half + 1, len(prof))
        seg = prof[lo:hi]
        ink = np.clip(1.0 - seg, 0, None)
        ink = np.clip(ink - 0.3 * ink.max(), 0, None)
        if ink.sum() < 1e-6:
            return float(c)
        return lo + float((ink * np.arange(lo, hi)).sum() / ink.sum()) - lo
    pad = 6
    ys = slice(int(y0) + (2 if thin else 6), int(y1) - (2 if thin else 6))
    if ys.stop <= ys.start:
        ys = slice(int(y0), int(y1))
    rowL = rot[ys, int(x0) - pad:int(x0) + pad + 1].mean(axis=0)
    rowR = rot[ys, int(x1) - pad - 1:int(x1) + pad].mean(axis=0)
    xs = slice(int(x0) + 6, int(x1) - 6)
    colT = rot[int(y0) - pad:int(y0) + pad + 1, xs].mean(axis=1)
    colB = rot[int(y1) - pad - 1:int(y1) + pad, xs].mean(axis=1)
    nx0 = int(x0) - pad + center(rowL, pad + 1)
    nx1 = int(x1) - pad - 1 + center(rowR, pad - 1)
    ny0 = int(y0) - pad + center(colT, pad + 1)
    ny1 = int(y1) - pad - 1 + center(colB, pad - 1)
    return (nx0, ny0, nx1, ny1)
