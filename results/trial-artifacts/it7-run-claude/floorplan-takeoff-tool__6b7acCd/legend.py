"""Find legend swatches (floor finishes and partition types)."""
import numpy as np
import cv2

FLOOR_CODES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']
WALL_CODES = ['EXT', 'P3', 'P2', 'P1']


def _components(dk, th, close):
    B = (dk > th).astype(np.uint8)
    if close:
        B = cv2.morphologyEx(B, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(B, connectivity=8)
    return st[1:]


def _find_floor(st, ppf):
    W = st[:, 2] / ppf
    Hh = st[:, 3] / ppf
    fl = st[(W > 5.5) & (W < 9.5) & (Hh > 2.6) & (Hh < 5.0) & (st[:, 2] > 1.4 * st[:, 3])]
    best = None
    for i in range(len(fl)):
        x, y, w, h, _ = fl[i]
        grp = fl[(np.abs(fl[:, 0] - x) <= 5) & (np.abs(fl[:, 2] - w) <= 6) & (np.abs(fl[:, 3] - h) <= 6)]
        if len(grp) < 8:
            continue
        grp = grp[np.argsort(grp[:, 1])]
        # drop duplicates (same y)
        uniq = [grp[0]]
        for g in grp[1:]:
            if g[1] - uniq[-1][1] > 0.5 * h:
                uniq.append(g)
        grp = np.array(uniq)
        for k in range(len(grp) - 7):
            g = grp[k:k + 8]
            p = np.diff(g[:, 1])
            if p.min() < h:
                continue
            err = p.std() / p.mean()
            if err < 0.06 and (best is None or err < best[0]):
                best = (err, g)
    return best


def find_swatches(dk, ppf):
    """Return dict with 'floor': list of 8 (x,y,w,h) and 'wall': list of 4 (x,y,w,h)."""
    sm = cv2.GaussianBlur(dk, (0, 0), 0.7)
    result = None
    for th, close in ((0.3, False), (0.3, True), (0.2, True), (0.13, True), (0.45, False), (0.08, True)):
        st = _components(sm, th, close)
        best = _find_floor(st, ppf)
        if best is None:
            continue
        floor = [tuple(int(v) for v in r[:4]) for r in best[1]]
        fx = np.median([f[0] for f in floor])
        fw = np.median([f[2] for f in floor])
        ybot = floor[-1][1] + floor[-1][3]
        pitch = np.median(np.diff([f[1] for f in floor]))
        wl = st[(np.abs(st[:, 0] - fx) < 0.6 * ppf) & (st[:, 2] > 0.85 * fw) & (st[:, 2] < 1.25 * fw)
                & (st[:, 3] / ppf > 0.3) & (st[:, 3] / ppf < 1.3) & (st[:, 1] > ybot)
                & (st[:, 1] < ybot + 8 * pitch)]
        wl = wl[np.argsort(wl[:, 1])]
        uniq = []
        for g in wl:
            if not uniq or g[1] - uniq[-1][1] > 0.5 * pitch:
                uniq.append(g)
        if len(uniq) < 4:
            if result is None:
                result = dict(floor=floor, wall=None)
            continue
        walls = [tuple(int(v) for v in r[:4]) for r in uniq[:4]]
        return dict(floor=floor, wall=walls)
    if result is None:
        raise RuntimeError('floor legend swatches not found')
    # partition swatches not found by components: predict them from the floor swatch layout
    floor = result['floor']
    fx = int(np.median([f[0] for f in floor]))
    fw = int(np.median([f[2] for f in floor]))
    y_last = floor[-1][1]
    walls = []
    for k, hft in enumerate((0.8, 0.8, 0.8, 0.57)):
        y = int(y_last + (11.2 + 4.0 * k) * ppf)
        walls.append((fx, y, fw, int(round(hft * ppf))))
    return dict(floor=floor, wall=walls)
