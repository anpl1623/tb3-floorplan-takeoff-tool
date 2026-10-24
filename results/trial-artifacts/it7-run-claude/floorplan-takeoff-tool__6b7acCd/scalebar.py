"""Locate the graphic scale bar (0-4-8-16-32 ft) and measure pixels per foot."""
import numpy as np
import cv2


def _crossings(p, thr):
    out = []
    for i in range(len(p) - 1):
        a, b = p[i] - thr, p[i + 1] - thr
        if a == 0:
            continue
        if a * b < 0 or b == 0:
            out.append((i + a / (a - b), 1 if a < 0 else -1))  # +1: light->dark
    return out


def find_bar_blocks(dk):
    """Find the two solid segments (0-4 ft and 8-16 ft) of the scale bar."""
    B = (dk > 0.5).astype(np.uint8)
    op = cv2.morphologyEx(B, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(op, connectivity=4)
    blocks = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w < 30 or h < 6 or w > 1500:
            continue
        if a < 0.85 * w * h:
            continue
        if h > 0.6 * w:
            continue
        blocks.append((x, y, w, h))
    best = None
    for A in blocks:
        for Bk in blocks:
            if Bk is A:
                continue
            xa, ya, wa, ha = A
            xb, yb, wb, hb = Bk
            if xb <= xa:
                continue
            if abs(ya - yb) > 4 or abs(ha - hb) > 4:
                continue
            r = wb / wa
            g = (xb - (xa + wa)) / wa
            if not (1.75 < r < 2.25 and 0.75 < g < 1.25):
                continue
            err = abs(r - 2) + abs(g - 1)
            if best is None or err < best[0]:
                best = (err, A, Bk)
    return best


def measure_scale(dk):
    """Return (px_per_ft, info dict)."""
    best = find_bar_blocks(dk)
    if best is None:
        raise RuntimeError('scale bar not found')
    _, A, Bk = best
    xa, ya, wa, ha = A
    xb, yb, wb, hb = Bk
    rough = (xb + wb - xa) / 16.0
    y0 = min(ya, yb) + max(2, ha // 4)
    y1 = max(ya + ha, yb + hb) - max(2, ha // 4)
    x0 = max(0, int(xa - 0.5 * rough))
    x1 = min(dk.shape[1], int(xa + 33.5 * rough))
    prof = dk[y0:y1, x0:x1].mean(0)
    thr = 0.5 * np.percentile(dk[y0:y1, xa:xa + wa], 50)
    cr = [(x0 + p, s) for p, s in _crossings(prof, thr)]

    def nearest(pos, sgn, tol):
        best = None
        for p, s in cr:
            if s != sgn or abs(p - pos) > tol:
                continue
            if best is None or abs(p - pos) < abs(best - pos):
                best = p
        return best
    e0 = nearest(xa, 1, 0.3 * rough)
    e4 = nearest(xa + wa, -1, 0.3 * rough)
    e8 = nearest(xb, 1, 0.3 * rough)
    e16 = nearest(xb + wb, -1, 0.3 * rough)
    ests = []
    if None not in (e0, e8):
        ests.append(((e8 - e0) / 8.0, 8.0))
    if None not in (e4, e16):
        ests.append(((e16 - e4) / 12.0, 12.0))
    if not ests:
        ests.append((rough, 1.0))
    ppf0 = sum(v * w for v, w in ests) / sum(w for _, w in ests)
    # right end of the bar: thin outline stroke near 32 ft
    info = dict(blocks=(A, Bk), e=(e0, e4, e8, e16))
    if e0 is not None:
        pos32 = e0 + 32 * ppf0
        # thin stroke at right end: find dark peak in profile near pos32 using band including outline
        lo = int(pos32 - 0.6 * ppf0) - x0
        hi = int(pos32 + 0.6 * ppf0) - x0
        if lo > 0 and hi < len(prof):
            seg = prof[lo:hi]
            k = int(np.argmax(seg))
            if seg[k] > 0.35:
                # centroid of the stroke
                a = max(0, k - 3)
                b = min(len(seg), k + 4)
                wv = np.clip(seg[a:b] - 0.1, 0, None)
                if wv.sum() > 0:
                    c = x0 + lo + a + (wv * np.arange(b - a)).sum() / wv.sum()
                    # stroke width estimate: number of samples above half peak
                    sw = float((seg[a:b] > 0.5 * seg[k]).sum())
                    left_c = e0 + 0.5 * min(sw, 3.0)
                    v = (c - left_c) / 32.0
                    if abs(v - ppf0) < 0.01 * ppf0:
                        ests.append((v, 32.0))
                    info['right'] = float(c)
    ppf = sum(v * w for v, w in ests) / sum(w for _, w in ests)
    info['ests'] = [float(v) for v, _ in ests]
    info['bar_y'] = (int(min(ya, yb)), int(max(ya + ha, yb + hb)))
    info['bar_x'] = (float(e0 if e0 is not None else xa), float(xa + 32 * ppf))
    return float(ppf), info
