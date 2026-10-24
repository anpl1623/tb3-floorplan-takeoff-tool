"""Sheet layout: legend swatches and graphic scale."""
import numpy as np
import cv2


def _components(bw):
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw.astype(np.uint8), connectivity=8)
    return n, lab, st


def _runs(b):
    """Runs of a 1D boolean array -> list of (value, start, end)."""
    out = []
    if len(b) == 0:
        return out
    cur = b[0]
    s = 0
    for i in range(1, len(b)):
        if b[i] != cur:
            out.append((bool(cur), s, i))
            cur = b[i]
            s = i
    out.append((bool(cur), s, len(b)))
    return out


def find_scale_bar(ink):
    """Find the graphic scale (0-4-8-16-32 ft bar).  Returns (px_per_ft, info).

    The bar has black fills on 0-4 and 8-16 ft.  Same-polarity edge pairs (white->black at 0
    and 8 ft, black->white at 4 and 16 ft) are used so that blur and outline width cancel."""
    H, W = ink.shape
    bw = ink > 0.35
    n, lab, st = _components(bw)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 8 or h > 90 or w < 10 * h or w < 300:
            continue
        y0 = y + max(1, h // 4)
        y1 = y + h - max(1, h // 4)
        if y1 <= y0:
            continue
        xa = max(0, x - 6)
        xb = min(W, x + w + 6)
        prof = ink[y0:y1, xa:xb].mean(0)
        ups, dns = [], []
        for j in range(len(prof) - 1):
            if (prof[j] - 0.5) * (prof[j + 1] - 0.5) < 0:
                t = (0.5 - prof[j]) / (prof[j + 1] - prof[j])
                (ups if prof[j + 1] > prof[j] else dns).append(xa + j + t)
        if len(ups) < 2 or len(dns) < 2:
            continue
        u0, u8 = ups[0], ups[1]
        d4, d16 = dns[0], dns[1]
        w1 = d4 - u0          # 0..4 black
        w2 = d16 - u8         # 8..16 black
        gap = u8 - d4         # 4..8 white
        if w1 >= 10 and abs(w2 - 2 * w1) <= 0.15 * w1 + 3 and abs(gap - w1) <= 0.2 * w1 + 3:
            ppf = ((u8 - u0) + (d16 - d4)) / 20.0
        else:
            # bar drawn the other way round (16..8 black, 4..0 black)
            ua, ub = ups[-2], ups[-1]
            da, db = dns[-2], dns[-1]
            wa = da - ua
            wb = db - ub
            gp = ub - da
            if wb >= 10 and abs(wa - 2 * wb) <= 0.15 * wb + 3 and abs(gp - wb) <= 0.2 * wb + 3:
                ppf = ((ub - ua) + (db - da)) / 20.0
            else:
                continue
        cand = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'u0': u0, 'u8': u8,
                'd4': d4, 'd16': d16}
        if best is None or w > best[0]:
            best = (w, ppf, cand)
    if best is None:
        return None, None
    return best[1], best[2]


def _find_legend_thr(ink, thr):
    H, W = ink.shape
    bw = (ink > thr).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st = _components(bw)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 80 <= w <= 500 and 40 <= h <= 260 and 1.4 <= w / h <= 3.2:
            sub = bw[y:y + h, x:x + w] > 0
            t = sub[:4].any(0).mean()
            b = sub[-4:].any(0).mean()
            l = sub[:, :4].any(1).mean()
            r = sub[:, -4:].any(1).mean()
            if min(t, b, l, r) > 0.85:
                boxes.append((int(x), int(y), int(w), int(h)))
    best = None
    for b0 in boxes:
        grp = [b for b in boxes if abs(b[0] - b0[0]) <= 8 and abs(b[2] - b0[2]) <= 8
               and abs(b[3] - b0[3]) <= 8]
        grp.sort(key=lambda b: b[1])
        if len(grp) >= 8:
            for k in range(len(grp) - 7):
                g = grp[k:k + 8]
                pitch = np.diff([b[1] for b in g])
                if pitch.max() - pitch.min() <= 10 and pitch.min() > g[0][3]:
                    if best is None or g[0][1] < best[0][1]:
                        best = g
                    break
    if best is None:
        return None
    floor = best
    pitch = float(np.median(np.diff([b[1] for b in floor])))
    fx, fw = floor[0][0], floor[0][2]
    parts = []
    ylo = floor[-1][1] + floor[-1][3]
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if abs(x - fx) <= 15 and abs(w - fw) <= 15 and 5 <= h <= 60 and ylo < y < ylo + 8 * pitch:
            parts.append((int(x), int(y), int(w), int(h)))
    parts.sort(key=lambda b: b[1])
    # remove duplicates (overlapping boxes)
    ded = []
    for p in parts:
        if ded and abs(p[1] - ded[-1][1]) < 6:
            continue
        ded.append(p)
    parts = ded[:4]
    return {'floor': floor, 'parts': parts, 'pitch': pitch}


def find_legend(ink):
    """Locate the floor-finish swatches (8) and partition swatches (4)."""
    best = None
    for thr in (0.35, 0.25, 0.5, 0.18, 0.12):
        r = _find_legend_thr(ink, thr)
        if r is not None and len(r['parts']) == 4:
            return r
        if r is not None and best is None:
            best = r
    return best


def scale_from_legend(leg):
    """Fallback scale: the legend swatches are drawn at plan scale, 7.29 ft wide."""
    if not leg or not leg.get('floor'):
        return None
    w = float(np.median([b[2] for b in leg['floor']]))
    return w / 7.29
