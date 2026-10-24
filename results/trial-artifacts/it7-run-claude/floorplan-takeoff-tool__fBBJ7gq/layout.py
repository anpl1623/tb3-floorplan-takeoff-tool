"""Sheet layout: legend swatches and partition-type bars."""
import numpy as np
import cv2

FINISH_CODES = ["LVT-1", "PC-1", "SC-1", "EP-1", "VCT-1", "CT-1", "CPT-1", "WOM-1"]
WALL_CODES = ["EXT", "P3", "P2", "P1"]          # legend order (top to bottom)


def _rect_cover(b, x, y, w, h, k=3):
    sub = b[y:y + h, x:x + w]
    return [float(sub[0:k].max(axis=0).mean()), float(sub[-k:].max(axis=0).mean()),
            float(sub[:, 0:k].max(axis=1).mean()), float(sub[:, -k:].max(axis=1).mean())]


def _binary(kn, thr):
    s = cv2.GaussianBlur(kn, (0, 0), 1.0)
    b = (s > thr).astype(np.uint8)
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _components(b, ppf, wr, hr, ar=None, cover=None):
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (wr[0] * ppf < w < wr[1] * ppf and hr[0] * ppf < h < hr[1] * ppf):
            continue
        if ar is not None and not (ar[0] < w / float(h) < ar[1]):
            continue
        if cover is not None:
            cov = _rect_cover(b, x, y, w, h)
            if min(cov) < cover:
                continue
        out.append((int(x), int(y), int(w), int(h)))
    return out


def _best_column(rects, n_needed, ppf, min_found):
    """Find an evenly pitched column of n_needed rectangles (allowing gaps)."""
    best = None
    for r in rects:
        grp = [q for q in rects if abs(q[0] - r[0]) <= 0.3 * ppf and abs(q[2] - r[2]) <= 0.3 * ppf
               and abs(q[3] - r[3]) <= 0.3 * ppf]
        grp = sorted(set(grp), key=lambda q: q[1])
        ded = []
        for q in grp:
            if ded and abs(q[1] - ded[-1][1]) < 0.3 * ppf:
                continue
            ded.append(q)
        grp = ded
        if len(grp) < min_found:
            continue
        ys = np.array([q[1] for q in grp], float)
        # candidate pitches from pairwise differences
        diffs = np.diff(ys)
        for p in diffs:
            if p < r[3]:
                continue
            for y0 in ys:
                idx = np.round((ys - y0) / p)
                ok = (np.abs(ys - (y0 + idx * p)) < 0.15 * ppf) & (idx >= 0) & (idx < n_needed)
                cnt = int(len(np.unique(idx[ok])))
                if cnt >= min_found and (best is None or cnt > best[0] or (cnt == best[0] and y0 < best[2])):
                    # refine pitch by least squares
                    ii = idx[ok]
                    yy = ys[ok]
                    if len(np.unique(ii)) >= 2:
                        A = np.vstack([np.ones_like(ii), ii]).T
                        sol = np.linalg.lstsq(A, yy, rcond=None)[0]
                        yb, pp = float(sol[0]), float(sol[1])
                    else:
                        yb, pp = float(y0), float(p)
                    best = (cnt, pp, yb, grp)
    return best


def find_legend(kn, ppf):
    """Find the 8 floor-finish swatches and the 4 partition bars of the legend.

    Swatches are outlined rectangles ~7.3' x 3.7' (at plan scale) stacked in a
    column with a constant pitch; the partition bars are thin outlined bars
    left-aligned below them."""
    rects = []
    best = None
    used_thr = []
    for thr in (0.4, 0.25, 0.15, 0.1, 0.07):
        b = _binary(kn, thr)
        rects += _components(b, ppf, (5, 10), (2.2, 5.5), (1.4, 3.2), cover=0.8)
        used_thr.append(thr)
        best = _best_column(rects, 8, ppf, 4)
        if best is not None and best[0] >= 7:
            break
    if best is None:
        return None
    cnt, pitch, ybase, grp = best
    x0 = int(np.median([q[0] for q in grp]))
    w0 = int(np.median([q[2] for q in grp]))
    h0 = int(np.median([q[3] for q in grp]))
    # resolve which 8 pitch positions hold swatches (missed ones may be at either end)
    if cnt < 8:
        sm = cv2.GaussianBlur(kn, (0, 0), 1.0)

        def evidence(yy):
            yy = int(round(yy))
            if yy < 2 or yy + h0 + 2 >= kn.shape[0]:
                return 0.0
            t = sm[yy - 2:yy + 3, x0 + 4:x0 + w0 - 4].max(axis=0)
            b = sm[yy + h0 - 3:yy + h0 + 2, x0 + 4:x0 + w0 - 4].max(axis=0)
            l_ = sm[yy + 4:yy + h0 - 4, x0 - 2:x0 + 3].max(axis=1)
            r_ = sm[yy + 4:yy + h0 - 4, x0 + w0 - 3:x0 + w0 + 2].max(axis=1)
            thr = 0.12
            return float(min((t > thr).mean(), (b > thr).mean(), (l_ > thr).mean(), (r_ > thr).mean()))
        best_k, best_ev = 0, -1.0
        for k in range(0, 8 - cnt + 1):
            yb0 = ybase - k * pitch
            ev = sum(evidence(yb0 + i * pitch) for i in range(8))
            if ev > best_ev + 1e-6:
                best_k, best_ev = k, ev
        ybase = ybase - best_k * pitch
    sw = []
    for i in range(8):
        yi = ybase + i * pitch
        # use detected rectangle if close
        m = [q for q in grp if abs(q[1] - yi) < 0.2 * ppf]
        if m:
            sw.append(m[0])
        else:
            sw.append((x0, int(round(yi)), w0, h0))
    ylast = sw[-1][1] + sw[-1][3]
    bars = []
    for thr in used_thr:
        b = _binary(kn, thr)
        bars += _components(b, ppf, (5, 10), (0.3, 1.3))
    cb = [q for q in bars if abs(q[0] - x0) < 0.5 * ppf and abs(q[2] - w0) < 0.6 * ppf
          and ylast < q[1] < ylast + 10 * pitch]
    cb = sorted(set(cb), key=lambda q: q[1])
    # dedupe near-identical
    ded = []
    for q in cb:
        if ded and abs(q[1] - ded[-1][1]) < 0.3 * ppf:
            continue
        ded.append(q)
    bars4 = None
    if len(ded) >= 4:
        for s in range(len(ded) - 3):
            g4 = ded[s:s + 4]
            pt = np.diff([q[1] for q in g4])
            if pt.min() > 0.5 * ppf and pt.max() - pt.min() < 0.4 * ppf:
                bars4 = g4
                break
    if bars4 is None and len(ded) >= 2:
        # infer missing bars from pitch
        pts = np.diff([q[1] for q in ded])
        p = float(np.median(pts))
        yb = ded[0][1]
        bars4 = []
        for i in range(4):
            yi = yb + i * p
            m = [q for q in ded if abs(q[1] - yi) < 0.3 * ppf]
            bars4.append(m[0] if m else (ded[0][0], int(round(yi)), ded[0][2], ded[0][3]))
    return dict(swatches=sw, bars=bars4, pitch=pitch)


def bar_outline_peak(kn, bars):
    """Median raw peak darkness of the partition legend bar outlines (heavy pen)."""
    vals = []
    for (x, y, w, h) in bars or []:
        xa, xb = x + int(0.15 * w), x + int(0.85 * w)
        prof = kn[max(0, y - 3):y + h + 3, xa:xb].mean(axis=1)
        if len(prof) < 6:
            continue
        m = len(prof) // 2
        vals.append(float(prof[:m].max()))
        vals.append(float(prof[m:].max()))
    return float(np.median(vals)) if vals else None
