"""Sheet layout: graphic scale, legend swatches."""
import numpy as np
import cv2


def _runs(mask):
    """Return list of (start, end_exclusive) runs of True in 1D bool array."""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


def _subpix_edge(profile, i, rising, level):
    """Find sub-pixel crossing of `level` near index i in 1D profile."""
    n = len(profile)
    lo = max(0, i - 4)
    hi = min(n - 1, i + 4)
    for j in range(lo, hi):
        a, b = profile[j], profile[j + 1]
        if rising and a < level <= b:
            return j + (level - a) / (b - a)
        if (not rising) and a >= level > b:
            return j + (a - level) / (a - b)
    return float(i)


def find_scale_bar(D):
    """Locate the graphic scale (alternating bar 0-4-8-16-32 ft) from its two solid
    segments (0-4 and 8-16). Returns dict or None."""
    H, W = D.shape
    best = None
    for thr in (0.5, 0.35):
        B = (D > thr).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(B, connectivity=4)
        # solid runs: analyse each component's middle row for long dark runs
        segs = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if not (8 <= h <= 60 and w >= 1.5 * h and w < 0.3 * W):
                continue
            y0 = y + int(round(h * 0.3))
            y1 = y + int(round(h * 0.7)) + 1
            prof = D[y0:y1, x:x + w].mean(axis=0)
            for r0, r1 in _runs(prof > 0.5 * max(thr, prof.max() * 0.8)):
                if r1 - r0 >= 1.5 * h:
                    # solid: column coverage over the component height
                    blk = D[y:y + h, x + r0:x + r1] > thr
                    if blk.mean() > 0.8:
                        segs.append((x + r0, x + r1, y, h, prof, x))
        for A in segs:
            for Bs in segs:
                if Bs[0] <= A[1]:
                    continue
                if abs(A[2] - Bs[2]) > 4 or abs(A[3] - Bs[3]) > 4:
                    continue
                wa = A[1] - A[0]
                wb = Bs[1] - Bs[0]
                if not (1.75 < wb / wa < 2.3):
                    continue
                gap = Bs[0] - A[1]
                if not (0.7 * wa < gap < 1.3 * wa):
                    continue
                yy = A[2] + A[3] // 2
                y0 = A[2] + int(round(A[3] * 0.3))
                y1 = A[2] + int(round(A[3] * 0.7)) + 1
                xa = max(0, A[0] - 10)
                xb = min(W, Bs[1] + int(4.4 * wa) + 20)
                prof = D[y0:y1, xa:xb].mean(axis=0)
                lvl = 0.5 * float(np.percentile(prof[A[0] - xa:A[1] - xa], 50))
                la = _subpix_edge(prof, A[0] - xa, True, lvl) + xa
                ra = _subpix_edge(prof, A[1] - 1 - xa, False, lvl) + xa
                lb = _subpix_edge(prof, Bs[0] - xa, True, lvl) + xa
                rb = _subpix_edge(prof, Bs[1] - 1 - xa, False, lvl) + xa
                u1 = (lb - la) / 8.0
                u2 = (rb - ra) / 12.0
                if abs(u1 - u2) > 0.05 * max(u1, u2):
                    continue
                u = 0.5 * (u1 + u2)
                # optional end stroke near 32 ft
                stroke = max(0.0, (ra - la) - 4 * u)
                x32 = la + stroke / 2.0 + 32 * u
                i0 = int(round(x32 - 0.6 * u)) - xa
                i1 = int(round(x32 + 0.6 * u)) - xa
                ests = [u1, u2]
                if 0 <= i0 < i1 <= len(prof):
                    seg = prof[i0:i1]
                    j = int(np.argmax(seg))
                    if seg[j] > 0.25:
                        xe = _peak_subpix_1d(seg, j) + i0 + xa
                        u3 = (xe - (la + stroke / 2.0)) / 32.0
                        if abs(u3 - u) < 0.03 * u:
                            ests.append(u3)
                u = float(np.median(ests))
                score = wa * wb * (1.5 if len(ests) == 3 else 1.0)
                cand = dict(px_per_ft=u, x0=la, y=float(yy), bbox=(A[0], A[2], Bs[1] - A[0], A[3]),
                            ests=ests, score=score)
                if best is None or cand['score'] > best['score']:
                    best = cand
        if best is not None:
            break
    return best


def _peak_subpix_1d(p, i):
    if 0 < i < len(p) - 1:
        a, b, c = p[i - 1], p[i], p[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def _regular_stack(cands, n_expected, min_found):
    """From candidate boxes (same column), find n_expected regularly stacked boxes; fill gaps."""
    best = None
    for r in cands:
        grp = [q for q in cands if abs(q[0] - r[0]) <= 8 and abs(q[2] - r[2]) <= 8 and abs(q[3] - r[3]) <= 8]
        grp = sorted(set(grp), key=lambda q: q[1])
        if len(grp) < min_found:
            continue
        ys = np.array([q[1] for q in grp], float)
        d = np.diff(ys)
        if len(d) == 0:
            continue
        pitch = float(np.median(d))
        if pitch < r[3]:
            continue
        # index each box on the pitch grid
        idx = np.round((ys - ys[0]) / pitch).astype(int)
        resid = np.abs(ys - ys[0] - idx * pitch)
        ok = resid < 0.12 * pitch
        idx = idx[ok]
        gg = [g for g, o in zip(grp, ok) if o]
        if len(gg) < min_found:
            continue
        # choose window of n_expected consecutive indices with most hits
        for start in range(idx.min(), idx.max() + 1):
            sel = [(k, g) for k, g in zip(idx, gg) if start <= k < start + n_expected]
            if len(sel) < min_found:
                continue
            score = len(sel)
            if best is None or score > best[0]:
                best = (score, start, pitch, sel)
    if best is None:
        return None
    score, start, pitch, sel = best
    ks = np.array([k for k, g in sel])
    ys = np.array([g[1] for k, g in sel], float)
    # least squares y = y0 + k * pitch
    A = np.vstack([np.ones(len(ks)), ks]).T
    y0, p = np.linalg.lstsq(A, ys, rcond=None)[0]
    x = int(np.median([g[0] for k, g in sel]))
    w = int(np.median([g[2] for k, g in sel]))
    h = int(np.median([g[3] for k, g in sel]))
    have = {k: g for k, g in sel}
    out = []
    for k in range(start, start + n_expected):
        if k in have:
            out.append(tuple(int(v) for v in have[k]))
        else:
            out.append((x, int(round(y0 + k * p)), w, h))
    return out


def find_legend(D, px_per_ft=None):
    """Find legend swatches. Returns dict with 'floor' (8 boxes) and 'wall' (4 boxes)."""
    H, W = D.shape
    B = (D > 0.32).astype(np.uint8)
    x0 = int(W * 0.5)
    sub = B[:, x0:]
    n, lab, st, cen = cv2.connectedComponentsWithStats(sub, connectivity=8)
    rects = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 0.012 * W or w > 0.06 * W or h < 6:
            continue
        s = sub[y:y + h, x:x + w]
        top = s[:3].max(axis=0).mean()
        bot = s[-3:].max(axis=0).mean()
        lef = s[:, :3].max(axis=1).mean()
        rig = s[:, -3:].max(axis=1).mean()
        cov = min(top, bot, lef, rig)
        if cov > 0.55 and (top + bot + lef + rig) / 4 > 0.75:
            rects.append((x + x0, y, w, h))
    big = [r for r in rects if 0.3 < r[3] / r[2] < 0.8]
    floor_boxes = _regular_stack(big, 8, 5)
    wall_boxes = None
    if floor_boxes:
        fx, fw = floor_boxes[0][0], floor_boxes[0][2]
        ylast = floor_boxes[-1][1] + floor_boxes[-1][3]
        thin = [r for r in rects if r[3] / r[2] < 0.25 and abs(r[0] - fx) < 0.05 * fw
                and abs(r[2] - fw) < 0.06 * fw and r[1] > ylast]
        thin.sort(key=lambda q: q[1])
        if len(thin) >= 4:
            wall_boxes = thin[:4]
        elif len(thin) >= 2:
            # thickness differs per type, so fit pitch on y only
            ys = np.array([q[1] for q in thin], float)
            pitch = float(np.median(np.diff(ys)))
            k = np.round((ys - ys[0]) / pitch).astype(int)
            have = {int(kk): q for kk, q in zip(k, thin)}
            wall_boxes = []
            for kk in range(4):
                if kk in have:
                    wall_boxes.append(have[kk])
                else:
                    ref = thin[0]
                    wall_boxes.append((ref[0], int(round(ys[0] + kk * pitch)), ref[2], ref[3]))
    return dict(floor=floor_boxes, wall=wall_boxes, rects=rects)
