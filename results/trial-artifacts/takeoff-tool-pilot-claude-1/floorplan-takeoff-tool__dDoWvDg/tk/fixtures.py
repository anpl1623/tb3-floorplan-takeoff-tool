"""Fixture counting: template matching for fixed symbols, structural analysis for gondolas."""
import os
import numpy as np
import cv2

C = 24.0   # canonical pixels per foot
_TDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


def load_template(name):
    return np.load(os.path.join(_TDIR, name + '.npy')).astype(np.float32)


def _variant_ops(rotations=(0, 1, 2, 3), flips=True):
    ops = []
    for k in rotations:
        ops.append((k, False))
        if flips:
            ops.append((k, True))
    return ops


def apply_op(a, op):
    k, fl = op
    r = np.rot90(a, k)
    return np.ascontiguousarray(np.fliplr(r) if fl else r)


def local_peaks(R, thr, rad):
    k = np.ones((2 * rad + 1, 2 * rad + 1), np.uint8)
    D = cv2.dilate(R, k)
    ys, xs = np.nonzero((R >= D - 1e-7) & (R >= thr))
    return list(zip(R[ys, xs].tolist(), xs.tolist(), ys.tolist()))


def nms(cands, rad=None, iou_thr=0.25):
    """Greedy overlap suppression on the matched template boxes: a candidate is dropped
    if it overlaps a stronger one (IoU above iou_thr) or its centre lies inside it."""
    cands = sorted(cands, key=lambda c: -c['score'])
    kept = []
    for c in cands:
        ok = True
        for q in kept:
            ix = min(c['x'] + c['w'], q['x'] + q['w']) - max(c['x'], q['x'])
            iy = min(c['y'] + c['h'], q['y'] + q['h']) - max(c['y'], q['y'])
            inter = max(0, ix) * max(0, iy)
            union = c['w'] * c['h'] + q['w'] * q['h'] - inter
            inside = (q['x'] <= c['cx'] <= q['x'] + q['w'] and q['y'] <= c['cy'] <= q['y'] + q['h'])
            if inter / float(union) > iou_thr or inside:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def match_symbol(Gc, tpl, thr, ops, peak_thr=None):
    """Return candidate matches for all template variants."""
    out = []
    pt = thr if peak_thr is None else peak_thr
    for op in ops:
        v = apply_op(tpl, op)
        if v.shape[0] >= Gc.shape[0] or v.shape[1] >= Gc.shape[1]:
            continue
        R = cv2.matchTemplate(Gc, v, cv2.TM_CCOEFF_NORMED)
        for sc, x, y in local_peaks(R, pt, max(2, min(v.shape) // 3)):
            out.append(dict(score=float(sc), x=int(x), y=int(y), cx=x + v.shape[1] / 2.0,
                            cy=y + v.shape[0] / 2.0, op=op, h=v.shape[0], w=v.shape[1]))
    return out


def _line_mask(shape, p0, p1, width=2):
    m = np.zeros(shape, np.uint8)
    cv2.line(m, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), 1, width)
    return m > 0


def diag_contrast(Gc, bgv, c, tpl_shape, rect, diag):
    """Mean darkness along the symbol's diagonal minus along the opposite diagonal.
    rect = (x0,y0,x1,y1) of the door rectangle in template coords; diag = 'up' (bottom-left
    to top-right) or 'down'."""
    x0, y0, x1, y1 = rect
    if diag == 'up':
        a, b = (x0, y1), (x1, y0)
        a2, b2 = (x0, y0), (x1, y1)
    else:
        a, b = (x0, y0), (x1, y1)
        a2, b2 = (x0, y1), (x1, y0)
    # shrink towards the centre to avoid the outline
    def shrink(p, q, f=0.15):
        return ((p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f), (q[0] + (p[0] - q[0]) * f, q[1] + (p[1] - q[1]) * f))
    a, b = shrink(a, b); a2, b2 = shrink(a2, b2)
    m1 = _line_mask(tpl_shape, a, b)
    m2 = _line_mask(tpl_shape, a2, b2)
    m1 = apply_op(m1.astype(np.uint8), c['op']) > 0
    m2 = apply_op(m2.astype(np.uint8), c['op']) > 0
    patch = Gc[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']]
    if patch.shape != m1.shape:
        return 0.0
    d1 = float((bgv - patch[m1]).mean())
    d2 = float((bgv - patch[m2]).mean())
    return d1 - d2


ALL_OPS = _variant_ops()
SYM = {
    # name: (template, ops, ncc threshold)
    'produce': ('produce', _variant_ops((0, 1), False), 0.75),
    'sco': ('sco', _variant_ops((0,), False), 0.75),
    'checkout': ('checkout', ALL_OPS, 0.80),
    'cooler': ('cooler', ALL_OPS, 0.70),
}
COOLER_RECT = (2.5, 2.5, 16.0, 57.0)


def count_symbols(Gc, bgv):
    res = {}
    dets = {}
    for name, (tname, ops, thr) in SYM.items():
        tpl = load_template(tname)
        cands = match_symbol(Gc, tpl, thr, ops)
        if name == 'cooler':
            good = []
            for c in cands:
                dc = diag_contrast(Gc, bgv, c, tpl.shape, COOLER_RECT, 'up')
                c['diag'] = dc
                if dc >= 10 and c['score'] >= 0.70:
                    good.append(c)
            cands = good
        kept = nms(cands, 0.45 * min(tpl.shape))
        dets[name] = kept
        res[name] = len(kept)
    return res, dets


def _peaks_1d(p, thr, min_sep):
    out = []
    for i in range(1, len(p) - 1):
        if p[i] >= thr and p[i] >= p[i - 1] and p[i] > p[i + 1]:
            if out and i - out[-1] < min_sep:
                if p[i] > p[out[-1]]:
                    out[-1] = i
                continue
            out.append(i)
    return out


def gondola_runs(Gc, bgv, transpose=False):
    """Find 4'-deep double-sided gondola runs drawn vertically (or horizontally if transpose)."""
    img = np.ascontiguousarray(Gc.T) if transpose else Gc
    D = np.clip(bgv - img, 0, None).astype(np.float32)
    k = int(C)  # 1 ft smoothing along the run
    Vs = cv2.blur(D, (1, k))
    Vm = cv2.dilate(Vs, np.ones((1, 3), np.uint8))   # +-1px tolerance across
    half = int(round(2 * C))
    H, W = img.shape
    R = np.zeros_like(Vs)
    R[:, half:W - half] = np.minimum(np.minimum(Vm[:, :W - 2 * half], Vm[:, half:W - half]), Vm[:, 2 * half:])
    # interior of each half should be blank (white fill masks the hatch)
    a, b = 5, half - 5
    I1 = cv2.dilate(Vs, np.ones((1, b - a + 1), np.uint8))   # max over a window of width (b-a+1)
    # centre of window for left half at x - (a+b)/2, right half at x + (a+b)/2
    off = (a + b) // 2
    Il = np.zeros_like(Vs); Ir = np.zeros_like(Vs)
    Il[:, off:] = I1[:, :W - off]
    Ir[:, :W - off] = I1[:, off:]
    line_thr = 35.0
    cand = (R > line_thr) & (Il < 25.0) & (Ir < 25.0)
    cand = cand.astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand, 8)
    runs = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if h < 2.0 * C:
            continue
        xc = int(round(cen[i][0]))
        runs.append(dict(xc=xc, y0=int(y), y1=int(y + h), transpose=transpose))
    # merge duplicates with nearby centre lines
    runs.sort(key=lambda r: (r['xc'], r['y0']))
    merged = []
    for r in runs:
        if merged and abs(r['xc'] - merged[-1]['xc']) <= 4 and r['y0'] <= merged[-1]['y1'] + 2 * C:
            merged[-1]['y1'] = max(merged[-1]['y1'], r['y1'])
            merged[-1]['y0'] = min(merged[-1]['y0'], r['y0'])
        else:
            merged.append(r)
    for r in merged:
        analyse_run(img, D, r, bgv)
    return merged


def analyse_run(img, D, r, bgv):
    """Locate dividers along a run; count 3ft/4ft sections and endcaps."""
    xc = r['xc']
    half = int(round(2 * C))
    H = img.shape[0]
    ext = int(3 * C)
    y0 = max(0, r['y0'] - ext); y1 = min(H, r['y1'] + ext)
    # cross-line profile: darkness averaged across both halves (excluding outer/centre lines)
    cols = list(range(xc - half + 5, xc - 4)) + list(range(xc + 5, xc + half - 4))
    cols = [c for c in cols if 0 <= c < img.shape[1]]
    prof = D[y0:y1, cols].mean(1)
    pk = _peaks_1d(prof, 40.0, int(0.5 * C))
    ys = [y0 + p for p in pk]
    segs = []
    for a, b in zip(ys[:-1], ys[1:]):
        L = (b - a) / C
        if b - a <= 6:
            continue
        mid = slice(int(a + 0.2 * (b - a)), int(b - 0.2 * (b - a)) + 1)
        cen_dark = D[mid, max(0, xc - 1):xc + 2].max(1).mean()
        left = D[mid, max(0, xc - half - 1):xc - half + 2].max(1).mean()
        right = D[mid, xc + half - 1:xc + half + 2].max(1).mean()
        inner = slice(int(a + 4), int(b - 3))
        lh = D[inner, xc - half + 5:xc - 4]
        rh = D[inner, xc + 5:xc + half - 4]
        blank_l = float(lh.mean()) if lh.size else 99.0
        blank_r = float(rh.mean()) if rh.size else 99.0
        # diagonals across the full width (endcaps)
        n = 30
        t = np.linspace(0.15, 0.85, n)
        xa = (xc - half) + t * (2 * half)
        ya1 = b - t * (b - a)
        ya2 = a + t * (b - a)
        def dmean(xs, ys_):
            xi = np.clip(np.round(xs).astype(int), 0, D.shape[1] - 1)
            vals = []
            for dy in (-1, 0, 1):
                yi = np.clip(np.round(ys_).astype(int) + dy, 0, D.shape[0] - 1)
                vals.append(D[yi, xi])
            return float(np.max(vals, 0).mean())
        dg1 = dmean(xa, ya1); dg2 = dmean(xa, ya2)
        segs.append(dict(a=a, b=b, L=L, cen=float(cen_dark), left=float(left), right=float(right),
                         blank=max(blank_l, blank_r), diag=max(dg1, dg2), adiag=min(dg1, dg2)))
    r['segs'] = segs
    r['profile_peaks'] = ys


def classify_segments(r, D):
    """Turn run segments into section / endcap counts."""
    n4 = n3 = ne = 0
    segs = r['segs']
    valid = []
    for sg in segs:
        sides = sg['left'] > 35 and sg['right'] > 35
        if sides and sg['cen'] > 35 and sg['blank'] < 9.0:
            valid.append(sg)
    # merge neighbouring pieces split by a spurious cross line (e.g. a grid line)
    def fits(L):
        return abs(L - 4.0) < 0.45 or abs(L - 3.0) < 0.45
    merged = []
    for sg in valid:
        if merged and abs(sg['a'] - merged[-1]['b']) <= 2 and not fits(sg['L']) and not fits(merged[-1]['L']) \
                and fits(sg['L'] + merged[-1]['L']):
            m = dict(merged[-1]); m['b'] = sg['b']; m['L'] = merged[-1]['L'] + sg['L']
            merged[-1] = m
        else:
            merged.append(sg)
    valid = merged
    for sg in valid:
        L = sg['L']
        if abs(L - 4.0) < 0.45:
            n4 += 1
        elif abs(L - 3.0) < 0.45:
            n3 += 1
        else:
            best = None
            for a4 in range(0, 8):
                for a3 in range(0, 10):
                    if a4 + a3 == 0:
                        continue
                    err = abs(4 * a4 + 3 * a3 - L)
                    if best is None or err < best[0]:
                        best = (err, a4, a3)
            if best and best[0] < 0.5:
                n4 += best[1]; n3 += best[2]
    if valid:
        for sg in segs:
            if any(sg['a'] >= v['a'] and sg['b'] <= v['b'] for v in valid):
                continue
            # an endcap closes the end of a run of sections
            adjacent = any(abs(sg['b'] - v['a']) <= 3 or abs(sg['a'] - v['b']) <= 3 for v in valid)
            if not adjacent:
                continue
            if 1.0 <= sg['L'] <= 2.2 and sg['left'] > 35 and sg['right'] > 35 and sg['cen'] < 35 \
                    and sg['diag'] > 30 and sg['adiag'] < 0.5 * sg['diag']:
                ne += 1
    r['n4'], r['n3'], r['ne'] = n4, n3, ne
    return n4, n3, ne


def resolve_runs(runs):
    """Runs whose 4'-deep bands overlap cannot coexist (back-to-back runs create a
    phantom run on their shared outer line).  Keep the heaviest non-overlapping set."""
    def w(r):
        return r.get('n4', 0) + r.get('n3', 0) + 0.5 * r.get('ne', 0)

    def conflict(a, b):
        if abs(a['xc'] - b['xc']) >= 3.5 * C:
            return False
        return min(a['y1'], b['y1']) - max(a['y0'], b['y0']) > 1.0 * C

    runs = [r for r in runs if w(r) > 0]
    n = len(runs)
    seen = [False] * n
    keep = []
    for i in range(n):
        if seen[i]:
            continue
        # connected cluster of conflicting runs
        stack, cl = [i], []
        seen[i] = True
        while stack:
            k = stack.pop(); cl.append(k)
            for j in range(n):
                if not seen[j] and conflict(runs[k], runs[j]):
                    seen[j] = True; stack.append(j)
        if len(cl) == 1 or len(cl) > 14:
            keep += [runs[k] for k in cl]
            continue
        best, best_w = [], -1.0
        for mask in range(1, 1 << len(cl)):
            sel = [cl[b] for b in range(len(cl)) if mask >> b & 1]
            if any(conflict(runs[a], runs[b]) for ai, a in enumerate(sel) for b in sel[ai + 1:]):
                continue
            tw = sum(w(runs[k]) for k in sel)
            if tw > best_w:
                best, best_w = sel, tw
        keep += [runs[k] for k in best]
    return keep
