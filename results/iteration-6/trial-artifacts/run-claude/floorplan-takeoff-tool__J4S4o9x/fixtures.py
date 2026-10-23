"""Fixture symbols: rectangles built from axis-aligned fixture linework."""
import numpy as np
import cv2

from canon import PPF
import params
from walls import _trace_cc


def fixture_mask(g, thr=190, sigma=0.8):
    gb = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma)
    return (gb < params.thr(thr)).astype(np.uint8)


def _segs(m, minlen, max_thick=100000):
    o = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, minlen), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(o, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h > max_thick:
            continue
        if h <= 4:
            out.append([float(cen[i][1]), int(x), int(x + w - 1)])
            continue
        if h > 12:
            # several lines joined by other strokes -> trace individual tracks
            cm = (lab[y:y + h, x:x + w] == i)
            for s in _trace_cc(cm, cm.astype(np.float32), int(x), int(y), minlen):
                out.append([s[0], s[1], s[2]])
            continue
        # thick: possibly several parallel lines merged; split at profile peaks
        prof = (lab[y:y + h, x:x + w] == i).sum(axis=1).astype(np.float64)
        pk = [k for k in range(h) if prof[k] >= 0.6 * prof.max() and
              (k == 0 or prof[k] >= prof[k - 1]) and (k == h - 1 or prof[k] >= prof[k + 1])]
        groups = []
        for k in pk:
            if groups and k - groups[-1][-1] < 3:
                groups[-1].append(k)
            else:
                groups.append([k])
        if len(groups) <= 1:
            out.append([float(cen[i][1]), int(x), int(x + w - 1)])
        else:
            for gk in groups:
                out.append([float(y + np.mean(gk)), int(x), int(x + w - 1)])
    # merge collinear with small gaps
    merged = []
    for s in sorted(out, key=lambda s: (s[0], s[1])):
        for m_ in merged:
            if abs(m_[0] - s[0]) <= 1.5 and s[1] <= m_[2] + 4 and s[2] >= m_[1] - 4:
                m_[1] = min(m_[1], s[1]); m_[2] = max(m_[2], s[2])
                break
        else:
            merged.append(list(s))
    return merged


class Rect:
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'feat')

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.feat = {}

    @property
    def w(self):
        return (self.x1 - self.x0) / PPF

    @property
    def h(self):
        return (self.y1 - self.y0) / PPF

    def __repr__(self):
        return 'Rect(%.0f,%.0f %.2fx%.2f ft %s)' % (self.x0, self.y0, self.w, self.h, self.feat)


def _coverage(segs_by_c, c, a0, a1, tol=2.0):
    """fraction of [a0,a1] covered by segments at coordinate c."""
    cov = []
    for s in segs_by_c:
        if abs(s[0] - c) <= tol and s[2] >= a0 and s[1] <= a1:
            cov.append((max(s[1], a0), min(s[2], a1)))
    if not cov:
        return 0.0
    cov.sort()
    tot = 0
    cur0, cur1 = cov[0]
    for u, v in cov[1:]:
        if u <= cur1 + 1:
            cur1 = max(cur1, v)
        else:
            tot += cur1 - cur0 + 1
            cur0, cur1 = u, v
    tot += cur1 - cur0 + 1
    return tot / max(a1 - a0 + 1, 1)


def line_masks(g, thr=192, sig_along=4.0, sig_across=0.6):
    """Masks of horizontal / vertical fixture linework, using smoothing along
    the line direction so faint but continuous outlines survive."""
    gf = g.astype(np.float32)
    hs = cv2.GaussianBlur(gf, (0, 0), sigmaX=sig_along, sigmaY=sig_across)
    vs = cv2.GaussianBlur(gf, (0, 0), sigmaX=sig_across, sigmaY=sig_along)
    t = params.thr(thr)
    return (hs < t).astype(np.uint8), (vs < t).astype(np.uint8)


def find_rects(fm, roi, minlen=9, min_side=8, max_side=20 * PPF, hv=None):
    x0r, y0r, x1r, y1r = roi
    if hv is None:
        hm = vm = fm
    else:
        hm, vm = hv
    H = _segs(hm[y0r:y1r, x0r:x1r], minlen)
    V = _segs(np.ascontiguousarray(vm[y0r:y1r, x0r:x1r].T), minlen)
    # shift to absolute coordinates
    H = [[s[0] + y0r, s[1] + x0r, s[2] + x0r] for s in H]
    V = [[s[0] + x0r, s[1] + y0r, s[2] + y0r] for s in V]
    rects = []
    Hs = sorted(H, key=lambda s: s[0])
    # candidate x-edges: endpoints of H segments plus V segment positions
    for i, A in enumerate(Hs):
        for B in Hs[i + 1:]:
            dy = B[0] - A[0]
            if dy < min_side:
                continue
            if dy > max_side:
                break
            # overlapping extent
            xa, xb = max(A[1], B[1]), min(A[2], B[2])
            if xb - xa < min_side:
                continue
            # vertical edges: V segments spanning A..B within [xa-2, xb+2]
            vx = [v[0] for v in V if xa - 8 <= v[0] <= xb + 8 and v[1] <= A[0] + 5 and v[2] >= B[0] - 5]
            if len(vx) < 2:
                continue
            vx = sorted(set(round(x, 1) for x in vx))
            # consecutive pairs of vertical edges -> rectangles
            for k in range(len(vx) - 1):
                xl, xr = vx[k], vx[k + 1]
                if xr - xl < min_side:
                    continue
                if _coverage(Hs, A[0], xl, xr) > 0.85 and _coverage(Hs, B[0], xl, xr) > 0.85:
                    rects.append(Rect(xl, A[0], xr, B[0]))
    return rects, H, V


def _line_hits(fm, p0, p1, tol=1):
    """fraction of points along p0->p1 that hit the mask (within tol px)."""
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    if n < 4:
        return 0.0
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    # ignore the ends (corners)
    k = max(2, n // 12)
    xs, ys = xs[k:-k], ys[k:-k]
    H, W = fm.shape
    hit = np.zeros(len(xs), bool)
    for dx in range(-tol, tol + 1):
        for dy in range(-tol, tol + 1):
            xi = np.clip(np.round(xs + dx).astype(int), 0, W - 1)
            yi = np.clip(np.round(ys + dy).astype(int), 0, H - 1)
            hit |= fm[yi, xi] > 0
    return float(hit.mean())


def _circle_hits(fm, cx, cy, r, tol=1):
    t = np.linspace(0, 2 * np.pi, max(24, int(2 * np.pi * r)), endpoint=False)
    xs = cx + r * np.cos(t)
    ys = cy + r * np.sin(t)
    H, W = fm.shape
    hit = np.zeros(len(t), bool)
    for dx in range(-tol, tol + 1):
        for dy in range(-tol, tol + 1):
            xi = np.clip(np.round(xs + dx).astype(int), 0, W - 1)
            yi = np.clip(np.round(ys + dy).astype(int), 0, H - 1)
            hit |= fm[yi, xi] > 0
    return float(hit.mean())


_GRAY = None


def cell_features(fm, r):
    x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
    d1 = _line_hits(fm, (x0, y0), (x1, y1))     # TL-BR
    d2 = _line_hits(fm, (x1, y0), (x0, y1))     # TR-BL
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(x1 - x0, y1 - y0)
    best = 0.0
    for rr in np.linspace(0.2 * side, 0.45 * side, 11):
        best = max(best, _circle_hits(fm, cx, cy, rr))
    # residual linework inside the cell away from the border and diagonals
    # (an opaque fixture is blank there; floor hatch is not)
    xi0, yi0 = int(np.ceil(x0 + 3)), int(np.ceil(y0 + 3))
    xi1, yi1 = int(np.floor(x1 - 3)), int(np.floor(y1 - 3))
    resid = 0.0
    if xi1 > xi0 and yi1 > yi0:
        sub = fm[yi0:yi1 + 1, xi0:xi1 + 1] > 0
        yy, xx = np.mgrid[yi0:yi1 + 1, xi0:xi1 + 1].astype(np.float32)
        w_, h_ = (x1 - x0), (y1 - y0)
        L = np.hypot(w_, h_)
        dist1 = np.abs((xx - x0) * h_ - (yy - y0) * w_) / L
        dist2 = np.abs((xx - x1) * h_ + (yy - y0) * w_) / L
        keep = (dist1 > 3.5) & (dist2 > 3.5)
        if keep.sum() >= 10:
            resid = float(sub[keep].mean())
    dark = 0.0
    if _GRAY is not None and xi1 > xi0 and yi1 > yi0:
        dark = float((_GRAY[yi0:yi1 + 1, xi0:xi1 + 1] < params.thr(120)).mean())
    r.feat = dict(d1=d1, d2=d2, circ=best, resid=resid, dark=dark)
    return r.feat


def minimal_cells(rects, H, V):
    out = []
    for r in rects:
        wpx, hpx = r.x1 - r.x0, r.y1 - r.y0
        split = False
        for s in H:
            if r.y0 + 3 < s[0] < r.y1 - 3:
                ov = min(s[2], r.x1) - max(s[1], r.x0)
                if ov > 0.85 * wpx:
                    split = True
                    break
        if not split:
            for s in V:
                if r.x0 + 3 < s[0] < r.x1 - 3:
                    ov = min(s[2], r.y1) - max(s[1], r.y0)
                    if ov > 0.85 * hpx:
                        split = True
                        break
        if not split:
            out.append(r)
    # dedupe near-identical
    uniq = []
    for r in out:
        if not any(abs(r.x0 - q.x0) < 3 and abs(r.y0 - q.y0) < 3 and abs(r.x1 - q.x1) < 3 and abs(r.y1 - q.y1) < 3 for q in uniq):
            uniq.append(r)
    return uniq


def _near(a, b, tol):
    return abs(a - b) <= tol


def gondola_runs(H, V, fm, width_ft=4.0, tol_px=5):
    """Runs: pairs of long parallel lines width_ft apart with full-width cross lines.
    Returns list of dict(orient, a, b, cuts) with cuts = cross-line positions."""
    runs = []
    for orient, L1, L2 in (('V', V, H), ('H', H, V)):
        # L1: long sides (for vertical runs: V segments), L2: cross lines
        Ls = sorted(L1, key=lambda s: s[0])
        for i, A in enumerate(Ls):
            if A[2] - A[1] < 2.5 * PPF:
                continue
            for B in Ls[i + 1:]:
                d = B[0] - A[0]
                if d > width_ft * PPF + tol_px:
                    break
                if d < width_ft * PPF - tol_px:
                    continue
                s0, s1 = max(A[1], B[1]), min(A[2], B[2])
                if s1 - s0 < 2.5 * PPF:
                    continue
                cuts = sorted(c[0] for c in L2
                              if c[1] <= A[0] + 8 and c[2] >= B[0] - 8 and s0 - 8 <= c[0] <= s1 + 8)
                # merge near-duplicate cuts
                cc = []
                for c in cuts:
                    if not cc or c - cc[-1] > 4:
                        cc.append(c)
                if len(cc) < 2:
                    continue
                runs.append(dict(orient=orient, a=A[0], b=B[0], cuts=cc))
    return runs


DIAG_T = 0.6


def count_fixtures(g, roi):
    global _GRAY
    _GRAY = g
    fm = fixture_mask(g)
    fmf = fixture_mask(g, thr=205)      # more lenient mask for symbol content tests
    hm, vm = line_masks(g)
    rects, H, V = find_rects(fm, roi, hv=(hm | fm, vm | fm))
    cells = minimal_cells(rects, H, V)
    res = dict(gondola_sections_4ft=0, gondola_sections_3ft=0, endcaps=0, checkouts=0,
               self_checkouts=0, cooler_doors=0, produce_tables=0)
    info = dict(cells=cells, runs=[], unknown=[])
    for r in cells:
        cell_features(fmf, r)
    # gondola runs: collect cells from both orientations, dedupe by geometry
    gr = gondola_runs(H, V, fm)
    gcells = []
    for run in gr:
        cuts = run['cuts']
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            if run['orient'] == 'V':
                rr = Rect(run['a'], c0, run['b'], c1)
            else:
                rr = Rect(c0, run['a'], c1, run['b'])
            if any(abs(rr.x0 - q.x0) < 4 and abs(rr.y0 - q.y0) < 4 and abs(rr.x1 - q.x1) < 4 and abs(rr.y1 - q.y1) < 4
                   for q, _ in gcells):
                continue
            gcells.append((rr, (c1 - c0) / PPF))
        info['runs'].append(run)
    for rr, l in gcells:
        f = cell_features(fmf, rr)
        diag = max(f['d1'], f['d2']) > DIAG_T
        xx = min(f['d1'], f['d2']) > DIAG_T
        if xx or f['resid'] > 0.25:      # X-ed (table) or hatched floor, not a shelf
            continue
        if abs(l - 1.5) <= 0.3 and diag:
            res['endcaps'] += 1
        elif abs(l - 4.0) <= 0.3 and not diag:
            res['gondola_sections_4ft'] += 1
        elif abs(l - 3.0) <= 0.3 and not diag:
            res['gondola_sections_3ft'] += 1
        else:
            info['unknown'].append(('gondola cell', l, f))
    for r in cells:
        w, h = r.w, r.h
        lo, hi = min(w, h), max(w, h)
        f = r.feat
        clean = f['resid'] < 0.2 and f['dark'] < 0.03     # symbols are light linework on blank
        x = min(f['d1'], f['d2']) > DIAG_T and clean
        one = max(f['d1'], f['d2']) > 0.85 and min(f['d1'], f['d2']) <= DIAG_T and clean
        if 3.8 <= lo <= 4.25 and 4.75 <= hi <= 5.3 and x:
            res['produce_tables'] += 1
        elif 2.75 <= lo <= 3.25 and 2.75 <= hi <= 3.25 and f['circ'] > 0.75 and not x and f['dark'] < 0.03:
            res['self_checkouts'] += 1
        elif 2.3 <= lo <= 2.75 and 2.75 <= hi <= 3.25 and x:
            res['checkouts'] += 1
        elif 0.45 <= lo <= 0.85 and 2.0 <= hi <= 2.8 and one:
            res['cooler_doors'] += 1
    return res, info
