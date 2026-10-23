"""Owner fixtures: gondola runs, endcaps, checkouts, self checkouts, cooler doors, produce tables."""
import numpy as np
import cv2
from scipy import ndimage as ndi

from . import walls as W


# ----------------------------------------------------------------------------- line evidence

class LineEvidence:
    """Sampling helpers on a lightly smoothed darkness image."""

    def __init__(self, D, k=1.0):
        self.G = cv2.GaussianBlur(D, (0, 0), 1.2)
        self.k = float(k)      # line response scale of this scan relative to the reference scans

    def sample(self, xs, ys):
        return ndi.map_coordinates(self.G, [ys, xs], order=1, mode='constant', cval=0.0)

    def line_score(self, x0, y0, x1, y1, off=3.5, trim=0.08, n=None, stat='median'):
        """Center-surround darkness along the segment (x0,y0)-(x1,y1)."""
        L = float(np.hypot(x1 - x0, y1 - y0))
        if L < 2:
            return 0.0
        if n is None:
            n = max(8, int(L / 2.0))
        t = np.linspace(trim, 1.0 - trim, n)
        xs = x0 + (x1 - x0) * t
        ys = y0 + (y1 - y0) * t
        nx, ny = -(y1 - y0) / L, (x1 - x0) / L
        c = self.sample(xs, ys)
        o1 = self.sample(xs + off * nx, ys + off * ny)
        o2 = self.sample(xs - off * nx, ys - off * ny)
        s = c - 0.5 * (o1 + o2)
        if stat == 'median':
            return float(np.median(s))
        if stat == 'frac':
            return s
        return float(np.mean(s))

    def best_line_score(self, x0, y0, x1, y1, jitter=2, **kw):
        best = -1e9
        L = float(np.hypot(x1 - x0, y1 - y0))
        nx, ny = -(y1 - y0) / max(L, 1e-6), (x1 - x0) / max(L, 1e-6)
        for d in range(-jitter, jitter + 1):
            best = max(best, self.line_score(x0 + d * nx, y0 + d * ny, x1 + d * nx, y1 + d * ny, **kw))
        return best

    def line_profile(self, x0, y0, x1, y1, jitter=2, trim=0.0):
        """Per-sample center-surround response along the segment, best lateral offset."""
        L = float(np.hypot(x1 - x0, y1 - y0))
        if L < 2:
            return np.zeros(0)
        nx, ny = -(y1 - y0) / L, (x1 - x0) / L
        best = None
        for d in range(-jitter, jitter + 1):
            s = self.line_score(x0 + d * nx, y0 + d * ny, x1 + d * nx, y1 + d * ny, trim=trim, stat='frac')
            if best is None or np.median(s) > np.median(best):
                best = s
        return best

    def circle_score(self, cx, cy, r, n=64):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best = -1e9
        for dr in (-2.0, -1.0, 0.0, 1.0, 2.0):
            rr = r + dr
            xs = cx + rr * np.cos(th)
            ys = cy + rr * np.sin(th)
            c = self.sample(xs, ys)
            o1 = self.sample(cx + (rr + 3.5) * np.cos(th), cy + (rr + 3.5) * np.sin(th))
            o2 = self.sample(cx + (rr - 3.5) * np.cos(th), cy + (rr - 3.5) * np.sin(th))
            best = max(best, float(np.median(c - 0.5 * (o1 + o2))))
        return best


# ----------------------------------------------------------------------------- fixture segments

def fixture_segments(D, ppf, ev, resp_min=30.0, resp_max=95.0):
    """Axis-aligned line segments of fixture pen weight, collinear fragments merged.

    Candidate lines come from the ridge detector; each is split where its center-surround
    response leaves the fixture range (hatch lines are much weaker, walls much stronger).
    """
    resp_min *= ev.k
    resp_max *= ev.k
    G = ev.G
    H, Wd = G.shape
    out = {}
    for axis in (0, 1):
        segs = W.extract_segments(D, axis, thrS=40.0, thrR=25.0, minlen=8)
        keep = []
        for sg in segs:
            if sg['b'] - sg['a'] < 0.3 * ppf:
                continue
            p = sg['pos']
            idx = np.arange(sg['a'], sg['b'] + 1)
            best = None
            for dp in (-1.0, -0.5, 0.0, 0.5, 1.0):
                q = p + dp
                if axis == 0:
                    c = ndi.map_coordinates(G, [np.full(len(idx), q), idx], order=1)
                    o1 = ndi.map_coordinates(G, [np.full(len(idx), q - 3.5), idx], order=1)
                    o2 = ndi.map_coordinates(G, [np.full(len(idx), q + 3.5), idx], order=1)
                else:
                    c = ndi.map_coordinates(G, [idx, np.full(len(idx), q)], order=1)
                    o1 = ndi.map_coordinates(G, [idx, np.full(len(idx), q - 3.5)], order=1)
                    o2 = ndi.map_coordinates(G, [idx, np.full(len(idx), q + 3.5)], order=1)
                r = c - 0.5 * (o1 + o2)
                if best is None or np.median(r) > np.median(best[0]):
                    best = (r, q)
            r, q = best
            r = ndi.median_filter(r, 5)
            ok = (r >= resp_min) & (r <= resp_max)
            # split into runs of ok samples, bridging tiny gaps
            k = 0
            n = len(ok)
            while k < n:
                if not ok[k]:
                    k += 1
                    continue
                j = k
                gap = 0
                last = k
                while j + 1 < n:
                    if ok[j + 1]:
                        last = j + 1
                        gap = 0
                    else:
                        gap += 1
                        if gap > 3:
                            break
                    j += 1
                a0, b0 = int(idx[k]), int(idx[last])
                if b0 - a0 >= 0.3 * ppf:
                    keep.append(dict(pos=float(q), a=a0, b=b0, resp=float(np.median(r[k:last + 1]))))
                k = last + 1
        out[axis] = merge_collinear(keep, ppf)
    return out


def merge_collinear(segs, ppf, pos_tol=1.6, gap=0.6):
    segs = sorted(segs, key=lambda s: (s['pos'], s['a']))
    groups = []
    for s in segs:
        for g in groups:
            if abs(g['pos'] - s['pos']) <= pos_tol:
                g['items'].append(s)
                g['pos'] = np.average([q['pos'] for q in g['items']], weights=[q['b'] - q['a'] + 1 for q in g['items']])
                break
        else:
            groups.append(dict(pos=s['pos'], items=[s]))
    out = []
    for g in groups:
        items = sorted(g['items'], key=lambda q: q['a'])
        cur = None
        for q in items:
            if cur is not None and q['a'] <= cur['b'] + gap * ppf:
                cur['b'] = max(cur['b'], q['b'])
                cur['parts'].append(q)
            else:
                if cur is not None:
                    out.append(cur)
                cur = dict(pos=float(g['pos']), a=q['a'], b=q['b'], parts=[q])
        if cur is not None:
            out.append(cur)
    for s in out:
        s['cov'] = sum(q['b'] - q['a'] + 1 for q in s['parts']) / float(s['b'] - s['a'] + 1)
    return out


# ----------------------------------------------------------------------------- geometry helpers

def to_xy(axis, along, across):
    """Convert (along, across) of a run to image (x, y).  axis=1: run is vertical (along = y)."""
    if axis == 1:
        return across, along
    return along, across


# ----------------------------------------------------------------------------- gondolas

def find_gondolas(D, ppf, segs, ev):
    """Find 4'-0" deep gondola runs and classify their cells."""
    runs = []
    depth = 4.0 * ppf
    for axis in (1, 0):            # axis of the long edges (1 = vertical lines -> vertical run)
        cand = [s for s in segs[axis] if (s['b'] - s['a']) >= 3.5 * ppf]
        cand.sort(key=lambda s: s['pos'])
        used = set()
        for i, A in enumerate(cand):
            for j in range(i + 1, len(cand)):
                B = cand[j]
                d = B['pos'] - A['pos']
                if d < depth - 0.35 * ppf:
                    continue
                if d > depth + 0.35 * ppf:
                    break
                lo = max(A['a'], B['a'])
                hi = min(A['b'], B['b'])
                if hi - lo < 3.5 * ppf:
                    continue
                run = analyse_run(ev, axis, A['pos'], B['pos'], lo, hi, ppf)
                if run is not None:
                    key = (axis, round(A['pos']), round(B['pos']))
                    if key in used:
                        continue
                    used.add(key)
                    runs.append(run)
    # de-duplicate overlapping runs
    final = []
    for r in sorted(runs, key=lambda r: -r['score']):
        dup = False
        for f in final:
            if f['axis'] == r['axis'] and abs(f['c1'] - r['c1']) < 0.5 * ppf and \
               min(f['b'], r['b']) - max(f['a'], r['a']) > 0.5 * (r['b'] - r['a']):
                dup = True
                break
        if not dup:
            final.append(r)
    # runs never overlap: a "run" lying across another one is made of its section lines
    def box(r):
        a, b = r['a'], r['b']
        x0, y0 = to_xy(r['axis'], a, r['c1'])
        x1, y1 = to_xy(r['axis'], b, r['c2'])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    keep = []
    for r in final:
        bx = box(r)
        area = (bx[2] - bx[0]) * (bx[3] - bx[1])
        drop = False
        for o in final:
            if o is r or o['axis'] == r['axis'] or (o['b'] - o['a']) <= (r['b'] - r['a']):
                continue
            ob = box(o)
            iw = min(bx[2], ob[2]) - max(bx[0], ob[0])
            ih = min(bx[3], ob[3]) - max(bx[1], ob[1])
            if iw > 0 and ih > 0 and iw * ih > 0.5 * area:
                drop = True
                break
        if not drop:
            keep.append(r)
    return keep


def analyse_run(ev, axis, e1, e2, lo, hi, ppf):
    """Cross lines along a candidate run between long edges at e1 < e2 (across coords)."""
    mid = 0.5 * (e1 + e2)

    def resp_line(across_pos, along_lo, along_hi):
        x0, y0 = to_xy(axis, along_lo, across_pos)
        x1, y1 = to_xy(axis, along_hi, across_pos)
        return ev.line_profile(x0, y0, x1, y1, jitter=2, trim=0.02)
    spine = resp_line(mid, lo, hi)
    edge1 = resp_line(e1, lo, hi)
    edge2 = resp_line(e2, lo, hi)
    if np.median(edge1) < 12 * ev.k or np.median(edge2) < 12 * ev.k:
        return None
    if np.mean(spine > 12 * ev.k) < 0.45:
        return None
    # cross line profile: response of lines perpendicular to the run, in both halves
    q1a, q1b = e1 + 4, mid - 4
    q2a, q2b = mid + 4, e2 - 4
    ext = int(0.6 * ppf)
    alo, ahi = int(lo - ext), int(hi + ext)
    prof1 = cross_profile(ev, axis, alo, ahi, q1a, q1b)
    prof2 = cross_profile(ev, axis, alo, ahi, q2a, q2b)
    prof = np.minimum(prof1, prof2)
    thr = 15.0 * ev.k
    pk = [k for k in range(1, len(prof) - 1) if prof[k] > prof[k - 1] and prof[k] >= prof[k + 1] and prof[k] > thr]
    # non-max suppression within 0.4 ft
    pk.sort(key=lambda k: -prof[k])
    chosen = []
    for k in pk:
        if all(abs(k - c) > 0.4 * ppf for c in chosen):
            chosen.append(k)
    lines = sorted(alo + k for k in chosen)
    if len(lines) < 2:
        return None
    cells = []
    for u, v in zip(lines, lines[1:]):
        L = (v - u) / ppf
        # diagonal check (endcaps): either diagonal of the cell
        xA, yA = to_xy(axis, u, e1)
        xB, yB = to_xy(axis, v, e2)
        xC, yC = to_xy(axis, u, e2)
        xD, yD = to_xy(axis, v, e1)
        d1 = ev.best_line_score(xA, yA, xB, yB, jitter=2, trim=0.15)
        d2 = ev.best_line_score(xC, yC, xD, yD, jitter=2, trim=0.15)
        sp = resp_line(mid, u + 3, v - 3)
        spf = float(np.mean(sp > 12 * ev.k)) if len(sp) else 0.0
        e1f = float(np.mean(resp_line(e1, u + 3, v - 3) > 12 * ev.k))
        e2f = float(np.mean(resp_line(e2, u + 3, v - 3) > 12 * ev.k))
        cells.append(dict(a=u, b=v, L=L, diag=max(d1, d2) / ev.k, spine=spf, edges=min(e1f, e2f)))
    return dict(axis=axis, c1=e1, c2=e2, a=lines[0], b=lines[-1], lines=lines, cells=cells,
                score=float(np.median(edge1) + np.median(edge2)))


def cross_profile(ev, axis, alo, ahi, qa, qb):
    """Mean center-surround response of lines perpendicular to the run at each along position."""
    G = ev.G
    alo = max(alo, 3)
    if axis == 1:
        ahi = min(ahi, G.shape[0] - 4)
        x0, x1 = int(round(qa)), int(round(qb))
        band = G[alo - 3:ahi + 4, x0:x1 + 1].mean(1)
    else:
        ahi = min(ahi, G.shape[1] - 4)
        y0, y1 = int(round(qa)), int(round(qb))
        band = G[y0:y1 + 1, alo - 3:ahi + 4].mean(0)
    c = band[3:-3]
    s = 0.5 * (band[0:-6] + band[6:])
    out = np.zeros(ahi - alo + 1)
    n = min(len(out), len(c))
    out[:n] = (c - s)[:n]
    return out


def _split_length(L, unit, others=(4.0, 3.0)):
    """Number of whole sections making up a cell of length L (a cross line was missed)."""
    k = int(round(L / unit))
    if k >= 2 and abs(L - k * unit) <= 0.25 + 0.05 * k:
        return {unit: k}
    best = None
    for a in range(0, int(L // 4.0) + 2):
        for b in range(0, int(L // 3.0) + 2):
            if a + b < 2:
                continue
            err = abs(L - 4.0 * a - 3.0 * b)
            if err <= 0.3 and (best is None or err < best[0]):
                best = (err, {4.0: a, 3.0: b})
    return best[1] if best else {}


def count_gondolas(runs, ppf):
    """Count 4'-0" and 3'-0" sections and endcaps of the detected runs.

    Cells are walked in order; a cell that is too short to be a section is joined with its
    neighbours (a spurious cross line), one that is too long is split (a missed cross line).
    """
    n4 = n3 = nend = 0
    for r in runs:
        cells = [c for c in r['cells'] if c['edges'] >= 0.5]
        for c in r['cells']:
            c['kind'] = None
        lengths = [c['L'] for c in cells if c['spine'] >= 0.5 and 2.5 <= c['L'] <= 4.6]
        unit = 4.0 if sum(1 for L in lengths if L >= 3.5) >= sum(1 for L in lengths if L < 3.5) else 3.0
        i = 0
        while i < len(cells):
            c = cells[i]
            L = c['L']
            if c['diag'] > 15 and c['spine'] < 0.5 and 0.9 <= L <= 2.4:
                nend += 1
                c['kind'] = 'endcap'
                i += 1
                continue
            if c['spine'] < 0.5:
                i += 1
                continue
            if 3.5 <= L <= 4.6:
                n4 += 1
                c['kind'] = '4ft'
                i += 1
                continue
            if 2.5 <= L < 3.5:
                n3 += 1
                c['kind'] = '3ft'
                i += 1
                continue
            if L > 4.6:
                parts = _split_length(L, unit)
                n4 += parts.get(4.0, 0)
                n3 += parts.get(3.0, 0)
                c['kind'] = 'multi'
                i += 1
                continue
            # too short: join with the following spine cells
            tot = L
            done = False
            for j in range(i + 1, min(i + 4, len(cells))):
                if cells[j]['spine'] < 0.5 or abs(cells[j]['a'] - cells[j - 1]['b']) > 3:
                    break
                tot += cells[j]['L']
                if abs(tot - unit) <= 0.3 or abs(tot - 4.0) <= 0.3 or abs(tot - 3.0) <= 0.3:
                    if abs(tot - 4.0) <= abs(tot - 3.0):
                        n4 += 1
                    else:
                        n3 += 1
                    for q in range(i, j + 1):
                        cells[q]['kind'] = 'joined'
                    i = j + 1
                    done = True
                    break
            if not done:
                i += 1
    return n4, n3, nend


# ----------------------------------------------------------------------------- cooler doors

def find_cooler_doors(D, ppf, segs, ev):
    """Rows of reach-in cooler doors: a strip ~0.6 ft deep divided into doors with a diagonal."""
    strips = []
    for axis in (0, 1):          # axis of the strip's long lines (0 = horizontal strip)
        cand = [s for s in segs[axis] if (s['b'] - s['a']) >= 1.5 * ppf]
        cand.sort(key=lambda s: s['pos'])
        for i, A in enumerate(cand):
            for j in range(i + 1, len(cand)):
                B = cand[j]
                d = B['pos'] - A['pos']
                if d < 0.40 * ppf:
                    continue
                if d > 0.85 * ppf:
                    break
                lo = max(A['a'], B['a'])
                hi = min(A['b'], B['b'])
                if hi - lo < 1.5 * ppf:
                    continue
                st = analyse_strip(ev, axis, A['pos'], B['pos'], lo, hi, ppf)
                if st is not None:
                    strips.append(st)
    # de-duplicate (same strip found from different segment pairs)
    final = []
    for s in sorted(strips, key=lambda s: -len(s['doors'])):
        dup = False
        for f in final:
            if f['axis'] == s['axis'] and abs(f['c1'] - s['c1']) < 3 and abs(f['c2'] - s['c2']) < 3 and \
               min(f['b'], s['b']) - max(f['a'], s['a']) > 0:
                # merge door lists
                for dd in s['doors']:
                    if all(abs(dd[0] - e[0]) > 0.5 * ppf for e in f['doors']):
                        f['doors'].append(dd)
                dup = True
                break
        if not dup:
            final.append(s)
    return final


def analyse_strip(ev, axis, e1, e2, lo, hi, ppf):
    depth = e2 - e1
    mid = 0.5 * (e1 + e2)
    # both long lines must be present over the overlap
    def prof_along(across_pos, a0, a1):
        x0, y0 = to_xy(axis, a0, across_pos)
        x1, y1 = to_xy(axis, a1, across_pos)
        return ev.line_profile(x0, y0, x1, y1, jitter=1, trim=0.0)
    p1 = prof_along(e1, lo, hi)
    p2 = prof_along(e2, lo, hi)
    if len(p1) == 0 or np.median(p1) < 10 * ev.k or np.median(p2) < 10 * ev.k:
        return None
    ext = int(0.3 * ppf)
    alo, ahi = int(lo - ext), int(hi + ext)
    pa = cross_profile(ev, axis, alo, ahi, e1 + 2.5, mid - 1.0)
    pb = cross_profile(ev, axis, alo, ahi, mid + 1.0, e2 - 2.5)
    prof = np.minimum(pa, pb)
    thr = 14.0 * ev.k
    pk = [k for k in range(1, len(prof) - 1) if prof[k] > prof[k - 1] and prof[k] >= prof[k + 1] and prof[k] > thr]
    pk.sort(key=lambda k: -prof[k])
    chosen = []
    for k in pk:
        if all(abs(k - c) > 0.35 * ppf for c in chosen):
            chosen.append(k)
    divs = sorted(alo + k for k in chosen)
    def door_cell(u, v):
        """Diagonal score of a door cell between dividers u and v (0 if not bounded)."""
        xA, yA = to_xy(axis, u, e1)
        xB, yB = to_xy(axis, v, e2)
        xC, yC = to_xy(axis, u, e2)
        xD, yD = to_xy(axis, v, e1)
        d1 = ev.best_line_score(xA, yA, xB, yB, jitter=2, trim=0.15)
        d2 = ev.best_line_score(xC, yC, xD, yD, jitter=2, trim=0.15)
        # the long edges must bound this cell
        f1 = prof_along(e1, u + 2, v - 2)
        f2 = prof_along(e2, u + 2, v - 2)
        if len(f1) == 0:
            return 0.0
        cov = min(float(np.mean(f1 > 10 * ev.k)), float(np.mean(f2 > 10 * ev.k)))
        if max(d1, d2) >= 15 * ev.k and cov >= 0.6:
            return max(d1, d2)
        return 0.0

    # door width of this strip from the clean cells between neighbouring dividers
    first = []
    for u, v in zip(divs, divs[1:]):
        W_ = (v - u) / ppf
        if 1.2 <= W_ <= 4.0:
            s = door_cell(u, v)
            if s > 0:
                first.append(W_)
    if len(first) < 2:
        return None
    med = float(np.median(first))
    # walk along the dividers; a span holding a whole number of doors (each with its own
    # diagonal) is accepted, which skips spurious dividers (specks) and recovers missed ones
    doors = []
    i = 0
    n = len(divs)
    while i < n - 1:
        accepted = False
        for j in range(i + 1, n):
            span = (divs[j] - divs[i]) / ppf
            if span > 6.5 * med:
                break
            k = int(round(span / med))
            if k < 1 or abs(span - k * med) > 0.3 + 0.1 * k:
                continue
            step = (divs[j] - divs[i]) / k
            parts = []
            for m in range(k):
                s = door_cell(divs[i] + m * step, divs[i] + (m + 1) * step)
                if s <= 0:
                    break
                parts.append((divs[i] + (m + 0.5) * step, step / ppf, s))
            if len(parts) == k:
                doors.extend(parts)
                i = j
                accepted = True
                break
        if not accepted:
            i += 1
    if len(doors) < 2:
        return None
    return dict(axis=axis, c1=e1, c2=e2, a=divs[0], b=divs[-1], doors=doors, width=med)


# ----------------------------------------------------------------------------- generic ladders

def find_cells(ev, segs, ppf, widths, tol=0.25, min_overlap=1.0, cross_thr=18.0):
    """Cells bounded by two parallel fixture lines (spacing in `widths`, ft) and cross lines."""
    cross_thr *= ev.k
    cells = []
    for axis in (0, 1):
        cand = [s for s in segs[axis] if (s['b'] - s['a']) >= min_overlap * ppf]
        cand.sort(key=lambda s: s['pos'])
        for i, A in enumerate(cand):
            for j in range(i + 1, len(cand)):
                B = cand[j]
                d = (B['pos'] - A['pos']) / ppf
                if d > max(widths) + tol:
                    break
                if not any(abs(d - w) <= tol for w in widths):
                    continue
                lo = max(A['a'], B['a'])
                hi = min(A['b'], B['b'])
                if hi - lo < min_overlap * ppf:
                    continue
                cells.extend(ladder_cells(ev, axis, A['pos'], B['pos'], lo, hi, ppf, cross_thr))
    return cells


def ladder_cells(ev, axis, e1, e2, lo, hi, ppf, cross_thr):
    mid = 0.5 * (e1 + e2)
    ext = int(0.6 * ppf)
    alo, ahi = int(lo - ext), int(hi + ext)
    pa = cross_profile(ev, axis, alo, ahi, e1 + 3, mid - 2)
    pb = cross_profile(ev, axis, alo, ahi, mid + 2, e2 - 3)
    prof = np.minimum(pa, pb)
    pk = [k for k in range(1, len(prof) - 1) if prof[k] > prof[k - 1] and prof[k] >= prof[k + 1] and prof[k] > cross_thr]
    pk.sort(key=lambda k: -prof[k])
    chosen = []
    for k in pk:
        if all(abs(k - c) > 0.35 * ppf for c in chosen):
            chosen.append(k)
    lines = sorted(alo + k for k in chosen)
    # cells between neighbouring cross lines, and also spans of a few cells so that a
    # spurious cross line (a speck, a crossing grid line) cannot hide a fixture
    pairs = [(lines[i], lines[j]) for i in range(len(lines)) for j in range(i + 1, min(i + 4, len(lines)))]
    out = []
    for u, v in pairs:
        L = (v - u) / ppf
        if L < 0.3 or L > 14.0:
            continue
        def edge_cov(across):
            x0, y0 = to_xy(axis, u + 2, across)
            x1, y1 = to_xy(axis, v - 2, across)
            p = ev.line_profile(x0, y0, x1, y1, jitter=1)
            return float(np.mean(p > 15 * ev.k)) if len(p) else 0.0
        cov = min(edge_cov(e1), edge_cov(e2))
        if cov < 0.75:
            continue
        xA, yA = to_xy(axis, u, e1)
        xB, yB = to_xy(axis, v, e2)
        xC, yC = to_xy(axis, u, e2)
        xD, yD = to_xy(axis, v, e1)
        d1 = ev.best_line_score(xA, yA, xB, yB, jitter=2, trim=0.12)
        d2 = ev.best_line_score(xC, yC, xD, yD, jitter=2, trim=0.12)
        cx, cy = to_xy(axis, 0.5 * (u + v), mid)
        x0, y0 = to_xy(axis, u, e1)
        x1, y1 = to_xy(axis, v, e2)
        out.append(dict(axis=axis, x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1),
                        w=(e2 - e1) / ppf, L=L, d1=d1 / ev.k, d2=d2 / ev.k, cx=cx, cy=cy, e1=e1, e2=e2, u=u, v=v))
    return out


def dedupe_cells(cells, ppf, tol=0.4):
    out = []
    for c in cells:
        dup = False
        for o in out:
            if abs(o['cx'] - c['cx']) < tol * ppf and abs(o['cy'] - c['cy']) < tol * ppf and \
               abs((o['x1'] - o['x0']) - (c['x1'] - c['x0'])) < tol * ppf and \
               abs((o['y1'] - o['y0']) - (c['y1'] - c['y0'])) < tol * ppf:
                dup = True
                break
        if not dup:
            out.append(c)
    return out


def find_produce_tables(ev, segs, ppf):
    cells = find_cells(ev, segs, ppf, widths=(4.0, 5.0), tol=0.3, min_overlap=3.0)
    tabs = []
    for c in cells:
        dims = sorted([c['w'], c['L']])
        if abs(dims[0] - 4.0) <= 0.35 and abs(dims[1] - 5.0) <= 0.35 and min(c['d1'], c['d2']) >= 15:
            tabs.append(c)
    return dedupe_cells(tabs, ppf)


def find_self_checkouts(ev, segs, ppf):
    cells = find_cells(ev, segs, ppf, widths=(3.0,), tol=0.3, min_overlap=2.2)
    out = []
    for c in cells:
        if abs(c['L'] - 3.0) > 0.35:
            continue
        best = max(ev.circle_score(c['cx'], c['cy'], r * ppf) for r in (0.6, 0.7, 0.8, 0.9, 1.0))
        best /= ev.k
        c['circle'] = best
        if best >= 15:
            out.append(c)
    return dedupe_cells(out, ppf)


def find_checkouts(ev, segs, ppf):
    """Checkout stand: 2'-6" wide belt box with an X-marked bagging box at one end."""
    cells = find_cells(ev, segs, ppf, widths=(2.5,), tol=0.3, min_overlap=2.0)
    xboxes = [c for c in cells if 2.2 <= c['L'] <= 4.0 and min(c['d1'], c['d2']) >= 15]
    belts = [c for c in cells if c['L'] >= 5.0 and max(c['d1'], c['d2']) < 15]
    out = []
    for xb in xboxes:
        ok = False
        for bt in belts:
            if bt['axis'] != xb['axis'] or abs(bt['e1'] - xb['e1']) > 3 or abs(bt['e2'] - xb['e2']) > 3:
                continue
            if abs(bt['u'] - xb['v']) <= 4 or abs(bt['v'] - xb['u']) <= 4:
                ok = True
                break
        if ok:
            out.append(xb)
    return dedupe_cells(out, ppf)


def line_scale(ev, runs, idxs, off, ref=110.0):
    """Line response of the exterior wall outer faces relative to the reference scans."""
    x0, y0 = off
    vals = []
    for i in idxs:
        r = runs[i]
        if r.get('type') != 'EXT' or not r.get('ext_side'):
            continue
        face = r['p1'] if r['ext_side'] < 0 else r['p2']
        lo = max(r['a'], r['sa']) + 30
        hi = min(r['b'], r['sb']) - 30
        if hi - lo < 50:
            continue
        if r['axis'] == 0:
            p = ev.line_profile(lo - x0, face - y0, hi - x0, face - y0, jitter=1)
        else:
            p = ev.line_profile(face - x0, lo - y0, face - x0, hi - y0, jitter=1)
        if len(p):
            vals.append(float(np.median(p)))
    if not vals:
        return 1.0
    return float(np.clip(np.median(vals) / ref, 0.5, 1.5))
