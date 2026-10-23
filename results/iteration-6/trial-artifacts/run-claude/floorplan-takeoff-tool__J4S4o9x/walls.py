"""Wall detection: pairs of parallel dark outlines with a wall-type fill."""
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d, median_filter

from canon import PPF

# nominal thickness (ft) by type, per the partition legend
NOMINAL_T = {'EXT': 8.0 / 12, 'P3': 8.0 / 12, 'P2': 7.25 / 12, 'P1': 4.875 / 12}

OPEN, P1, P2, CMU = 4, 1, 2, 3
TYPE_NAME = {P1: 'P1', P2: 'P2', CMU: 'CMU', OPEN: 'OPEN'}


def _trace_cc(m, wt, x, y, minlen):
    """Split a thick line component into line tracks.
    m: bool mask (h x w) of the component, wt: darkness weights (same shape,
    rows padded by 0).  Returns [[yc, x0, x1, h]] in absolute coordinates."""
    h, w = m.shape
    rows = np.arange(h)
    # per column: vertical runs -> centroid
    tracks = []      # each: [sum_c, n, x_start, x_last, last_c]
    active = []
    for j in range(w):
        col = m[:, j]
        if not col.any():
            for t in active:
                tracks.append(t)
            active = []
            continue
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0] - 1
        cents = []
        for s, e in zip(starts, ends):
            ww = wt[s:e + 1, j]
            cents.append(float((rows[s:e + 1] * ww).sum() / max(ww.sum(), 1e-6)) if ww.sum() > 0 else (s + e) / 2.0)
        new_active = []
        used = set()
        for c in cents:
            best = None
            for k, t in enumerate(active):
                if k in used:
                    continue
                mu = t[0] / t[1]
                if abs(mu - c) <= 1.0 and abs(t[4] - c) <= 1.0 and (best is None or abs(mu - c) < abs(active[best][0] / active[best][1] - c)):
                    best = k
            if best is None:
                new_active.append([c, 1, j, j, c])
            else:
                t = active[best]
                used.add(best)
                t[0] += c; t[1] += 1; t[3] = j; t[4] = c
                new_active.append(t)
        for k, t in enumerate(active):
            if k not in used:
                tracks.append(t)
        active = new_active
    tracks.extend(active)
    out = []
    for t in tracks:
        if t[3] - t[2] + 1 >= minlen:
            out.append([y + t[0] / t[1], x + t[2], x + t[3], 3])
    return out


def line_segments(g, thr=135, minlen=15):
    """Horizontal dark line segments: [yc, x0, x1, h]."""
    dark = (g < thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (minlen, 1))
    hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    n, lab, st, cen = cv2.connectedComponentsWithStats(hl, 8)
    segs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h > 40:
            continue
        ys = np.arange(max(0, y - 2), min(g.shape[0], y + h + 2))
        blk = 255.0 - g[ys[0]:ys[-1] + 1, x:x + w].astype(np.float32)
        base = np.minimum(blk[0], blk[-1])[None, :]
        wt = np.clip(blk - base, 0, None)
        if h <= 4:
            prof = wt.mean(axis=1)
            yc = (ys * prof).sum() / max(prof.sum(), 1e-6)
            segs.append([float(yc), int(x), int(x + w - 1), int(h)])
            continue
        m = np.zeros((len(ys), w), bool)
        m[(y - ys[0]):(y - ys[0]) + h] = (lab[y:y + h, x:x + w] == i)
        # weights only on mask pixels (+1 px) so neighbouring structures don't bias
        mm = cv2.dilate(m.astype(np.uint8), np.ones((3, 1), np.uint8)).astype(bool)
        segs.extend(_trace_cc(m | (mm & False), np.where(mm, wt, 0), int(x), int(ys[0]), minlen))
    segs.sort()
    return _bridge_faint(g, segs)


def _bridge_faint(g, segs, max_gap=90, faint=195):
    """Join collinear segments across short stretches where the same line is
    present but lighter than the detection threshold."""
    if not segs:
        return segs
    H = g.shape[0]
    segs = sorted(segs, key=lambda s: (round(s[0]), s[1]))
    out = []
    used = [False] * len(segs)
    order = sorted(range(len(segs)), key=lambda i: (segs[i][0], segs[i][1]))
    # bucket by integer row for neighbour search
    from collections import defaultdict
    buck = defaultdict(list)
    for i, s in enumerate(segs):
        buck[int(round(s[0]))].append(i)
    for i in order:
        if used[i]:
            continue
        cur = list(segs[i])
        used[i] = True
        grown = True
        while grown:
            grown = False
            r0 = int(round(cur[0]))
            cands = [j for rr in (r0 - 1, r0, r0 + 1) for j in buck.get(rr, []) if not used[j] and abs(segs[j][0] - cur[0]) <= 1.0]
            for j in sorted(cands, key=lambda j: segs[j][1]):
                s = segs[j]
                if s[1] > cur[2] and s[1] - cur[2] <= max_gap:
                    gap0, gap1 = cur[2] + 1, s[1]
                elif s[2] < cur[1] and cur[1] - s[2] <= max_gap:
                    gap0, gap1 = s[2] + 1, cur[1]
                elif s[1] <= cur[2] and s[2] >= cur[1]:
                    gap0 = gap1 = None
                else:
                    continue
                if gap0 is not None and gap1 > gap0:
                    yy = int(round(cur[0]))
                    band = g[max(yy - 1, 0):min(yy + 2, H), gap0:gap1].astype(np.float32)
                    if band.size == 0 or band.min(axis=0).mean() > faint:
                        continue
                w0 = cur[2] - cur[1] + 1
                w1 = s[2] - s[1] + 1
                cur[0] = (cur[0] * w0 + s[0] * w1) / (w0 + w1)
                cur[1] = min(cur[1], s[1])
                cur[2] = max(cur[2], s[2])
                used[j] = True
                grown = True
        out.append(cur)
    out.sort()
    return out


def _classify_cols(inner, t):
    m = uniform_filter1d(inner.mean(axis=0), 7)
    sd = uniform_filter1d(inner.std(axis=0), 7)
    c = np.zeros(inner.shape[1], np.int8)
    if 7.5 <= t <= 12.5:
        c[(m >= 196) & (m <= 229) & (sd < 14)] = P1
    elif 13.5 <= t <= 18.8:
        c[(m >= 130) & (m <= 196) & (sd < 22)] = P2
        if inner.shape[0] >= 2 and inner.shape[1] >= 3:
            dh = np.abs(np.diff(inner, axis=1))
            dh = np.concatenate([dh, dh[:, -1:]], axis=1).mean(axis=0)
            a = np.abs(inner[1:, 1:] - inner[:-1, :-1])
            b = np.abs(inner[1:, :-1] - inner[:-1, 1:])
            dd = np.minimum(a, b)
            dd = np.concatenate([dd, dd[:, -1:]], axis=1).mean(axis=0)
            dh = uniform_filter1d(dh, 11)
            dd = uniform_filter1d(dd, 11)
            c[(m >= 188) & (m <= 237) & (sd >= 10) & (dh > 5) & (dh > 1.35 * dd)] = CMU
    if 7.5 <= t <= 18.8:
        c[(m >= 233) & (sd < 10)] = OPEN
    return c


def _runs01(c, minlen):
    out = []
    n = len(c)
    i = 0
    while i < n:
        if c[i] == 0:
            i += 1
            continue
        j = i
        while j < n and c[j] == c[i]:
            j += 1
        if j - i >= minlen:
            out.append((i, j - 1, int(c[i])))
        i = j
    return out


def bands(g, segs, tmin=7.5, tmax=18.8, minlen=10):
    out = []
    for i, A in enumerate(segs):
        for B in segs[i + 1:]:
            dy = B[0] - A[0]
            if dy > tmax:
                break
            if dy < tmin:
                continue
            x0 = max(A[1], B[1])
            x1 = min(A[2], B[2])
            if x1 - x0 < minlen:
                continue
            ya = int(np.ceil(A[0] + 2.5))
            yb = int(np.floor(B[0] - 2.5))
            if yb < ya:
                continue
            inner = g[ya:yb + 1, x0:x1 + 1].astype(np.float32)
            c = _classify_cols(inner, dy)
            for r0, r1, k in _runs01(c, minlen):
                out.append(dict(c0=A[0], c1=B[0], s0=x0 + r0, s1=x0 + r1, t=dy, typ=k))
    return out


def detect_bands(g):
    H = bands(g, line_segments(g))
    gt = np.ascontiguousarray(g.T)
    V = bands(gt, line_segments(gt))
    return H, V


class Run:
    def __init__(self, orient, typ, c, t, s0, s1, pieces):
        self.orient, self.typ, self.c, self.t = orient, typ, c, t
        self.s0, self.s1 = s0, s1
        self.pieces = pieces          # list of (s0, s1, kind) kind: wall type or OPEN
        self.e0, self.e1 = float(s0), float(s1)
        self.j0 = self.j1 = None      # perpendicular run met at each end
        self.name = None

    def length(self):
        return (self.e1 - self.e0) / PPF

    def __repr__(self):
        return '%s %s c=%.1f [%d..%d] e=[%.1f..%.1f] t=%.1f' % (self.orient, self.name or TYPE_NAME.get(self.typ), self.c, self.s0, self.s1, self.e0, self.e1, self.t)


def group_runs(bs, orient, gap_max=30 * PPF):
    """Group collinear bands of one wall type into runs, bridging openings."""
    walls = [b for b in bs if b['typ'] != OPEN]
    opens = [b for b in bs if b['typ'] == OPEN]
    walls.sort(key=lambda b: ((b['c0'] + b['c1']) / 2, b['s0']))
    groups = []
    for b in walls:
        c = (b['c0'] + b['c1']) / 2
        for gr in groups:
            if gr['typ'] == b['typ'] and abs(gr['c'] - c) < 2.0 and abs(gr['t'] - b['t']) < 2.5:
                gr['items'].append(b)
                n = len(gr['items'])
                gr['c'] = gr['c'] + (c - gr['c']) / n
                break
        else:
            groups.append(dict(typ=b['typ'], c=c, t=b['t'], items=[b]))
    runs = []
    for gr in groups:
        its = sorted(gr['items'], key=lambda b: b['s0'])
        # openings collinear with this group
        ops = [o for o in opens if abs((o['c0'] + o['c1']) / 2 - gr['c']) < 2.5 and abs(o['t'] - gr['t']) < 3.0]
        cur = None
        gmax = gap_max * 1.5 if gr['typ'] == CMU else gap_max   # masonry walls are long
        for b in its:
            if cur is not None and b['s0'] - cur[-1]['s1'] <= gmax:
                cur.append(b)
            else:
                if cur:
                    runs.append(_mk_run(cur, orient, gr['typ'], ops))
                cur = [b]
        if cur:
            runs.append(_mk_run(cur, orient, gr['typ'], ops))
    return runs


def _mk_run(items, orient, typ, ops):
    w = np.array([b['s1'] - b['s0'] + 1 for b in items], float)
    c = float(np.average([(b['c0'] + b['c1']) / 2 for b in items], weights=w))
    t = float(np.average([b['t'] for b in items], weights=w))
    s0 = min(b['s0'] for b in items)
    s1 = max(b['s1'] for b in items)
    pieces = [(b['s0'], b['s1'], typ) for b in items]
    # extend through adjacent openings at the ends
    changed = True
    while changed:
        changed = False
        for o in ops:
            if o['s1'] < s0 and s0 - o['s1'] <= 4 and o['s0'] < s0:
                pieces.append((o['s0'], o['s1'], OPEN)); s0 = o['s0']; changed = True
            elif o['s0'] > s1 and o['s0'] - s1 <= 4 and o['s1'] > s1:
                pieces.append((o['s0'], o['s1'], OPEN)); s1 = o['s1']; changed = True
    # interior openings
    for o in ops:
        if o['s0'] >= s0 and o['s1'] <= s1 and (o['s0'], o['s1'], OPEN) not in pieces:
            pieces.append((o['s0'], o['s1'], OPEN))
    pieces.sort()
    return Run(orient, typ, c, t, s0, s1, pieces)


def snap_ends(runs):
    """Measure each run between the centerlines of the walls it runs into."""
    H = [r for r in runs if r.orient == 'H']
    V = [r for r in runs if r.orient == 'V']
    for r in runs:
        r.e0, r.e1 = float(r.s0), float(r.s1)
        r.j0 = r.j1 = None
    for r in runs:
        perp = V if r.orient == 'H' else H
        hr = NOMINAL_T_PX(r) / 2
        for end in (0, 1):
            s = r.s0 if end == 0 else r.s1
            best = None
            for p in perp:
                hp = NOMINAL_T_PX(p) / 2
                tol = hr + 4
                if not (p.s0 - tol <= r.c <= p.s1 + tol):
                    continue
                d = abs(p.c - s)
                if d <= hp + 8 and (best is None or d < best[0]):
                    best = (d, p)
            if best is not None:
                if end == 0:
                    r.e0, r.j0 = best[1].c, best[1]
                else:
                    r.e1, r.j1 = best[1].c, best[1]
    # collinear abutting runs of different type without a perpendicular junction
    for r in runs:
        for q in runs:
            if q is r or q.orient != r.orient or abs(q.c - r.c) > 3:
                continue
            if r.j1 is None and 0 <= q.s0 - r.s1 <= 6:
                m = (q.s0 + r.s1) / 2.0
                r.e1 = m
                if q.j0 is None:
                    q.e0 = m


def NOMINAL_T_PX(r):
    name = r.name
    if name is None:
        name = {P1: 'P1', P2: 'P2', CMU: 'P3'}.get(r.typ, 'P1')
    return NOMINAL_T[name] * PPF


def _drop_contained(runs):
    keep = []
    for r in runs:
        bad = False
        for q in runs:
            if q is r or q.orient != r.orient or q.t < r.t + 1.5:
                continue
            if abs(r.c - q.c) <= q.t / 2 + 1 and r.s0 >= q.s0 - PPF and r.s1 <= q.s1 + PPF:
                bad = True
                break
        if not bad:
            keep.append(r)
    return keep


def find_walls(g):
    Hb, Vb = detect_bands(g)
    runs = group_runs(Hb, 'H') + group_runs(Vb, 'V')
    runs = [r for r in runs if (r.s1 - r.s0) >= 0.75 * PPF]
    # building extent from long CMU runs
    cm = [r for r in runs if r.typ == CMU and (r.s1 - r.s0) > 20 * PPF]
    hx = [r for r in cm if r.orient == 'H']
    vx = [r for r in cm if r.orient == 'V']
    if len(hx) < 2 or len(vx) < 2:
        raise RuntimeError('building outline not found')
    top = min(r.c for r in hx); bot = max(r.c for r in hx)
    left = min(r.c for r in vx); right = max(r.c for r in vx)
    m = 1.0 * PPF
    def inside(r):
        if r.orient == 'H':
            return top - m <= r.c <= bot + m and r.s1 >= left - m and r.s0 <= right + m
        return left - m <= r.c <= right + m and r.s1 >= top - m and r.s0 <= bot + m
    runs = [r for r in runs if inside(r)]
    runs = _drop_contained(runs)
    runs = split_unexplained_gaps(runs)
    return runs, (left, top, right, bot)


def split_unexplained_gaps(runs):
    """A gap between two wall pieces whose both ends meet different
    perpendicular walls is a space (e.g. a corridor), not an opening."""
    out = []
    for r in runs:
        perp = [p for p in runs if p.orient != r.orient]
        walls = sorted([p for p in r.pieces if p[2] != OPEN])
        def meets(s):
            best = None
            for p in perp:
                tol = r.t / 2 + 4
                if not (p.s0 - tol <= r.c <= p.s1 + tol):
                    continue
                d = abs(p.c - s)
                if d <= p.t / 2 + 5 and (best is None or d < best[0]):
                    best = (d, p)
            return best[1] if best else None
        groups = [[walls[0]]]
        for a, b in zip(walls[:-1], walls[1:]):
            gap = b[0] - a[1]
            pa, pb = meets(a[1]), meets(b[0])
            opened = sum(max(0, min(p[1], b[0]) - max(p[0], a[1])) for p in r.pieces if p[2] == OPEN)
            if gap > 1.2 * PPF and pa is not None and pb is not None and pa is not pb and opened < 0.3 * gap:
                groups.append([b])
            else:
                groups[-1].append(b)
        if len(groups) == 1:
            out.append(r)
            continue
        for gr in groups:
            s0 = gr[0][0]; s1 = gr[-1][1]
            pcs = [p for p in r.pieces if p[0] >= s0 - 4 and p[1] <= s1 + 4]
            out.append(Run(r.orient, r.typ, r.c, r.t, s0, s1, pcs))
    return out


def raster_runs(runs, shape, use_nominal=True, extend=True, pad=0.0, origin=(0, 0)):
    """Rasterise wall runs (full length, openings closed) into a uint8 mask
    of the given shape whose top-left pixel is at `origin` (x, y)."""
    mask = np.zeros(shape, np.uint8)
    ox, oy = origin
    for r in runs:
        h = (NOMINAL_T_PX(r) if use_nominal else r.t) / 2 + pad
        if extend:
            a0 = r.e0 - (NOMINAL_T_PX(r.j0) / 2 if r.j0 is not None else 0)
            a1 = r.e1 + (NOMINAL_T_PX(r.j1) / 2 if r.j1 is not None else 0)
            a0 = min(a0, r.s0); a1 = max(a1, r.s1)
        else:
            a0, a1 = r.s0, r.s1
        c0, c1 = r.c - h, r.c + h
        i0, i1 = int(np.ceil(c0 - 0.5)), int(np.floor(c1 - 0.5))
        k0, k1 = int(np.ceil(a0 - 0.5)), int(np.floor(a1 - 0.5))
        if r.orient == 'H':
            i0, i1, k0, k1 = i0 - oy, i1 - oy, k0 - ox, k1 - ox
            mask[max(i0, 0):max(i1 + 1, 0), max(k0, 0):max(k1 + 1, 0)] = 1
        else:
            k0, k1, i0, i1 = k0 - oy, k1 - oy, i0 - ox, i1 - ox
            mask[max(k0, 0):max(k1 + 1, 0), max(i0, 0):max(i1 + 1, 0)] = 1
    return mask


def outside_region(wall_mask):
    """Pixels connected to the border of the mask without crossing walls."""
    m = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    h, w = m.shape
    free = np.ones((h + 2, w + 2), np.uint8)
    free[1:-1, 1:-1] = (m == 0)
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(free, ff, (0, 0), 2, flags=4)
    return free[1:-1, 1:-1] == 2


def name_runs(runs, shape, bbox=None):
    snap_ends(runs)
    H, W = shape
    # rasterise only around the walls; everything beyond is outside
    xs = [v for r in runs for v in ((r.c,) if r.orient == 'V' else (r.s0, r.s1))]
    ys = [v for r in runs for v in ((r.c,) if r.orient == 'H' else (r.s0, r.s1))]
    m = int(3 * PPF)
    ox, oy = max(int(min(xs)) - m, 0), max(int(min(ys)) - m, 0)
    ex, ey = min(int(max(xs)) + m, W), min(int(max(ys)) + m, H)
    wm = raster_runs(runs, (ey - oy, ex - ox), origin=(ox, oy))
    out = np.ones(shape, bool)
    out[oy:ey, ox:ex] = outside_region(wm)
    del wm
    if bbox is not None:
        # safety net: if the exterior leaked inside (wall loop not closed),
        # fall back to the rectangle of the exterior wall centerlines
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        inner = (~out[y0:y1, x0:x1]).mean() if (x1 > x0 and y1 > y0) else 1.0
        if inner < 0.3:
            out = np.ones(shape, bool)
            out[y0:y1 + 1, x0:x1 + 1] = False
    for r in runs:
        if r.typ != CMU:
            r.name = TYPE_NAME[r.typ]
            continue
        d = r.t / 2 + 5
        ss = np.linspace(r.s0, r.s1, 25)[2:-2]
        votes = 0
        for side in (-1, 1):
            cc = r.c + side * d
            if r.orient == 'H':
                ys = np.full_like(ss, cc); xs = ss
            else:
                xs = np.full_like(ss, cc); ys = ss
            xi = np.clip(np.round(xs).astype(int), 0, W - 1)
            yi = np.clip(np.round(ys).astype(int), 0, H - 1)
            votes = max(votes, out[yi, xi].mean())
        r.name = 'EXT' if votes > 0.5 else 'P3'
    snap_ends(runs)
    return out
