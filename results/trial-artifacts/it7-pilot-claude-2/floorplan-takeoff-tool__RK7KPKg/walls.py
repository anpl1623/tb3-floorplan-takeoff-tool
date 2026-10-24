"""Wall (partition) detection, classification and centreline measurement."""
import numpy as np
import cv2


class Seg:
    """Axis-aligned wall band.  For horizontal walls (o='h'): a..b is the x-extent,
    c is the y of the centreline, t the thickness (face to face, px)."""
    __slots__ = ('o', 'a', 'b', 'c', 't', 'f1', 'f2', 'kind', 'feat', 'id', 'ext',
                 'ea', 'eb', 'door_parts', 'score', 'cls_scores', 'len_px', 'meta')

    def __init__(self, o, a, b, c, t):
        self.o = o
        self.a = a
        self.b = b
        self.c = c
        self.t = t
        self.f1 = c - t / 2.0
        self.f2 = c + t / 2.0
        self.kind = None
        self.feat = None
        self.id = -1
        self.ext = False
        self.ea = a
        self.eb = b
        self.door_parts = []
        self.score = None
        self.cls_scores = None
        self.len_px = None
        self.meta = {}

    def __repr__(self):
        return 'Seg(%s %.1f..%.1f @%.1f t=%.1f %s)' % (self.o, self.a, self.b, self.c, self.t, self.kind)


def line_strength(ink, horiz):
    """Integrated ink across a line (3 px) averaged along the line (7 px)."""
    if horiz:
        return cv2.blur(ink, (7, 3)) * 3.0
    return cv2.blur(ink, (3, 7)) * 3.0


def _valid_runs(C, hmin, hmax):
    """C: uint8 mask; runs along axis 0 (down columns).  Returns arrays (y0, y1, x)."""
    H, W = C.shape
    pad = np.zeros((1, W), np.int8)
    d = np.diff(np.vstack([pad, C.astype(np.int8), pad]), axis=0)
    ys, xs = np.nonzero(d == 1)
    ye, xe = np.nonzero(d == -1)
    o1 = np.lexsort((ys, xs))
    o2 = np.lexsort((ye, xe))
    ys, xs = ys[o1], xs[o1]
    ye = ye[o2]
    h = ye - ys
    ok = (h >= hmin) & (h <= hmax)
    return ys[ok], ye[ok], xs[ok]


def _link_runs(ys, ye, xs, max_dy=2, max_gap=1):
    """Link runs in neighbouring columns into segments.  Returns list of arrays of run indices."""
    order = np.lexsort((ys, xs))
    ys, ye, xs = ys[order], ye[order], xs[order]
    # group runs by column
    segs = []          # list of dict(last_x, last_top, last_bot, idx list)
    active = []
    col_start = 0
    n = len(xs)
    i = 0
    while i < n:
        x = xs[i]
        j = i
        while j < n and xs[j] == x:
            j += 1
        new_active = []
        used = set()
        for k in range(i, j):
            best = None
            for si, sg in enumerate(active):
                if si in used:
                    continue
                if x - sg['x'] > max_gap + 1:
                    continue
                dt = abs(sg['top'] - ys[k]) + abs(sg['bot'] - ye[k])
                if abs(sg['top'] - ys[k]) <= max_dy and abs(sg['bot'] - ye[k]) <= max_dy:
                    if best is None or dt < best[0]:
                        best = (dt, si)
            if best is not None:
                sg = active[best[1]]
                used.add(best[1])
                sg['x'] = x
                sg['top'] = ys[k]
                sg['bot'] = ye[k]
                sg['idx'].append(order[k])
                new_active.append(sg)
            else:
                sg = {'x': x, 'top': ys[k], 'bot': ye[k], 'idx': [order[k]]}
                segs.append(sg)
                new_active.append(sg)
        # keep segments that may still continue (gap tolerance)
        for si, sg in enumerate(active):
            if si not in used and x - sg['x'] <= max_gap:
                new_active.append(sg)
        active = new_active
        i = j
    return [np.array(s['idx']) for s in segs]


def _subpix_peaks(p, thr):
    out = []
    for i in range(1, len(p) - 1):
        if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] > thr:
            lo = max(0, i - 2)
            hi = min(len(p), i + 3)
            w = np.clip(p[lo:hi] - 0.05, 0, None)
            if w.sum() <= 0:
                continue
            out.append(((np.arange(lo, hi) * w).sum() / w.sum(), p[i]))
    return out


def suppress_fills(x, min_level=0.2, max_std=0.07, size=15):
    """Zero out large uniform solid fills (e.g. walk-off mats, carpet) so that they do not
    merge with the walls around them.  Wall poche is narrower than `size` and is kept."""
    x = x.astype(np.float32)
    m = cv2.blur(x, (9, 9))
    m2 = cv2.blur(x * x, (9, 9))
    sd = np.sqrt(np.maximum(m2 - m * m, 0))
    fill = ((m > min_level) & (sd < max_std)).astype(np.uint8)
    fill = cv2.morphologyEx(fill, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    if not fill.any():
        return x
    # extend over the 9x9 window support (the uniform test centre is 4 px inside the fill)
    fill = cv2.dilate(fill, np.ones((7, 7), np.uint8))
    # but never over line-like pixels darker than the fill
    lvl = cv2.blur(x * fill, (31, 31)) / np.maximum(cv2.blur(fill.astype(np.float32), (31, 31)), 1e-3)
    keep = x > lvl + 0.15
    out = x.copy()
    out[(fill > 0) & ~keep] = 0.0
    return out


def find_bands(ink, roi, ppf, tau):
    """Find horizontal and vertical wall-band segments inside roi=(x0,y0,x1,y1)."""
    x0r, y0r, x1r, y1r = roi
    sub = suppress_fills(ink[y0r:y1r, x0r:x1r], size=int(round(1.25 * ppf)))
    tmax = int(round(1.0 * ppf)) + 4
    hmin = int(round(0.3 * ppf))
    hmax = int(round(1.05 * ppf))
    out = []
    for horiz in (True, False):
        L = line_strength(sub, horiz)
        B = (L > tau).astype(np.uint8)
        if horiz:
            C = cv2.morphologyEx(B, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, tmax)))
            ys, ye, xs = _valid_runs(C, hmin, hmax)
        else:
            C = cv2.morphologyEx(B, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (tmax, 1)))
            ys, ye, xs = _valid_runs(C.T, hmin, hmax)
        groups = _link_runs(ys, ye, xs)
        for g in groups:
            if len(g) < 3:
                continue
            a = int(xs[g].min())
            b = int(xs[g].max()) + 1
            top = float(np.median(ys[g]))
            bot = float(np.median(ye[g]))
            if horiz:
                seg = Seg('h', a + x0r, b + x0r, (top + bot) / 2.0 + y0r, bot - top)
            else:
                seg = Seg('v', a + y0r, b + y0r, (top + bot) / 2.0 + x0r, bot - top)
            seg.meta['ncols'] = len(g)
            out.append(seg)
    return out


def refine_seg(ink, seg, ppf):
    """Sub-pixel face-line positions from the averaged cross profile."""
    r = int(round(0.75 * ppf))
    c = int(round(seg.c))
    a = int(round(seg.a)) + 2
    b = int(round(seg.b)) - 2
    if b - a < 3:
        a = int(round(seg.a))
        b = int(round(seg.b))
    if seg.o == 'h':
        patch = ink[c - r:c + r + 1, a:b]
        prof = np.median(patch, axis=1) if patch.shape[1] >= 5 else patch.mean(1)
    else:
        patch = ink[a:b, c - r:c + r + 1]
        prof = np.median(patch, axis=0) if patch.shape[0] >= 5 else patch.mean(0)
    pk = _subpix_peaks(prof, 0.3)
    seg.meta['prof'] = prof
    if len(pk) >= 2:
        # choose the pair with spacing closest to the band thickness estimate and strong peaks
        best = None
        for i in range(len(pk)):
            for j in range(i + 1, len(pk)):
                d = pk[j][0] - pk[i][0]
                if d < 0.25 * ppf or d > 0.95 * ppf:
                    continue
                # prefer outermost strong peaks around the band
                mid = 0.5 * (pk[i][0] + pk[j][0]) - r
                sc = abs(d - (seg.t - 3)) * 0.5 + abs(mid) + (2 - pk[i][1] - pk[j][1]) * 3
                if best is None or sc < best[0]:
                    best = (sc, pk[i][0], pk[j][0])
        if best is not None:
            f1 = best[1] - r + c
            f2 = best[2] - r + c
            seg.f1, seg.f2 = f1, f2
            seg.c = 0.5 * (f1 + f2)
            seg.t = f2 - f1
            return True
    # fall back: solid band – use half-max edges
    seg.f1 = seg.c - (seg.t - 3) / 2.0
    seg.f2 = seg.c + (seg.t - 3) / 2.0
    seg.t = seg.f2 - seg.f1
    return False


# ----------------------------------------------------------------------------
# Poche description
# ----------------------------------------------------------------------------

def interior_strip(ink, o, f1, f2, a, b, d=2.5):
    """Interior of a wall band (between face lines), as a horizontal strip."""
    y0 = int(np.ceil(f1 + d))
    y1 = int(np.floor(f2 - d)) + 1
    a = int(round(a))
    b = int(round(b))
    if y1 - y0 < 1 or b - a < 4:
        return None
    if o == 'h':
        return ink[y0:y1, a:b]
    return ink[a:b, y0:y1].T


def acf_feat(S, h, maxdx=8, maxdy=2):
    """Small-lag autocorrelation of the central h rows of strip S."""
    if S is None or S.shape[0] < 1:
        return None
    h = min(h, S.shape[0])
    o = (S.shape[0] - h) // 2
    S = S[o:o + h].astype(np.float32)
    hh, w = S.shape
    md = min(maxdy, (hh - 1) // 2)
    out = []
    for dy in range(-maxdy, maxdy + 1):
        for dx in range(0, maxdx + 1):
            if dx == 0 and dy < 0:
                continue
            ady = min(abs(dy), md) * (1 if dy >= 0 else -1)
            if w - dx < 4:
                out.append(0.0)
                continue
            if ady >= 0:
                A = S[0:hh - ady, 0:w - dx]
                B = S[ady:hh, dx:w]
            else:
                A = S[-ady:hh, 0:w - dx]
                B = S[0:hh + ady, dx:w]
            out.append(float((A * B).mean()))
    return np.array(out, np.float32)


class PocheModel:
    """Wall-type templates from the legend partition swatches (EXT, P3, P2, P1)."""
    NAMES = ['EXT', 'P3', 'P2', 'P1']

    def __init__(self, ink, parts, ppf):
        self.ok = len(parts) == 4
        self.t = []
        self.strips = []
        self.mean = []
        self.peak = []
        for (x, y, w, h) in parts:
            sg = Seg('h', x + 8, x + w - 8, y + h / 2.0, h - 3)
            refine_seg(ink, sg, ppf)
            st = interior_strip(ink, 'h', sg.f1, sg.f2, x + 10, x + w - 10)
            self.t.append(sg.t)
            self.strips.append(st)
            self.mean.append(float(st.mean()) if st is not None else 0.0)
            prof = sg.meta.get('prof')
            self.peak.append(float(prof.max()) if prof is not None else 1.0)
        # missing swatches (legend partly unreadable): pad so that indexing stays valid
        while len(self.strips) < 4:
            self.t.append(0.6 * ppf)
            self.strips.append(None)
            self.mean.append(0.0)
            self.peak.append(1.0)
        self.hint = 7

    def dist(self, S, k, h=None):
        """ACF distance of wall interior strip S to template k."""
        T = self.strips[k]
        if S is None or T is None:
            return 9.0
        if h is None:
            h = min(S.shape[0], T.shape[0], self.hint)
        A = acf_feat(S, h)
        B = acf_feat(T, h)
        if A is None or B is None:
            return 9.0
        return float(np.sqrt(np.mean((A - B) ** 2)))

    def mean_dist(self, S, k):
        T = self.strips[k]
        if S is None or T is None:
            return 9.0
        h = min(S.shape[0], T.shape[0], self.hint)
        oS = (S.shape[0] - h) // 2
        oT = (T.shape[0] - h) // 2
        return abs(float(S[oS:oS + h].mean()) - float(T[oT:oT + h].mean()))


# ----------------------------------------------------------------------------
# Full wall analysis
# ----------------------------------------------------------------------------

class Frame:
    """Mapping between pixels and feet (origin at the building's outer top-left corner)."""

    def __init__(self, X0, Y0, ppfx, ppfy, W_ft, H_ft, t_ext_ft):
        self.X0 = X0
        self.Y0 = Y0
        self.px = ppfx
        self.py = ppfy
        self.W = W_ft
        self.H = H_ft
        self.text = t_ext_ft

    def fx(self, x):
        return (x - self.X0) / self.px

    def fy(self, y):
        return (y - self.Y0) / self.py

    def ix(self, xf):
        return self.X0 + xf * self.px

    def iy(self, yf):
        return self.Y0 + yf * self.py

    # along/cross helpers for a line orientation
    def along(self, o, v):
        return self.fx(v) if o == 'h' else self.fy(v)

    def cross(self, o, v):
        return self.fy(v) if o == 'h' else self.fx(v)

    def along_px(self, o, f):
        return self.ix(f) if o == 'h' else self.iy(f)

    def cross_px(self, o, f):
        return self.iy(f) if o == 'h' else self.ix(f)

    def pp(self, o):
        """pixels per foot along a line of orientation o"""
        return self.px if o == 'h' else self.py


def calibrate_tau(ink, parts):
    vals = []
    for (x, y, w, h) in parts:
        L = line_strength(ink[y - 6:y + h + 6, x + 10:x + w - 10], True)
        prof = np.median(L, axis=1)
        vals.append(float(prof.max()))
    if not vals:
        return 1.4
    v = float(np.median(vals))
    return float(np.clip(0.6 * v, 1.0, 1.8))


def _snap(v, tol1=0.2, tol2=0.1):
    r = round(v)
    if abs(v - r) <= tol1:
        return float(r)
    r2 = round(v * 2) / 2.0
    if abs(v - r2) <= tol2:
        return float(r2)
    return float(v)


class WallLine:
    def __init__(self, o, c):
        self.o = o
        self.c = c          # snapped centreline (ft)
        self.segs = []      # (a_ft, b_ft, seg)
        self.t = None       # thickness ft
        self.ext = False
        self.runs = []      # list of [s, e] in ft
        self.pieces = []    # (s, e, kind)
        self.crossings = [] # (c_perp, t_perp, line)
        self.f1 = None
        self.f2 = None
        self.openings = []  # bridged gaps (ft)

    def __repr__(self):
        return 'WallLine(%s c=%.2f t=%.2fin n=%d %s)' % (self.o, self.c, (self.t or 0) * 12, len(self.segs),
                                                       'EXT' if self.ext else '')


def _window_votes(ink, sg, pm, ppf, thick_ft_thr):
    """Per-window poche distances along a segment: returns list of (pos_a, pos_b, dists[4])."""
    out = []
    L = sg.b - sg.a
    win = 48
    if L < 20:
        return out
    if L < win + 8:
        starts = [sg.a + 3]
        win = int(L - 6)
    else:
        starts = list(np.arange(sg.a + 4, sg.b - win - 3, 24))
    for a in starts:
        S = interior_strip(ink, sg.o, sg.f1, sg.f2, a, a + win)
        if S is None:
            continue
        d = [pm.dist(S, k) for k in range(4)]
        out.append((a, a + win, d))
    return out


def analyze_walls(ink, ppf, legend, roi, dbg=None):
    parts = legend['parts']
    pm = PocheModel(ink, parts, ppf)
    tau = calibrate_tau(ink, parts)
    segs = find_bands(ink, roi, ppf, tau)
    for sg in segs:
        refine_seg(ink, sg, ppf)
    segs = [s for s in segs if 0.28 * ppf <= s.t <= 0.8 * ppf and (s.b - s.a) >= 0.3 * ppf]
    thick_thr = 6.2 / 12.0
    # poche votes
    for sg in segs:
        sg.meta['votes'] = _window_votes(ink, sg, pm, ppf, thick_thr)
        sg.meta['thin'] = sg.t / ppf < thick_thr
    # ---- building frame
    longh = [s for s in segs if s.o == 'h' and not s.meta['thin'] and (s.b - s.a) > 15 * ppf]
    longv = [s for s in segs if s.o == 'v' and not s.meta['thin'] and (s.b - s.a) > 15 * ppf]
    if not longh or not longv:
        raise RuntimeError('no building frame')
    top_c = min(s.c for s in longh)
    bot_c = max(s.c for s in longh)
    lef_c = min(s.c for s in longv)
    rig_c = max(s.c for s in longv)

    def on(sl, c):
        return [s for s in sl if abs(s.c - c) < 4]
    tops, bots, lefs, rigs = on(longh, top_c), on(longh, bot_c), on(longv, lef_c), on(longv, rig_c)
    ext_segs = tops + bots + lefs + rigs
    t_ext_px = float(np.median([s.t for s in ext_segs]))
    X0 = float(np.median([s.f1 for s in lefs]))
    X1 = float(np.median([s.f2 for s in rigs]))
    Y0 = float(np.median([s.f1 for s in tops]))
    Y1 = float(np.median([s.f2 for s in bots]))
    # outer face is the outer edge of the wall: f1 for top/left (face line centre)
    Wf = (X1 - X0) / ppf
    Hf = (Y1 - Y0) / ppf
    Wr, Hr = round(Wf), round(Hf)
    ppfx = ppf
    ppfy = ppf
    if abs(Wf - Wr) < 0.35 and Wr > 0:
        q = (X1 - X0) / Wr
        if abs(q / ppf - 1) < 0.004:
            ppfx = q
        Wf = float(Wr)
    if abs(Hf - Hr) < 0.35 and Hr > 0:
        q = (Y1 - Y0) / Hr
        if abs(q / ppf - 1) < 0.004:
            ppfy = q
        Hf = float(Hr)
    t_ext_in = t_ext_px / ppf * 12.0
    t_ext_ft = round(t_ext_in) / 12.0 if abs(t_ext_in - round(t_ext_in)) < 0.45 else t_ext_in / 12.0
    fr = Frame(X0, Y0, ppfx, ppfy, Wf, Hf, t_ext_ft)
    # ---- group into lines
    lines = {}
    ext_coords = {('h', round(t_ext_ft / 2, 4)), ('h', round(Hf - t_ext_ft / 2, 4)),
                  ('v', round(t_ext_ft / 2, 4)), ('v', round(Wf - t_ext_ft / 2, 4))}
    for sg in segs:
        cf = fr.cross(sg.o, sg.c)
        # inside the building (with margin)?
        lim = Hf if sg.o == 'h' else Wf
        if cf < -0.5 or cf > lim + 0.5:
            continue
        af = fr.along(sg.o, sg.a)
        bf = fr.along(sg.o, sg.b)
        alim = Wf if sg.o == 'h' else Hf
        if bf < -0.5 or af > alim + 0.5:
            continue
        # snap
        cs = None
        for (o2, ce) in ext_coords:
            if o2 == sg.o and abs(cf - ce) < 0.25:
                cs = ce
        if cs is None:
            cs = _snap(cf)
        key = (sg.o, round(cs, 3))
        if key not in lines:
            lines[key] = WallLine(sg.o, cs)
            lines[key].ext = (sg.o, round(cs, 4)) in ext_coords
        lines[key].segs.append((af, bf, sg))
    # ---- validity of segments: poche votes (relative to the template's distance from an
    #      empty band, so text and symbols that match no partition type are rejected)
    de = [max(pm.dist(np.zeros_like(pm.strips[k]), k), 1e-4) if pm.strips[k] is not None else 1.0
          for k in range(4)]
    pm.de = de
    for L in lines.values():
        for (af, bf, sg) in L.segs:
            v = sg.meta['votes']
            good = 0
            for (_, _, d) in v:
                if sg.meta['thin']:
                    ok = d[3] <= 0.6 * de[3]
                else:
                    ok = min(d[0] / de[0], d[1] / de[1], d[2] / de[2]) <= 0.6
                good += ok
            sg.meta['good'] = good
            sg.meta['nwin'] = len(v)
    # keep lines that have strong evidence: long segments with good poche votes
    keep = {}
    for k, L in lines.items():
        tot = sum(bf - af for (af, bf, sg) in L.segs
                  if sg.meta['nwin'] and sg.meta['good'] >= 0.5 * sg.meta['nwin'])
        # long, continuous bands of wall thickness are walls even if their poche matches
        # the legend poorly (degraded scans); short ones need good poche evidence
        longsegs = sum(bf - af for (af, bf, sg) in L.segs if bf - af >= 3.0)
        if L.ext or tot >= 2.0 or longsegs >= 10.0:
            keep[k] = L
            L.weak = tot < 2.0
    lines = keep
    # line thickness and faces (px)
    for L in lines.values():
        ts = [(sg.t, bf - af) for (af, bf, sg) in L.segs]
        wsum = sum(w for _, w in ts)
        L.t = (sum(t * w for t, w in ts) / max(wsum, 1e-6)) / fr.pp('v' if L.o == 'h' else 'h')
        L.f1 = L.c - L.t / 2
        L.f2 = L.c + L.t / 2
    if dbg is not None:
        dbg['frame'] = fr
        dbg['lines'] = lines
        dbg['segs'] = segs
        dbg['pm'] = pm
        dbg['tau'] = tau
    return fr, lines, pm


def _gap_profile(ink, fr, L, g0, g1):
    """Median cross profile of ink over the along-range [g0,g1] (ft) of line L.
    Returns (profile, offset_px) where index i corresponds to cross px = base + i."""
    o = L.o
    a = int(round(fr.along_px(o, g0))) + 2
    b = int(round(fr.along_px(o, g1))) - 2
    if b - a < 3:
        return None, None
    cpx = fr.cross_px(o, L.c)
    tpx = L.t * fr.pp('v' if o == 'h' else 'h')
    r = int(round(tpx / 2 + 6))
    c0 = int(round(cpx)) - r
    c1 = int(round(cpx)) + r + 1
    if o == 'h':
        patch = ink[c0:c1, a:b]
        prof = np.median(patch, axis=1)
    else:
        patch = ink[a:b, c0:c1]
        prof = np.median(patch, axis=0)
    return prof, c0


def door_threshold_evidence(ink, fr, L, g0, g1):
    """True if thin lines continue along both wall faces across the gap (a door opening)."""
    prof, base = _gap_profile(ink, fr, L, g0, g1)
    if prof is None:
        return False
    o = L.o
    pp = fr.pp('v' if o == 'h' else 'h')
    f1 = fr.cross_px(o, L.c) - L.t * pp / 2 - base
    f2 = fr.cross_px(o, L.c) + L.t * pp / 2 - base
    def peak_near(f):
        i0 = max(0, int(np.floor(f - 2.5)))
        i1 = min(len(prof), int(np.ceil(f + 2.5)) + 1)
        if i1 <= i0:
            return 0.0
        return float(prof[i0:i1].max())
    p1 = peak_near(f1)
    p2 = peak_near(f2)
    # background level: outside the band
    outside = np.concatenate([prof[:max(1, int(f1 - 3))], prof[min(len(prof) - 1, int(f2 + 4)):]])
    bgl = float(np.median(outside)) if len(outside) else 0.0
    return (p1 - bgl > 0.18) and (p2 - bgl > 0.18)


def _crossings(L, lines, use_runs):
    out = []
    for P in lines.values():
        if P.o == L.o:
            continue
        tol = L.t / 2 + 0.5
        cov = P.runs if (use_runs and P.runs) else [(a, b) for a, b, s in P.segs]
        if any(a - tol <= L.c <= b + tol for a, b in cov):
            out.append((P.c, P.t, P))
    out.sort(key=lambda q: q[0])
    return out


def _runs_for_line(ink, fr, L, max_door):
    iv = sorted([(a, b) for a, b, s in L.segs])
    runs = []
    cur = list(iv[0])
    L.openings = []
    for a, b in iv[1:]:
        g0, g1 = cur[1], a
        gap = g1 - g0
        bridge = False
        if gap <= 1.2:
            bridge = True
        else:
            for (cp, tp, P) in L.crossings:
                if g0 - 0.4 <= cp - tp / 2 and cp + tp / 2 <= g1 + 0.4 and gap <= tp + 0.9:
                    bridge = True
            if not bridge and gap <= max_door and door_threshold_evidence(ink, fr, L, g0, g1):
                bridge = True
                L.openings.append((g0, g1))
        if bridge:
            cur[1] = max(cur[1], b)
        else:
            runs.append(cur)
            cur = [a, b]
    runs.append(cur)
    out = []
    for s, e in runs:
        best = None
        for (cp, tp, P) in L.crossings:
            if abs(s - cp) <= tp / 2 + 0.55:
                if best is None or abs(s - cp) < abs(s - best):
                    best = cp
        if best is None:
            for (cp, tp, P) in reversed(L.crossings):
                if cp < s and s - cp <= max_door and door_threshold_evidence(ink, fr, L, cp + tp / 2, s):
                    best = cp
                    L.openings.append((cp + tp / 2, s))
                    break
        if best is not None:
            s = best
        best = None
        for (cp, tp, P) in L.crossings:
            if abs(e - cp) <= tp / 2 + 0.55:
                if best is None or abs(e - cp) < abs(e - best):
                    best = cp
        if best is None:
            for (cp, tp, P) in L.crossings:
                if cp > e and cp - e <= max_door and door_threshold_evidence(ink, fr, L, e, cp - tp / 2):
                    best = cp
                    L.openings.append((e, cp - tp / 2))
                    break
        if best is not None:
            e = best
        out.append([s, e])
    # merge runs that now touch/overlap
    out.sort()
    merged = []
    for r in out:
        if merged and r[0] <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return merged


def build_runs(ink, fr, lines, max_door=18.0, iters=3):
    """Compute crossings, runs (with door bridging) and end snapping for every wall line."""
    for L in lines.values():
        L.runs = []
    for it in range(iters):
        for L in lines.values():
            L.crossings = _crossings(L, lines, use_runs=(it > 0))
        new = {}
        for k, L in lines.items():
            new[k] = _runs_for_line(ink, fr, L, max_door)
        # the exterior wall is a closed loop: each exterior line runs corner to corner
        for k, L in lines.items():
            if not L.ext:
                continue
            perp = sorted(P.c for P in lines.values() if P.ext and P.o != L.o)
            if len(perp) >= 2:
                new[k] = [[perp[0], perp[-1]]]
        changed = any(new[k] != lines[k].runs for k in lines)
        for k, L in lines.items():
            L.runs = new[k]
        if not changed and it > 0:
            break
    for L in lines.values():
        L.crossings = _crossings(L, lines, use_runs=True)


def classify_pieces(fr, lines, pm, thick_thr_ft=6.2 / 12.0):
    """Split runs at crossings and assign a partition type to each piece."""
    for L in lines.values():
        L.pieces = []
        pp_along = fr.pp(L.o)
        for (s, e) in L.runs:
            cuts = [s] + [cp for (cp, tp, P) in L.crossings if s + 0.05 < cp < e - 0.05] + [e]
            for p0, p1 in zip(cuts[:-1], cuts[1:]):
                votes = {'CMU': 0.0, 'P2': 0.0, 'P1': 0.0}
                ts = []
                for (a, b, sg) in L.segs:
                    if b < p0 or a > p1:
                        continue
                    ov = min(b, p1) - max(a, p0)
                    if ov > 0:
                        ts.append((sg.t, ov))
                    thin = sg.meta['thin']
                    for (wa, wb, d) in sg.meta['votes']:
                        cf = fr.along(L.o, 0.5 * (wa + wb))
                        if not (p0 + 0.1 <= cf <= p1 - 0.1):
                            continue
                        if any(o0 - 0.2 <= cf <= o1 + 0.2 for (o0, o1) in L.openings):
                            continue
                        w = (wb - wa) / pp_along
                        de = getattr(pm, 'de', [1.0, 1.0, 1.0, 1.0])
                        if thin:
                            if d[3] <= 0.6 * de[3]:
                                votes['P1'] += w
                        else:
                            dc = min(d[0] / de[0], d[1] / de[1])
                            dp = d[2] / de[2]
                            if min(dc, dp) <= 0.6 and (min(dc, dp) < 0.7 * max(dc, dp) or min(dc, dp) < 0.08):
                                votes['CMU' if dc < dp else 'P2'] += w
                tpx = None
                if ts:
                    tw = sum(w for _, w in ts)
                    tpx = sum(t * w for t, w in ts) / tw
                kind = max(votes, key=votes.get) if max(votes.values()) > 0 else None
                L.pieces.append([p0, p1, kind, tpx, votes])
        # pieces without any confident vote: nearest template over all windows
        for p in L.pieces:
            if max(p[4].values()) > 0:
                continue
            acc = {'CMU': 0.0, 'P2': 0.0, 'P1': 0.0}
            for (a, b, sg) in L.segs:
                if b < p[0] or a > p[1]:
                    continue
                de = getattr(pm, 'de', [1.0, 1.0, 1.0, 1.0])
                for (wa, wb, d) in sg.meta['votes']:
                    if sg.meta['thin']:
                        acc['P1'] += 1
                    else:
                        dc = min(d[0] / de[0], d[1] / de[1])
                        dp = d[2] / de[2]
                        acc['CMU' if dc < dp else 'P2'] += 1
            if max(acc.values()) > 0:
                p[2] = max(acc, key=acc.get)
        # fill unknown kinds from neighbours / line majority
        tot = {}
        for p in L.pieces:
            if p[2]:
                tot[p[2]] = tot.get(p[2], 0) + (p[1] - p[0])
        maj = max(tot, key=tot.get) if tot else None
        for i, p in enumerate(L.pieces):
            if p[2] is None:
                nb = None
                for j in (i - 1, i + 1):
                    if 0 <= j < len(L.pieces) and L.pieces[j][2] and abs(L.pieces[j][0] - p[1]) < 0.01 or \
                            (0 <= j < len(L.pieces) and L.pieces[j][2] and abs(L.pieces[j][1] - p[0]) < 0.01):
                        nb = L.pieces[j][2]
                        break
                p[2] = nb or maj
        if L.ext:
            for p in L.pieces:
                p[2] = 'EXT'


def wall_lengths(lines):
    out = {'EXT': 0.0, 'P1': 0.0, 'P2': 0.0, 'P3': 0.0}
    for L in lines.values():
        for p0, p1, kind, tpx, votes in L.pieces:
            k = kind
            if k == 'CMU':
                k = 'P3'
            if k is None:
                continue
            out[k] += p1 - p0
    return out


def filter_lines(lines):
    """Drop lines that do not connect to other walls (text, door leaves, symbols).
    Interior wall runs must end on other walls at both ends unless they are long."""
    keep = {}
    for k, L in lines.items():
        if L.ext:
            keep[k] = L
            continue
        tin = L.t * 12
        if not (3.3 <= tin <= 9.5):
            continue
        good_runs = []
        for (s, e) in L.runs:
            at_s = any(abs(s - cp) < 1e-6 for (cp, tp, P) in L.crossings)
            at_e = any(abs(e - cp) < 1e-6 for (cp, tp, P) in L.crossings)
            if (at_s and at_e) or ((at_s or at_e) and e - s >= 8.0):
                good_runs.append((s, e))
        if not good_runs:
            continue
        # keep only the segments that lie within accepted runs
        segs = [(a, b, sg) for (a, b, sg) in L.segs
                if any(a >= s - 1.0 and b <= e + 1.0 for (s, e) in good_runs)]
        if not segs:
            continue
        L.segs = segs
        keep[k] = L
    return keep
