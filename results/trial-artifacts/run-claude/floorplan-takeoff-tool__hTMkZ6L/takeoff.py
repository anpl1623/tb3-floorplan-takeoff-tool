#!/usr/bin/env python3
"""Quantity takeoff from a scanned Cedar Valley Market A101 fixture & finish plan.

Usage: python3 takeoff.py <scan.pdf> <takeoff.json>

Pipeline
  1. extract the scan image from the PDF, remove the scan rotation
  2. scale (px/ft) from the graphic scale bar (fallback: sheet border size)
  3. walls: pairs of heavy parallel face lines -> bands, typed by thickness/fill
     (CMU hatch, dark-gray 1-hr stud, light-gray stud, door opening); collinear
     bands chained through doors/junctions; ends extended to the centerline of
     the wall they run into
  4. floor finishes: building interior minus wall bands, split into regions by
     heavy finish-transition lines, each region classified by its hatch
  5. doors: openings in wall bands, typed by leaves / sliding panels
  6. fixtures: strips of medium-weight parallel lines of the standard fixture
     widths, cut into pieces at cross lines, classified by size and diagonals
"""
import sys, json, os, traceback
import numpy as np
import cv2
import fitz  # PyMuPDF
from scipy import ndimage as ndi

DEBUG = bool(os.environ.get('TAKEOFF_DEBUG'))

FINISHES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']
WALL_TYPES = ['EXT', 'P1', 'P2', 'P3']
DOOR_TYPES = ['single', 'pair', 'overhead_coiling', 'auto_sliding']
FIXTURE_TYPES = ['gondola_sections_4ft', 'gondola_sections_3ft', 'endcaps', 'checkouts',
                 'self_checkouts', 'cooler_doors', 'produce_tables']

# sheet border size in "feet at plan scale" (border is drawn at a fixed size on the sheet)
BORDER_W_FT = 280.89
BORDER_H_FT = 184.89

CMU, P2, P1, OPEN, JGAP = 1, 2, 3, 4, 5   # JGAP: gap bridged at a wall junction


def log(*a):
    if DEBUG:
        print(*a, file=sys.stderr)


# ----------------------------------------------------------------------------- image

def load_image(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    im = _load_page_image(doc, page)
    # the pipeline's pixel-level parameters assume ~200 dpi; resample other resolutions
    dpi = im.shape[1] / (page.rect.width / 72.0)
    if abs(dpi / 200.0 - 1) > 0.25:
        f = 200.0 / dpi
        im = cv2.resize(im, (int(round(im.shape[1] * f)), int(round(im.shape[0] * f))),
                        interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    return im


def _load_page_image(doc, page):
    imgs = page.get_images(full=True)
    if len(imgs) >= 1:
        best = max(imgs, key=lambda t: t[2] * t[3])
        try:
            info = doc.extract_image(best[0])
            arr = np.frombuffer(info['image'], np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if im is not None and im.shape[0] * im.shape[1] > 4e6:
                # the image should fill the page; otherwise render the page instead
                pr = page.rect
                if abs(im.shape[1] / im.shape[0] - pr.width / pr.height) < 0.02 and len(imgs) == 1:
                    return im
        except Exception:
            pass
    dpi = 200
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    return im


def estimate_skew(im):
    small = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small < 110)
    if len(ys) > 3_000_000:
        sel = np.random.default_rng(0).choice(len(ys), 3_000_000, replace=False)
        ys, xs = ys[sel], xs[sel]
    ys = ys.astype(np.float64)
    xs = xs.astype(np.float64)

    def score(th):
        c, s = np.cos(th), np.sin(th)
        yr = ys * c - xs * s
        xr = xs * c + ys * s
        hy = np.bincount((yr - yr.min()).astype(np.int64)).astype(np.float64)
        hx = np.bincount((xr - xr.min()).astype(np.int64)).astype(np.float64)
        return float((hy ** 2).sum() + (hx ** 2).sum())

    best = 0.0
    for step, span in [(0.1, 4.0), (0.01, 0.12), (0.002, 0.012)]:
        cands = np.arange(best - span, best + span + 1e-9, step)
        scores = [score(np.deg2rad(a)) for a in cands]
        best = float(cands[int(np.argmax(scores))])
    return best


def deskew(im, angle_deg):
    h, w = im.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=255)


# ----------------------------------------------------------------------------- scale

def find_scale_bar(d):
    """Graphic scale: [0-4 black][4-8 white][8-16 black][16-32 white], outlined."""
    blk = (d < 100).astype(np.uint8)
    blk = cv2.morphologyEx(blk, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(blk, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 8 or h > 80 or w < 2 * h or w > 2500:
            continue
        if a < 0.85 * w * h:
            continue
        cands.append((int(x), int(y), int(w), int(h)))
    best = None
    for (x1, y1, w1, h1) in cands:
        for (x2, y2, w2, h2) in cands:
            if x2 <= x1 + w1:
                continue
            if abs(y1 - y2) > 4 or abs(h1 - h2) > 4:
                continue
            u = (w2 - w1) / 4.0
            if u <= 2:
                continue
            gap = x2 - (x1 + w1)
            lw = w1 - 4 * u
            if abs(lw) > 0.4 * 4 * u:
                continue
            if abs(gap - (4 * u - lw)) > max(4, 0.15 * u * 4):
                continue
            err = abs(gap + w1 - 8 * u)
            if best is None or err < best[0]:
                best = (err, (x1, y1, w1, h1), (x2, y2, w2, h2), u)
    if best is None:
        return None
    _, A, B, u = best
    x1, y1, w1, h1 = A
    x2, y2, w2, h2 = B
    yc0 = int(min(y1, y2) + 0.25 * h1)
    yc1 = int(max(y1 + h1, y2 + h2) - 0.25 * h1)
    prof = 255.0 - d[yc0:yc1, :].astype(np.float64).mean(axis=0)
    hi = np.median(prof[x1 + 3:x1 + w1 - 3])
    lo = np.median(prof[x1 + w1 + 3:x2 - 3])
    mid = (hi + lo) / 2

    def crossing(xa, xb, rising):
        for x in range(xa, xb):
            p0, p1 = prof[x], prof[x + 1]
            if rising and p0 < mid <= p1:
                return x + (mid - p0) / (p1 - p0)
            if not rising and p0 >= mid > p1:
                return x + (p0 - mid) / (p0 - p1)
        return None

    aL = crossing(x1 - 6, x1 + 6, True)
    aR = crossing(x1 + w1 - 6, x1 + w1 + 6, False)
    bL = crossing(x2 - 6, x2 + 6, True)
    bR = crossing(x2 + w2 - 6, x2 + w2 + 6, False)
    if None in (aL, aR, bL, bR):
        return None
    cA = (aL + aR) / 2
    cB = (bL + bR) / 2
    ppf10 = (cB - cA) / 10.0
    if ppf10 <= 3:
        return None
    exp = cA + 30 * ppf10
    xs = int(exp - 0.9 * ppf10)
    xe = int(exp + 0.9 * ppf10)
    colprof = 255.0 - d[y1 + 3:y1 + h1 - 3, xs:xe].astype(np.float64).mean(axis=0)
    ppf, consistent = ppf10, False
    if len(colprof) > 3:
        k = int(np.argmax(colprof))
        lo_k, hi_k = max(0, k - 3), min(len(colprof), k + 4)
        wts = np.clip(colprof[lo_k:hi_k] - np.median(colprof), 0, None)
        if wts.sum() > 0 and colprof[k] - np.median(colprof) > 40:
            x32 = xs + lo_k + (wts * np.arange(len(wts))).sum() / wts.sum()
            ppf30 = (x32 - cA) / 30.0
            if abs(ppf30 / ppf10 - 1) < 0.01:
                ppf, consistent = ppf30, True
    return ppf, consistent


def border_scale(d):
    """Fallback scale from the sheet border rectangle."""
    dark = d < 128
    H, W = d.shape

    def first(v):
        idx = np.nonzero(v)[0]
        return idx[0] if len(idx) else None
    lefts, rights, tops, bots = [], [], [], []
    for y in range(int(H * 0.2), int(H * 0.8), 50):
        a = first(dark[y, :W // 6]); b = first(dark[y, ::-1][:W // 6])
        if a is not None: lefts.append(a)
        if b is not None: rights.append(W - 1 - b)
    for x in range(int(W * 0.2), int(W * 0.8), 50):
        a = first(dark[:H // 6, x]); b = first(dark[::-1, x][:H // 6])
        if a is not None: tops.append(a)
        if b is not None: bots.append(H - 1 - b)
    if not (lefts and rights and tops and bots):
        return None
    w = np.median(rights) - np.median(lefts)
    h = np.median(bots) - np.median(tops)
    return 0.5 * (w / BORDER_W_FT + h / BORDER_H_FT)


# ----------------------------------------------------------------------------- line segments

def extract_segments(img, depth_min=70, along=9, min_len=12):
    """Horizontal line segments (lines along axis 1).
    Returns list of (y_center_subpix, x0, x1, depth, npix)."""
    img = np.ascontiguousarray(img)
    f = cv2.blur(img.astype(np.float32), (along, 1))
    f = cv2.GaussianBlur(f, (1, 0), sigmaX=0.1, sigmaY=0.7)
    bg = ndi.maximum_filter1d(f, size=13, axis=0)
    bg -= f                                   # depth below the local background
    m = np.zeros(f.shape, bool)
    c = f[1:-1]
    m[1:-1] = (c < f[:-2]) & (c <= f[2:]) & (bg[1:-1] > depth_min)
    ys, xs = np.nonzero(m)
    fu = f[ys - 1, xs]; fc = f[ys, xs]; fd = f[ys + 1, xs]
    dp_all = bg[ys, xs]
    del bg, f, c
    denom = fu - 2 * fc + fd
    ok = np.abs(denom) > 1e-3
    off = np.where(ok, 0.5 * (fu - fd) / np.where(ok, denom, 1.0), 0.0)
    off = np.clip(off, -0.5, 0.5)
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    labs = lab[ys, xs]
    del lab, m
    order = np.argsort(labs, kind='stable')
    ys, xs, labs, off, dp = ys[order], xs[order], labs[order], off[order], dp_all[order]
    yo = ys + off
    bounds = np.searchsorted(labs, np.arange(1, n + 1))
    segs = []
    for i in range(1, n):
        a, b = bounds[i - 1], bounds[i]
        if b - a < min_len:
            continue
        xx = xs[a:b]; yy = yo[a:b]; dd = dp[a:b]
        if yy.max() - yy.min() <= 2.5:
            groups = [np.ones(len(yy), bool)]
        else:
            lv = np.round(yy).astype(int)
            vals, cnts = np.unique(lv, return_counts=True)
            groups = []
            used = np.zeros(len(yy), bool)
            for vi in np.argsort(-cnts):
                if cnts[vi] < min_len:
                    break
                near = (~used) & (np.abs(yy - vals[vi]) <= 0.75)
                if near.sum() < min_len:
                    continue
                cen = np.median(yy[near])
                g = (~used) & (np.abs(yy - cen) <= 1.25)
                if g.sum() >= min_len:
                    groups.append(g)
                    used |= g
        for g in groups:
            gx = xx[g]; gy = yy[g]; gd = dd[g]
            o = np.argsort(gx, kind='stable')
            gx, gy, gd = gx[o], gy[o], gd[o]
            brk = np.nonzero(np.diff(gx) > 6)[0]
            starts = np.concatenate([[0], brk + 1])
            ends = np.concatenate([brk + 1, [len(gx)]])
            for s0, s1 in zip(starts, ends):
                if s1 - s0 < min_len or gx[s1 - 1] - gx[s0] + 1 < min_len:
                    continue
                segs.append((float(np.median(gy[s0:s1])), int(gx[s0]), int(gx[s1 - 1]),
                             float(np.median(gd[s0:s1])), int(s1 - s0)))
    return segs


def merge_collinear(segs, gap=6, ytol=1.2):
    order = sorted(range(len(segs)), key=lambda i: segs[i][0])
    groups = []
    for i in order:
        if groups and abs(segs[i][0] - groups[-1][-1][0]) <= ytol:
            groups[-1].append(segs[i])
        else:
            groups.append([segs[i]])
    out = []
    for g in groups:
        g = sorted(g, key=lambda s: s[1])
        cur = list(g[0])
        for s in g[1:]:
            if s[1] <= cur[2] + gap:
                if s[2] > cur[2]:
                    wa = cur[2] - cur[1] + 1
                    wb = s[2] - s[1] + 1
                    cur[0] = (cur[0] * wa + s[0] * wb) / (wa + wb)
                    cur[3] = (cur[3] * wa + s[3] * wb) / (wa + wb)
                    cur[2] = s[2]
            else:
                out.append(tuple(cur))
                cur = list(s)
        out.append(tuple(cur))
    return out


# ----------------------------------------------------------------------------- walls

def runs_of(cls):
    out = []
    s = 0
    for i in range(1, len(cls) + 1):
        if i == len(cls) or cls[i] != cls[s]:
            out.append((s, i - 1, int(cls[s])))
            s = i
    return out


def rolling_iqr(v, w):
    if len(v) < w:
        q = np.percentile(v, [25, 75]) if len(v) else [0, 0]
        return np.full(len(v), q[1] - q[0], np.float32)
    pad = np.pad(v, w // 2, mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(pad, w)[:len(v)]
    q = np.percentile(win, [25, 75], axis=1)
    return (q[1] - q[0]).astype(np.float32)


def classify_columns(img, y1, y2, x0, x1, ppf, bgl):
    """Fill class of each column of a horizontal band between face lines y1 < y2."""
    r0 = int(np.ceil(y1 + 2.5))
    r1 = int(np.floor(y2 - 2.5))
    if r1 < r0:
        r0 = r1 = int(round((y1 + y2) / 2))
    inner = img[r0:r1 + 1, x0:x1 + 1].astype(np.float32)
    h = inner.shape[0]
    w = max(3, int(0.4 * ppf) | 1)
    kk = np.ones(w) / w

    def smooth(v):
        return np.convolve(np.pad(v, w // 2, mode='edge'), kk, mode='valid')
    t_ft = (y2 - y1) / ppf
    if t_ft < 0.5:
        # light-gray stud wall fill is uniform across the band; openings are white (maybe with panels)
        rel = bgl - smooth(inner.mean(axis=0))
        crng = smooth(inner.max(axis=0) - inner.min(axis=0))
        cls = np.where((rel > 12) & (rel < 60) & (crng < 45), P1, OPEN)
    else:
        k = max(1, h // 3)
        outer = inner[np.r_[0:k, h - k:h]]          # outer thirds: ignore center lines of openings
        rel = bgl - smooth(outer.mean(axis=0))
        # CMU hatch: diagonal strokes make the outer-third row means oscillate along the wall
        W = max(5, int(1.2 * ppf) | 1)
        osc = np.minimum(rolling_iqr(inner[:k].mean(axis=0), W), rolling_iqr(inner[h - k:].mean(axis=0), W))
        cls = np.where(rel > 50, P2, np.where(osc > 8, CMU, OPEN))
    return cls.astype(np.int8)


def find_pieces(img, segs, ppf, bgl, tmin_ft=0.3, tmax_ft=0.8, min_overlap=4, min_seg_ft=1.5):
    segs = sorted(segs, key=lambda s: s[0])
    ys = np.array([s[0] for s in segs])
    pieces = []
    for i, s1 in enumerate(segs):
        y1, a0, a1 = s1[0], s1[1], s1[2]
        j0 = np.searchsorted(ys, y1 + tmin_ft * ppf)
        j1 = np.searchsorted(ys, y1 + tmax_ft * ppf)
        for j in range(j0, j1):
            s2 = segs[j]
            y2, b0, b1 = s2[0], s2[1], s2[2]
            if max(a1 - a0, b1 - b0) < min_seg_ft * ppf:
                continue
            o0, o1 = max(a0, b0), min(a1, b1)
            if o1 - o0 < min_overlap:
                continue
            blocked = np.zeros(o1 - o0 + 1, bool)
            for k in range(i + 1, j):
                s3 = segs[k]
                if s3[0] - y1 > 2.0 and y2 - s3[0] > 2.0:
                    c0, c1 = max(o0, s3[1]), min(o1, s3[2])
                    if c1 >= c0:
                        blocked[c0 - o0:c1 - o0 + 1] = True
            if blocked.all():
                continue
            cls = classify_columns(img, y1, y2, o0, o1, ppf, bgl)
            cls[blocked] = OPEN
            for (r0, r1, c) in runs_of(cls):
                pieces.append(dict(c=(y1 + y2) / 2, t=y2 - y1, a=o0 + r0, b=o0 + r1, cls=int(c),
                                   f1=y1, f2=y2))
    return pieces


def group_lines(pieces, tol=2.5):
    pieces = sorted(pieces, key=lambda p: p['c'])
    lines = []
    for p in pieces:
        if lines and abs(p['c'] - lines[-1]['c']) <= tol:
            L = lines[-1]
            L['pieces'].append(p)
            L['c'] = float(np.average([q['c'] for q in L['pieces']],
                                      weights=[q['b'] - q['a'] + 1 for q in L['pieces']]))
        else:
            lines.append(dict(c=p['c'], pieces=[p]))
    return lines


def smooth_classes(segs, min_len):
    """segs: contiguous [s, e, cls] along a run (0 = gap, OPEN = opening).
    Short wall fragments are absorbed, openings/gaps take the neighbouring wall class."""
    segs = [list(s) for s in segs]
    changed = True
    while changed:
        changed = False
        out = []
        for s in segs:
            if out and out[-1][2] == s[2]:
                out[-1][1] = s[1]
            else:
                out.append(s)
        segs = out
        cand = [(s[1] - s[0] + 1, i) for i, s in enumerate(segs)
                if s[2] in (CMU, P2, P1) and s[1] - s[0] + 1 < min_len]
        if cand and len([s for s in segs if s[2] in (CMU, P2, P1)]) > 1:
            _, i = min(cand)
            segs[i][2] = OPEN
            changed = True
    walls = [i for i, s in enumerate(segs) if s[2] in (CMU, P2, P1)]
    if not walls:
        return []
    for i, s in enumerate(segs):
        if s[2] in (CMU, P2, P1):
            continue
        left = max([w for w in walls if w < i], default=None)
        right = min([w for w in walls if w > i], default=None)
        if left is None:
            s[2] = -segs[right][2]
        elif right is None:
            s[2] = -segs[left][2]
        elif segs[left][2] == segs[right][2]:
            s[2] = -segs[left][2]
        else:
            s[2] = 0
    out = []
    for s in segs:
        c = abs(s[2])
        if c == 0:
            mid = (s[0] + s[1]) / 2.0
            if out:
                out[-1][1] = mid
            out.append([mid, s[1], None])
            continue
        if out and (out[-1][2] == c or out[-1][2] is None):
            out[-1][1] = s[1]
            out[-1][2] = c
        else:
            out.append([s[0], s[1], c])
    return [o for o in out if o[2] is not None]


def build_runs(lines, ppf, segs_along, gap_ft=1.0, door_max_ft=18.0):
    S = np.array([(s[0], s[1], s[2]) for s in segs_along]) if segs_along else np.zeros((0, 3))

    def face_cov(y, x0, x1):
        if x1 <= x0:
            return 1.0
        m = np.abs(S[:, 0] - y) <= 3.0
        cov = np.zeros(int(x1 - x0) + 1, bool)
        for (yy, a, b) in S[m]:
            a0 = int(max(a, x0)); b0 = int(min(b, x1))
            if b0 >= a0:
                cov[a0 - int(x0):b0 - int(x0) + 1] = True
        return float(cov.mean())

    out = []
    for L in lines:
        ps = sorted(L['pieces'], key=lambda p: p['a'])
        a0 = min(p['a'] for p in ps)
        b0 = max(p['b'] for p in ps)
        occ = np.zeros(b0 - a0 + 1, np.int8)
        tt = np.zeros(b0 - a0 + 1, np.float32)
        f1 = np.zeros(b0 - a0 + 1, np.float32)
        f2 = np.zeros(b0 - a0 + 1, np.float32)
        for p in sorted(ps, key=lambda p: (p['cls'] == OPEN)):
            sl = slice(p['a'] - a0, p['b'] - a0 + 1)
            m = occ[sl] == 0
            occ[sl][m] = p['cls']
            tt[sl][m] = p['t']
            f1[sl][m] = p['f1']
            f2[sl][m] = p['f2']
        rr = runs_of(occ)
        chains = []
        cur = []
        for (s, e, c) in rr:
            if c == 0:
                glen = e - s + 1
                gcls = JGAP if glen <= gap_ft * ppf else None
                if gcls is None and cur and glen <= door_max_ft * ppf and s > 0 and e < len(occ) - 1:
                    cA = face_cov(f1[s - 1], s + a0, e + a0)
                    cB = face_cov(f2[s - 1], s + a0, e + a0)
                    if cA > 0.85 and cB > 0.85:
                        gcls = 0        # opening whose band was not paired (e.g. sliding panels)
                    elif glen <= 2.0 * ppf and max(cA, cB) >= 0.5:
                        gcls = JGAP     # staggered junctions on both faces
                if gcls is None:
                    if cur:
                        chains.append(cur)
                    cur = []
                else:
                    cur.append((s + a0, e + a0, gcls))
                continue
            cur.append((s + a0, e + a0, c))
        if cur:
            chains.append(cur)
        for ch in chains:
            while ch and ch[-1][2] in (0, JGAP):
                ch.pop()
            while ch and ch[0][2] in (0, JGAP):
                ch.pop(0)
            if not ch:
                continue
            lo, hi = ch[0][0] - a0, ch[-1][1] - a0 + 1
            tv = tt[lo:hi][occ[lo:hi] > 0]
            # a wall is mostly fill; doors are the minority of its length
            wall_px = sum(e - s_ + 1 for (s_, e, c) in ch if c in (CMU, P2, P1))
            if wall_px < 0.35 * (ch[-1][1] - ch[0][0] + 1):
                continue
            sm = smooth_classes(ch, min_len=0.5 * ppf)
            if not sm:
                continue
            ts = []
            for (ss, ee, cc) in sm:
                lo2, hi2 = int(np.floor(ss)) - a0, int(np.ceil(ee)) - a0 + 1
                v = tt[lo2:hi2][(occ[lo2:hi2] > 0) & (occ[lo2:hi2] != OPEN)]
                if len(v) == 0:
                    v = tt[lo2:hi2][occ[lo2:hi2] > 0]
                ts.append(float(np.median(v)) if len(v) else float(np.median(tv)))
            out.append(dict(c=L['c'], raw=ch, segs=sm, ts=ts, t=float(np.median(tv)),
                            a=ch[0][0], b=ch[-1][1]))
    return out


def local_t(q, x):
    for (ss, ee, cc), t in zip(q['segs'], q['ts']):
        if ss - 1 <= x <= ee + 1:
            return t
    best = min(range(len(q['segs'])),
               key=lambda i: min(abs(q['segs'][i][0] - x), abs(q['segs'][i][1] - x)))
    return q['ts'][best]


def drop_nested(rl):
    keep = []
    for i, r in enumerate(rl):
        nested = False
        for j, q in enumerate(rl):
            if i == j or q['b'] - q['a'] <= r['b'] - r['a']:
                continue
            ov = min(r['b'], q['b']) - max(r['a'], q['a'])
            if ov < 0.8 * (r['b'] - r['a']):
                continue
            tq = local_t(q, (r['a'] + r['b']) / 2)
            if abs(r['c'] - q['c']) + r['t'] / 2 <= tq / 2 + 2.5:
                nested = True
                break
        if not nested:
            keep.append(r)
    return keep


def wall_segments(hruns, vruns, ppf, tol_ft=0.3):
    """Centerline pieces (orient, c, t, cls, start, end, run) with ends extended to the
    centerline of the perpendicular wall they run into."""
    tol = tol_ft * ppf
    tol_span = 0.6 * ppf     # the wall it runs into may itself be traced a little short at a corner

    def ends_for(r, perp):
        res = []
        for side in (0, 1):
            x = r['a'] - 0.5 if side == 0 else r['b'] + 0.5
            t_here = r['ts'][0] if side == 0 else r['ts'][-1]
            best = None
            for q in perp:
                if not (q['a'] - t_here / 2 - tol_span <= r['c'] <= q['b'] + t_here / 2 + tol_span):
                    continue
                tq = local_t(q, r['c'])
                if q['c'] - tq / 2 - tol <= x <= q['c'] + tq / 2 + tol:
                    dist = abs(q['c'] - x)
                    if best is None or dist < best[0]:
                        best = (dist, q['c'])
            res.append(x if best is None else best[1])
        return res

    out = []
    for rl, perp, orient in [(hruns, vruns, 'H'), (vruns, hruns, 'V')]:
        for r in rl:
            e0, e1 = ends_for(r, perp)
            sg = r['segs']
            bounds = [e0]
            for i in range(len(sg) - 1):
                bounds.append((sg[i][1] + sg[i + 1][0]) / 2.0)
            bounds.append(e1)
            for i, (s_, e_, c) in enumerate(sg):
                out.append((orient, r['c'], r['ts'][i], c, bounds[i], bounds[i + 1], r))
    return out


def raster_walls(shape, wsegs):
    m = np.zeros(shape, np.uint8)
    H, W = shape
    for ws in wsegs:
        (o, c, t, cls, s, e) = ws[:6]
        lo = c - t / 2.0
        hi = c + t / 2.0
        r0 = int(np.ceil(lo - 0.5)); r1 = int(np.floor(hi - 0.5))
        s0 = int(np.ceil(min(s, e) - 0.5)); s1 = int(np.floor(max(s, e) - 0.5))
        if r1 < r0 or s1 < s0:
            continue
        if o == 'H':
            m[max(r0, 0):min(r1 + 1, H), max(s0, 0):min(s1 + 1, W)] = 1
        else:
            m[max(s0, 0):min(s1 + 1, H), max(r0, 0):min(r1 + 1, W)] = 1
    return m


def detect_walls(d, ppf, bgl, heavy):
    hs = extract_segments(d, depth_min=heavy)
    vs = extract_segments(d.T, depth_min=heavy)
    hp = find_pieces(d, hs, ppf, bgl)
    vp = find_pieces(d.T, vs, ppf, bgl)
    hr = build_runs(group_lines(hp), ppf, hs)
    vr = build_runs(group_lines(vp), ppf, vs)
    hr = [r for r in hr if (r['b'] - r['a']) > 0.8 * ppf]
    vr = [r for r in vr if (r['b'] - r['a']) > 0.8 * ppf]
    hr = drop_nested(hr)
    vr = drop_nested(vr)
    return hr, vr, hs, vs


def building_from_walls(shape, wsegs, ppf):
    """Building footprint = largest wall network with its enclosed area filled."""
    W = raster_walls(shape, wsegs)
    n, lab, st, _ = cv2.connectedComponentsWithStats(W, connectivity=8)
    if n < 2:
        return None, None
    best = 1 + int(np.argmax(st[1:, 2].astype(np.int64) * st[1:, 3]))
    x, y, w, h = [int(v) for v in st[best, :4]]
    comp = (lab[y:y + h, x:x + w] == best)
    filled = ndi.binary_fill_holes(comp)
    if filled.sum() < 0.3 * w * h:
        # exterior ring not closed (undetected opening): bridge gaps progressively
        for k_ft in (2, 4, 8, 14):
            k = int(k_ft * ppf) | 1
            pad = k
            cp = np.pad(comp, pad).astype(np.uint8)
            closed = cv2.morphologyEx(cp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            filled = ndi.binary_fill_holes(closed)[pad:-pad, pad:-pad]
            if filled.sum() >= 0.3 * w * h:
                log('footprint closed with %d ft kernel' % k_ft)
                break
    fp = np.zeros(shape, bool)
    fp[y:y + h, x:x + w] = filled
    return fp, (x, y, w, h)


def run_inside(r, orient, fp):
    mid = (r['a'] + r['b']) / 2
    y, x = (r['c'], mid) if orient == 'H' else (mid, r['c'])
    yi, xi = int(round(y)), int(round(x))
    if not (0 <= yi < fp.shape[0] and 0 <= xi < fp.shape[1]):
        return False
    return bool(fp[yi, xi])


def is_exterior(r, orient, t, fp):
    """band whose one side is outside the building footprint"""
    H, W = fp.shape
    xs = np.linspace(r['a'], r['b'], 9)[1:-1]
    outs = 0
    for x in xs:
        for sgn in (-1, 1):
            yy = r['c'] + sgn * (t / 2 + 5)
            py, px = (yy, x) if orient == 'H' else (x, yy)
            py, px = int(round(py)), int(round(px))
            if not (0 <= py < H and 0 <= px < W) or not fp[py, px]:
                outs += 1
                break
    return outs >= 0.6 * len(xs)


# ----------------------------------------------------------------------------- floors

def noise_sigma(x):
    v = x[::3, ::3].ravel()
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def sheet_noise(d):
    """Noise of the line / speck detector responses, measured over bands across the whole sheet
    (mostly blank paper, so hatching inside the building does not inflate it)."""
    band = np.concatenate([d[y:y + 64] for y in range(0, d.shape[0] - 64, 256)], axis=0)
    f = band.astype(np.float32)
    fh = cv2.blur(f, (7, 1))
    dh = fh - np.minimum(np.roll(fh, 3, axis=0), np.roll(fh, -3, axis=0))
    g = cv2.GaussianBlur(f, (0, 0), 0.8)
    r = cv2.medianBlur(band, 9).astype(np.float32) - g
    return noise_sigma(dh), noise_sigma(r)


def oriented_line_maps(g, delta):
    """Thin dark line pixels in 4 orientations."""
    f = g.astype(np.float32)
    k = 7
    fh = cv2.blur(f, (k, 1))
    dh = fh - np.minimum(np.roll(fh, 3, axis=0), np.roll(fh, -3, axis=0))
    del fh
    res = {'h': dh < -delta}
    del dh
    fv = cv2.blur(f, (1, k))
    res['v'] = (fv - np.minimum(np.roll(fv, 3, axis=1), np.roll(fv, -3, axis=1))) < -delta
    del fv
    kd1 = np.zeros((k, k), np.float32)
    for i in range(k):
        kd1[k - 1 - i, i] = 1.0 / k
    kd2 = np.zeros((k, k), np.float32)
    for i in range(k):
        kd2[i, i] = 1.0 / k
    fd = cv2.filter2D(f, -1, kd1)
    res['d1'] = (fd - np.minimum(np.roll(np.roll(fd, 2, axis=0), 2, axis=1),
                                 np.roll(np.roll(fd, -2, axis=0), -2, axis=1))) < -delta
    fd = cv2.filter2D(f, -1, kd2)
    res['d2'] = (fd - np.minimum(np.roll(np.roll(fd, 2, axis=0), -2, axis=1),
                                 np.roll(np.roll(fd, -2, axis=0), 2, axis=1))) < -delta
    return res


def speck_centers(d, thr):
    g = cv2.GaussianBlur(d.astype(np.float32), (0, 0), 0.8)
    bg = cv2.medianBlur(d, 9).astype(np.float32)
    r = bg - g
    m = (r > thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    ok = (st[:, 2] <= 6) & (st[:, 3] <= 6) & (st[:, 4] <= 20)
    ok[0] = False
    return cen[ok]


def classify_finish(f):
    h, v, d1, d2, sp = f['h'], f['v'], f['d1'], f['d2'], f['specks']
    if h > 4 and h > 3 * v and f['hlpf'] > 2.6:
        return 'WOM-1'
    if d1 > 3.5 and d2 > 3.5 and min(d1, d2) > 0.5 * max(d1, d2):
        return 'CT-1'
    if d1 > 3.5 and d2 < 0.4 * d1:
        return 'EP-1'
    if d2 > 3.5 and d1 < 0.4 * d2:
        return 'CPT-1'
    if h > 4 and v > 4 and 0.6 < h / v < 1.6 and f['hlpf'] > 0.7 and f['vlpf'] > 0.7:
        return 'VCT-1'
    if h > 4 and h > 2 * v:
        return 'LVT-1'
    return 'SC-1' if sp > 0.4 else 'PC-1'


def close_boundary_gaps(barrier, hsegs, vsegs, interior, ppf, max_gap_ft=15.0):
    """Finish-transition lines never end in open floor; collinear free ends facing each other
    across a gap mean the line is occluded (e.g. by a fixture or a tag) -> bridge the gap."""
    H, W = barrier.shape
    closed = 0
    for orient, segs in (('H', hsegs), ('V', vsegs)):
        B = barrier if orient == 'H' else barrier.T
        I = interior if orient == 'H' else interior.T
        ends = []
        for (y, a, b, dp, n) in segs:
            if b - a < 2 * ppf:
                continue
            yi = int(round(y))
            if not (2 <= yi < B.shape[0] - 2):
                continue
            mid = (a + b) // 2
            if not I[yi, min(max(mid, 0), B.shape[1] - 1)]:
                continue
            for side, x in ((-1, a), (1, b)):
                lo, hi = (x - 10, x + 2) if side < 0 else (x - 2, x + 10)
                if lo < 0 or hi >= B.shape[1] or yi - 8 < 0 or yi + 9 > B.shape[0]:
                    continue
                box = B[yi - 8:yi + 9, lo:hi + 1].copy()
                # ignore the line's own blurred tail near its end
                tail = (np.arange(lo, hi + 1) >= x - 4) if side < 0 else (np.arange(lo, hi + 1) <= x + 4)
                box[6:11, tail] = False
                if not box.any() and I[yi, lo:hi + 1].all():
                    ends.append((y, x, side))
        used = set()
        for i, (y1, xa, s1) in enumerate(ends):
            if s1 != 1 or i in used:
                continue
            best = None
            for j, (y2, xb, s2) in enumerate(ends):
                if s2 != -1 or j in used or abs(y1 - y2) > 2.0:
                    continue
                gap = xb - xa
                if 0 < gap <= max_gap_ft * ppf and (best is None or gap < best[0]):
                    best = (gap, j)
            if best is not None:
                j = best[1]
                used.add(i); used.add(j)
                y2, xb, _ = ends[j]
                yy = int(round((y1 + y2) / 2))
                B[yy - 1:yy + 2, xa:xb + 1] = True
                closed += 1
                log('closed boundary gap %s at %.0f: %d-%d' % (orient, yy, xa, xb))
    return closed


def floor_takeoff(d, ppf, fp, wall_mask, T, hsegs=(), vsegs=(), fixture_boxes=(), noise=(2.4, 1.85)):
    interior = fp & (wall_mask == 0)
    fixt = np.zeros(d.shape, bool)
    for (x0, y0, x1, y1) in fixture_boxes:
        fixt[max(0, int(y0) - 3):int(y1) + 4, max(0, int(x0) - 3):int(x1) + 4] = True
    g = cv2.GaussianBlur(d, (0, 0), 1.5)
    barrier = g < T
    try:
        close_boundary_gaps(barrier, hsegs, vsegs, interior, ppf)
    except Exception:
        traceback.print_exc()
    free = interior & ~barrier
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    amin = 3.0 * ppf * ppf
    sig = [i for i in range(1, n2) if st2[i, 4] >= amin]
    keep = np.zeros(n2, np.int32)
    for k, i in enumerate(sig):
        keep[i] = k + 1
    L = keep[lab2]
    if len(sig) and (interior & (L == 0)).any():
        _, (iy, ix) = ndi.distance_transform_edt(L == 0, return_indices=True)
        L = np.where(interior, L[iy, ix], 0)
    delta = max(7.0, 5.7 * noise[0])
    sthr = max(8.0, 8.0 * noise[1])
    log('line delta %.1f speck thr %.1f' % (delta, sthr))
    maps = oriented_line_maps(d, delta)
    sp = speck_centers(d, sthr)
    spl = L[np.clip(sp[:, 1].round().astype(int), 0, d.shape[0] - 1),
            np.clip(sp[:, 0].round().astype(int), 0, d.shape[1] - 1)] if len(sp) else np.zeros(0, int)
    spc = np.bincount(spl, minlength=len(sig) + 1)
    areas = np.bincount(L.ravel(), minlength=len(sig) + 1)
    tot = {k: 0.0 for k in FINISHES}
    kern = np.ones((7, 7), np.uint8)
    objs = ndi.find_objects(L)
    for k in range(1, len(sig) + 1):
        sl = objs[k - 1]
        if sl is None:
            continue
        m = (L[sl] == k)
        area = areas[k] / ppf ** 2
        mi = cv2.erode(m.astype(np.uint8), kern).astype(bool)
        mf = mi & ~fixt[sl]                  # texture statistics ignore fixture footprints
        if mf.sum() >= max(50, 0.15 * mi.sum()):
            mi = mf
        if mi.sum() < 50:
            mi = m
        f = {kk: float(maps[kk][sl][mi].mean() * 100) for kk in maps}
        hm = maps['h'][sl] & mi
        f['hlpf'] = float((hm[1:] & ~hm[:-1]).sum()) / max(int(mi.sum()), 1) * ppf
        vm = maps['v'][sl] & mi
        f['vlpf'] = float((vm[:, 1:] & ~vm[:, :-1]).sum()) / max(int(mi.sum()), 1) * ppf
        vis_area = max(float((m & ~fixt[sl]).sum()), 0.15 * float(m.sum())) / ppf ** 2
        f['specks'] = spc[k] / max(vis_area, 1e-6)
        c = classify_finish(f)
        tot[c] += area
        log('region %d %s %.1f sf' % (k, c, area), {kk: round(vv, 2) for kk, vv in f.items()})
    return tot


# ----------------------------------------------------------------------------- doors

def find_openings(runs, ppf, min_w_ft=1.5):
    """Door openings: stretches of a wall run without fill (white band between continuing face
    lines, or a gap where both face lines continue). Junction gaps don't count as opening."""
    ops = []
    for r in runs:
        parts = []
        for (s0, e0, c) in r['raw']:
            short = c in (CMU, P2, P1) and (e0 - s0 + 1) < 0.5 * ppf
            isopen = c in (0, OPEN, JGAP) or short
            if parts and parts[-1][2] == isopen:
                parts[-1][1] = e0
                parts[-1][3].append((s0, e0, c))
            else:
                parts.append([s0, e0, isopen, [(s0, e0, c)]])
        for (s0, e0, iso, items) in parts:
            if not iso:
                continue
            real = [(a, b) for (a, b, c) in items if c in (0, OPEN)]
            if not real or sum(b - a + 1 for (a, b) in real) < 1.2 * ppf:
                continue
            a, b = real[0][0], real[-1][1]
            if (b - a + 1) <= min_w_ft * ppf:
                continue
            t = r['t']
            for (ss, ee, cc), tt in zip(r['segs'], r['ts']):
                if ss - 2 <= (a + b) / 2 <= ee + 2:
                    t = tt
            ops.append(dict(c=r['c'], t=t, a=a, b=b))
    return ops


def opening_darkness(img_o, op, bgl):
    """median over the opening of the darkness integrated across the band (faces + anything inside)"""
    c, t, a, b = op['c'], op['t'], op['a'], op['b']
    r0 = int(np.floor(c - t / 2 - 3)); r1 = int(np.ceil(c + t / 2 + 3))
    if b - a < 8 or r0 < 0 or r1 >= img_o.shape[0]:
        return None
    blk = bgl - img_o[r0:r1 + 1, a + 3:b - 2].astype(np.float32)
    return float(np.median(np.clip(blk, 0, None).sum(axis=0)))


def is_wall_face(seg, perp_runs):
    x, y0, y1 = seg[0], seg[1], seg[2]
    for q in perp_runs:
        if y1 < q['a'] - 3 or y0 > q['b'] + 3:
            continue
        t = local_t(q, (max(y0, q['a']) + min(y1, q['b'])) / 2)
        if abs(x - (q['c'] - t / 2)) <= 2.5 or abs(x - (q['c'] + t / 2)) <= 2.5:
            return True
    return False


def classify_door(op, par_segs, perp_segs, ppf, ref, dark_ratio=None, perp_runs=()):
    c, t, a, b = op['c'], op['t'], op['a'], op['b']
    w = b - a + 1
    top, bot = c - t / 2, c + t / 2
    panels = [s for s in par_segs if top < s[0] < bot and abs(s[0] - top) > 1.0 and abs(s[0] - bot) > 1.0
              and s[3] > 0.6 * ref and (s[2] - s[1]) > 1.0 * ppf
              and min(s[2], b + 0.5 * w) - max(s[1], a - 0.5 * w) > 0.5 * ppf]
    above = [s for s in panels if s[0] < c]
    below = [s for s in panels if s[0] > c]
    if above and below:
        return 'auto_sliding'
    if dark_ratio is not None and dark_ratio > 1.4 and panels:
        return 'auto_sliding'
    if dark_ratio is not None and dark_ratio > 1.5:
        return 'auto_sliding'
    jt = 0.35 * ppf
    lmin = 1.0 * ppf
    near = 0.6 * ppf
    hinges = set()
    for s in perp_segs:
        x, y0, y1, dp = s[:4]
        if dp < 0.5 * ref:
            continue
        side = None
        if abs(x - a) <= jt:
            side = 'L'
        elif abs(x - b) <= jt:
            side = 'R'
        if side is None:
            continue
        if perp_runs and is_wall_face(s, perp_runs):
            continue
        # a leaf reaches about the opening width (single) or half of it (pair) from the wall face
        if y1 >= top - near and y0 <= top - lmin and 0.35 * w <= top - y0 <= 1.3 * w:
            hinges.add((side, 'up'))
        if y0 <= bot + near and y1 >= bot + lmin and 0.35 * w <= y1 - bot <= 1.3 * w:
            hinges.add((side, 'down'))
    sides = set(h[0] for h in hinges)
    if len(sides) == 2:
        return 'pair'
    if len(sides) == 1:
        return 'single'
    wft = w / ppf
    if wft >= 7.0:
        return 'overhead_coiling'
    return 'pair' if wft >= 5.0 else 'single'


def detect_doors(dc, offset, hr, vr, ppf, ref, bgl):
    ox, oy = offset
    hs = [(y + oy, a + ox, b + ox, dp, n) for (y, a, b, dp, n) in extract_segments(dc, depth_min=0.3 * ref, min_len=10)]
    vs = [(x + ox, a + oy, b + oy, dp, n) for (x, a, b, dp, n) in extract_segments(dc.T, depth_min=0.3 * ref, min_len=10)]
    cnt = {k: 0 for k in DOOR_TYPES}
    dct = np.ascontiguousarray(dc.T)
    items = []
    for runs, prun, par, perp, img_o, (oa, oc) in [(hr, vr, hs, vs, dc, (ox, oy)), (vr, hr, vs, hs, dct, (oy, ox))]:
        for op in find_openings(runs, ppf):
            # opening in crop-frame coordinates for the darkness measure
            opc = dict(op, c=op['c'] - oc, a=op['a'] - oa, b=op['b'] - oa)
            items.append((op, par, perp, opening_darkness(img_o, opc, bgl), prun))
    darks = [it[3] for it in items if it[3] is not None]
    base = float(np.percentile(darks, 35)) if len(darks) >= 3 else None   # typical swing-door opening
    for op, par, perp, dk, prun in items:
        ratio = dk / base if (base and dk is not None) else None
        typ = classify_door(op, par, perp, ppf, ref, ratio, prun)
        cnt[typ] += 1
        log('door', {k: round(float(v), 1) for k, v in op.items()}, 'ratio', None if ratio is None else round(ratio, 2), typ)
    return cnt


# ----------------------------------------------------------------------------- fixtures

def find_strips(segs, w_px, wtol, min_len, dmin=0):
    segs = sorted([s for s in segs if s[3] >= dmin], key=lambda s: s[0])
    ys = np.array([s[0] for s in segs])
    out = []
    for i, s1 in enumerate(segs):
        j0 = np.searchsorted(ys, s1[0] + w_px - wtol)
        j1 = np.searchsorted(ys, s1[0] + w_px + wtol)
        for j in range(j0, j1):
            s2 = segs[j]
            a, b = max(s1[1], s2[1]), min(s1[2], s2[2])
            if b - a >= min_len:
                out.append((s1[0], s2[0], a, b, s1, s2))
    return out


def darkness_profile(img, y1, y2, a, b, bgl, excl_center=False, inset=3):
    r0 = int(np.ceil(y1 + inset)); r1 = int(np.floor(y2 - inset))
    if r1 < r0:
        r0 = r1 = int(round((y1 + y2) / 2))
    blk = bgl - img[r0:r1 + 1, a:b + 1].astype(np.float32)
    if excl_center:
        yc = (y1 + y2) / 2
        rows = np.arange(r0, r1 + 1)
        keep = np.abs(rows - yc) > 3
        if keep.any():
            blk = blk[keep]
    return blk.mean(axis=0)


def line_dark(imgf, p0, p1, bgl):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    L = np.linalg.norm(p1 - p0)
    n = max(2, int(L))
    t = np.linspace(0.12, 0.88, n)
    pts = p0[None] + (p1 - p0)[None] * t[:, None]
    dvec = (p1 - p0) / max(L, 1e-6)
    nvec = np.array([-dvec[1], dvec[0]])
    vals = []
    for off in (-1.5, -0.75, 0, 0.75, 1.5):
        q = pts + nvec[None] * off
        xs = np.clip(q[:, 0], 0, imgf.shape[1] - 1).astype(np.float32)
        ys = np.clip(q[:, 1], 0, imgf.shape[0] - 1).astype(np.float32)
        v = cv2.remap(imgf, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR)[0]
        vals.append(bgl - v)
    return float(np.median(np.max(np.array(vals), axis=0)))


def circle_dark(imgf, cx, cy, r, bgl, n=72):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vals = []
    for dr in (-1.5, -0.75, 0, 0.75, 1.5):
        xs = (cx + (r + dr) * np.cos(ang)).astype(np.float32)
        ys = (cy + (r + dr) * np.sin(ang)).astype(np.float32)
        v = cv2.remap(imgf, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR)[0]
        vals.append(bgl - v)
    return float(np.median(np.max(np.array(vals), axis=0)))


def find_peaks(p, thr, min_sep, prom=0.0, prom_r=5):
    n = len(p)
    cand = []
    for k in range(n):
        if p[k] < thr or (k > 0 and p[k] < p[k - 1]) or (k < n - 1 and p[k] < p[k + 1]):
            continue
        lo = p[max(0, k - prom_r):k + prom_r + 1].min()
        if p[k] - lo < prom:
            continue
        cand.append(k)
    idx = []
    for k in sorted(cand, key=lambda k: -p[k]):
        if all(abs(k - j) >= min_sep for j in idx):
            idx.append(k)
    return sorted(idx)


def to_img(orient, x, y):
    return (x, y) if orient == 'H' else (y, x)


def gondola_body_ok(img_o, y1, y2, a, b, bgl, side, noise):
    """A real gondola run is a white-filled outline split lengthwise by a center line; a strip
    formed across an aisle is not."""
    yc = int(round((y1 + y2) / 2))
    a, b = int(a), int(b)
    if b - a < 10:
        return False
    cen = (bgl - img_o[yc - 1:yc + 2, a:b + 1].astype(np.float32)).max(axis=0)
    if np.median(cen) > 0.3 * side:
        return True
    r0, r1 = int(np.ceil(y1 + 4)), int(np.floor(y2 - 4))
    blk = img_o[r0:r1 + 1, a:b + 1].astype(np.float32)
    rows = np.abs(np.arange(r0, r1 + 1) - yc) > 3
    blk = blk[rows]
    # hatch-free interior: dark-pixel fraction no higher than noise would give
    return float((blk < bgl - max(25.0, 6 * noise)).mean()) < 0.02


def strip_pieces(img_o, imgf, segs, ppf, bgl, orient, w_ft, wtol_ft, min_len_ft, dmin, dmax, inside,
                 excl_center=False, gondola=False, noise=3.0):
    out = []
    for (y1, y2, a, b, s1, s2) in find_strips([s for s in segs if s[3] <= dmax], w_ft * ppf,
                                               wtol_ft * ppf, min_len_ft * ppf, dmin=dmin):
        a0 = int(a - 0.3 * ppf); b0 = int(b + 0.3 * ppf)
        if a0 < 0 or b0 >= img_o.shape[1]:
            continue
        cx, cy = to_img(orient, (a + b) / 2, (y1 + y2) / 2)
        if not inside[int(cy), int(cx)]:
            continue
        side = min(s1[3], s2[3])
        if gondola:
            if side < 0.8 * max(s1[3], s2[3]):
                continue
            if not gondola_body_ok(img_o, y1, y2, a, b, bgl, side, noise):
                continue
        p = darkness_profile(img_o, y1, y2, a0, b0, bgl, excl_center=excl_center)
        pk = find_peaks(p, 0.6 * side, max(3, int(0.15 * ppf)), prom=0.35 * side, prom_r=max(3, int(0.3 * ppf)))
        pk = [k + a0 for k in pk if a - 0.25 * ppf <= k + a0 <= b + 0.25 * ppf]
        sA = to_img(orient, pk[0], y1) if pk else None
        sB = to_img(orient, pk[-1], y2) if pk else None
        sbox = (min(sA[0], sB[0]), min(sA[1], sB[1]), max(sA[0], sB[0]), max(sA[1], sB[1])) if pk else None
        for k0, k1 in zip(pk[:-1], pk[1:]):
            pA = to_img(orient, k0, y1); pB = to_img(orient, k1, y2)
            pC = to_img(orient, k0, y2); pD = to_img(orient, k1, y1)
            d1 = line_dark(imgf, pA, pB, bgl)
            d2 = line_dark(imgf, pC, pD, bgl)
            xs = sorted([pA[0], pB[0]]); ys = sorted([pA[1], pB[1]])
            out.append(dict(orient=orient, along=(k1 - k0) / ppf, across=(y2 - y1) / ppf, d1=d1, d2=d2,
                            side=side, box=(xs[0], ys[0], xs[1], ys[1]), sbox=sbox))
    return out


def iou(b1, b2):
    ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = ix * iy
    u = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / u if u > 0 else 0


def dedupe(items):
    out = []
    for it in items:
        if all(iou(it['box'], o['box']) < 0.5 for o in out):
            out.append(it)
    return out


def detect_fixtures(d, ppf, bgl, inside, ref, noise=2.4):
    imgf = d.astype(np.float32)
    dmin_seg = 0.24 * ref
    hs = merge_collinear(extract_segments(d, depth_min=dmin_seg, min_len=8))
    vs = merge_collinear(extract_segments(d.T, depth_min=dmin_seg, min_len=8))
    dmin = 0.3 * ref
    dmax = 0.75 * ref
    res = {k: [] for k in FIXTURE_TYPES}
    for orient, segs, img_o in [('H', hs, d), ('V', vs, d.T)]:
        # gondola runs: 4'-0" deep
        for pc in strip_pieces(img_o, imgf, segs, ppf, bgl, orient, 4.0, 0.3, 1.2, dmin, dmax, inside,
                               excl_center=True, gondola=True, noise=noise):
            diag = max(pc['d1'], pc['d2']) > 0.35 * pc['side']
            both = min(pc['d1'], pc['d2']) > 0.35 * pc['side']
            L = pc['along']
            if 0.9 <= L <= 2.6 and diag and not both:
                res['endcaps'].append(pc)
            elif abs(L - 3.0) <= 0.4 and not diag:
                res['gondola_sections_3ft'].append(pc)
            elif abs(L - 4.0) <= 0.4 and not diag:
                res['gondola_sections_4ft'].append(pc)
            elif abs(L - 5.0) <= 0.4 and both:
                res['produce_tables'].append(pc)
        # produce tables 5'-0" x 4'-0" with an X
        for pc in strip_pieces(img_o, imgf, segs, ppf, bgl, orient, 5.0, 0.3, 3.0, dmin, dmax, inside):
            both = min(pc['d1'], pc['d2']) > 0.35 * pc['side']
            if abs(pc['along'] - 4.0) <= 0.4 and both:
                res['produce_tables'].append(pc)
        # self-checkout kiosk: 3'-0" square with a circle
        for pc in strip_pieces(img_o, imgf, segs, ppf, bgl, orient, 3.0, 0.3, 2.0, dmin, dmax, inside):
            if abs(pc['along'] - 3.0) <= 0.4:
                x0, y0, x1, y1 = pc['box']
                cd = circle_dark(imgf, (x0 + x1) / 2, (y0 + y1) / 2, 0.75 * ppf, bgl)
                both = min(pc['d1'], pc['d2']) > 0.35 * pc['side']
                if cd > 0.35 * pc['side'] and not both:
                    res['self_checkouts'].append(pc)
        # checkout stand: 2'-6" wide counter whose end box carries an X
        for pc in strip_pieces(img_o, imgf, segs, ppf, bgl, orient, 2.5, 0.25, 6.0, dmin, dmax, inside):
            both = min(pc['d1'], pc['d2']) > 0.35 * pc['side']
            if abs(pc['along'] - 3.0) <= 0.5 and both:
                res['checkouts'].append(pc)
        # reach-in cooler doors: 0.6' deep strip of doors with a diagonal each
        for pc in strip_pieces(img_o, imgf, segs, ppf, bgl, orient, 0.6, 0.12, 1.5, dmin, dmax, inside):
            diag = max(pc['d1'], pc['d2']) > 0.35 * pc['side']
            if 1.8 <= pc['along'] <= 3.2 and diag:
                res['cooler_doors'].append(pc)
    out = {}
    boxes = []
    for k in FIXTURE_TYPES:
        kept = dedupe(res[k])
        out[k] = len(kept)
        for pc in kept:
            boxes.append(pc['sbox'] if k == 'checkouts' and pc.get('sbox') else pc['box'])
    return out, boxes


# ----------------------------------------------------------------------------- main

def takeoff(pdf_path):
    result = {
        'floor_sf': {k: 0.0 for k in FINISHES},
        'wall_lf': {k: 0.0 for k in WALL_TYPES},
        'doors': {k: 0 for k in DOOR_TYPES},
        'fixtures': {k: 0 for k in FIXTURE_TYPES},
    }
    im = load_image(pdf_path)
    angle = estimate_skew(im)
    d = deskew(im, angle)
    del im
    bgl = float(np.median(d[::7, ::7]))
    noise = sheet_noise(d)
    log('sheet noise', noise)
    try:
        bar = find_scale_bar(d)
        if bar is None:      # sheet scanned upside down: the bar reads right-to-left
            bar = find_scale_bar(np.ascontiguousarray(d[:, ::-1]))
    except Exception:
        traceback.print_exc()
        bar = None
    ppf_bar, bar_ok = bar if bar is not None else (None, False)
    try:
        ppf_border = border_scale(d)
    except Exception:
        ppf_border = None
    ppf = ppf_bar
    if ppf is None:
        ppf = ppf_border
    elif ppf_border is not None and abs(ppf / ppf_border - 1) > 0.015 and not bar_ok:
        ppf = ppf_border
    log('angle %.3f ppf bar %s border %s -> %s' % (angle, ppf_bar, ppf_border, ppf))
    if ppf is None:
        ppf = 24.5      # nominal 1/8" = 1'-0" at 200 dpi, typical print reduction

    # heavy-line reference: the longest lines on the sheet are border/exterior walls
    hs0 = extract_segments(d, depth_min=60, min_len=40)
    longest = sorted(hs0, key=lambda s: -(s[2] - s[1]))[:4]
    ref0 = float(np.median([s[3] for s in longest])) if longest else 170.0
    heavy = 0.62 * ref0
    log('ref0 %.1f heavy %.1f' % (ref0, heavy))

    best = None
    for hf in (1.0, 0.8, 1.2, 0.65):
        hr, vr, hs, vs = detect_walls(d, ppf, bgl, heavy * hf)
        wsegs = wall_segments(hr, vr, ppf)
        fp, bbox = building_from_walls(d.shape, wsegs, ppf)
        if fp is None:
            continue
        wft, hft = bbox[2] / ppf, bbox[3] / ppf
        ok = 40 <= wft <= 600 and 40 <= hft <= 600 and fp.sum() >= 0.3 * bbox[2] * bbox[3]
        log('heavy x%.2f building %.1f x %.1f ft ok=%s' % (hf, wft, hft, ok))
        if ok:
            best = (hr, vr, hs, vs, fp, bbox)
            break
        if best is None:
            best = (hr, vr, hs, vs, fp, bbox)
    if best is None:
        raise RuntimeError('no building found')
    hr, vr, hs, vs, fp, bbox = best

    def wall_depth(hs, bbox):
        bx, by, bw, bh = bbox
        dps = [dp for (y, a, b, dp, n) in hs
               if b - a > 20 * ppf and by - 5 <= y <= by + bh + 5 and a >= bx - 10 and b <= bx + bw + 10]
        return float(np.median(dps)) if len(dps) >= 2 else None

    # second pass if the face lines of the walls are much lighter/darker than the sheet border
    wd = wall_depth(hs, bbox)
    if wd is not None and abs(0.62 * wd / heavy - 1) > 0.15:
        log('re-detecting walls with heavy %.1f (wall depth %.1f)' % (0.62 * wd, wd))
        hr2, vr2, hs2, vs2 = detect_walls(d, ppf, bgl, 0.62 * wd)
        fp2, bbox2 = building_from_walls(d.shape, wall_segments(hr2, vr2, ppf), ppf)
        if fp2 is not None and fp2.sum() >= 0.3 * bbox2[2] * bbox2[3] and \
                abs(bbox2[2] - bbox[2]) < 0.05 * bbox[2] and abs(bbox2[3] - bbox[3]) < 0.05 * bbox[3]:
            hr, vr, hs, vs, fp, bbox = hr2, vr2, hs2, vs2, fp2, bbox2
    hr = [r for r in hr if run_inside(r, 'H', fp)]
    vr = [r for r in vr if run_inside(r, 'V', fp)]
    wsegs = wall_segments(hr, vr, ppf)

    ext_runs = set()
    for (o, c, t, cls, s, e, r) in wsegs:
        if cls == CMU and is_exterior(r, o, t, fp):
            ext_runs.add(id(r))
    ref = wall_depth(hs, bbox) or ref0
    log('wall ref depth %.1f' % ref)

    walls = {k: 0.0 for k in WALL_TYPES}
    for (o, c, t, cls, s, e, r) in wsegs:
        L = abs(e - s) / ppf
        if cls == CMU:
            name = 'EXT' if id(r) in ext_runs else 'P3'
        else:
            name = {P2: 'P2', P1: 'P1'}[cls]
        walls[name] += L
        log('wall %s %s c=%.1f t=%.1f %.1f-%.1f %.2f ft' % (o, name, c, t, s, e, L))
    result['wall_lf'] = {k: round(walls[k], 2) for k in WALL_TYPES}

    wall_mask = raster_walls(d.shape, wsegs)
    # later stages work on a crop around the building
    bx, by, bw, bh = bbox
    mg = int(3 * ppf)
    x0, y0 = max(0, bx - mg), max(0, by - mg)
    x1, y1 = min(d.shape[1], bx + bw + mg), min(d.shape[0], by + bh + mg)
    dc = np.ascontiguousarray(d[y0:y1, x0:x1])
    fpc = fp[y0:y1, x0:x1]
    wmc = wall_mask[y0:y1, x0:x1]
    del wall_mask
    boxes = []
    try:
        interior = fpc & (wmc == 0)
        result['fixtures'], boxes = detect_fixtures(dc, ppf, bgl, interior, ref, noise[0])
    except Exception:
        traceback.print_exc()
    try:
        T = bgl - 0.42 * ref
        hsc = [(y - y0, a - x0, b - x0, dp, n) for (y, a, b, dp, n) in hs]
        vsc = [(x - x0, a - y0, b - y0, dp, n) for (x, a, b, dp, n) in vs]
        result['floor_sf'] = {k: round(v, 1) for k, v in
                              floor_takeoff(dc, ppf, fpc, wmc, T, hsc, vsc, boxes, noise).items()}
    except Exception:
        traceback.print_exc()
    try:
        result['doors'] = detect_doors(dc, (x0, y0), hr, vr, ppf, ref, bgl)
    except Exception:
        traceback.print_exc()
    return result


def main():
    if len(sys.argv) != 3:
        print('usage: takeoff.py <scan.pdf> <takeoff.json>', file=sys.stderr)
        sys.exit(2)
    pdf, out = sys.argv[1], sys.argv[2]
    try:
        res = takeoff(pdf)
    except Exception:
        traceback.print_exc()
        res = {
            'floor_sf': {k: 0.0 for k in FINISHES},
            'wall_lf': {k: 0.0 for k in WALL_TYPES},
            'doors': {k: 0 for k in DOOR_TYPES},
            'fixtures': {k: 0 for k in FIXTURE_TYPES},
        }
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
