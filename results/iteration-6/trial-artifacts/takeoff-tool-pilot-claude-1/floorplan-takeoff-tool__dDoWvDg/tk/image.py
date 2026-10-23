"""Loading, deskewing and scale calibration of scanned A101 sheets."""
import numpy as np
import cv2


def load_pdf_gray(path):
    """Return the scanned sheet as a uint8 grayscale array (page orientation)."""
    import fitz
    doc = fitz.open(path)
    page = doc[0]
    prect = page.rect
    best = None
    for info in page.get_image_info(xrefs=True):
        if not info.get('xref'):
            continue
        if best is None or info['width'] * info['height'] > best['width'] * best['height']:
            best = info
    img = None
    if best is not None and best['width'] * best['height'] > 4e6 and page.rotation % 360 == 0:
        a, b, c, d, e, f = best['transform']
        x0, y0, x1, y1 = best['bbox']
        plain = abs(b) < 1e-6 and abs(c) < 1e-6 and a > 0 and d > 0
        full = (abs(x0 - prect.x0) < 2 and abs(y0 - prect.y0) < 2 and
                abs(x1 - prect.x1) < 2 and abs(y1 - prect.y1) < 2)
        if plain and full:
            try:
                data = doc.extract_image(best['xref'])
                arr = np.frombuffer(data['image'], np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if img is not None and (img.shape[1], img.shape[0]) != (best['width'], best['height']):
                    img = None
            except Exception:
                img = None
    if img is None:
        # render the page; aim for the scan's native resolution (about 200 dpi)
        zoom = 200.0 / 72.0
        if best is not None:
            bw = best['bbox'][2] - best['bbox'][0]
            if bw > 0:
                zoom = min(max(best['width'] / bw, 150.0 / 72.0), 300.0 / 72.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, -1)[:, :, 0].copy()
    return img


def _coarse_angle(g):
    small = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small < 160)
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    if len(xs) > 400000:
        sel = np.random.RandomState(0).choice(len(xs), 400000, replace=False)
        xs, ys = xs[sel], ys[sel]

    def score(a):
        t = np.radians(a)
        yr = -xs * np.sin(t) + ys * np.cos(t)
        xr = xs * np.cos(t) + ys * np.sin(t)
        hy = np.bincount(np.round(yr - yr.min()).astype(int))
        hx = np.bincount(np.round(xr - xr.min()).astype(int))
        return (hy.astype(np.float64) ** 2).sum() + (hx.astype(np.float64) ** 2).sum()

    angs = np.arange(-5, 5.001, 0.05)
    sc = [score(a) for a in angs]
    a0 = angs[int(np.argmax(sc))]
    angs2 = np.arange(a0 - 0.06, a0 + 0.0601, 0.005)
    sc2 = [score(a) for a in angs2]
    return float(angs2[int(np.argmax(sc2))])


def rotate(g, ang):
    H, W = g.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
    return cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)


def _line_slopes(r):
    """Slopes of long straight lines in a nearly deskewed image (both axes)."""
    b = (r < 128).astype(np.uint8)
    slopes, weights = [], []
    for axis in (0, 1):
        bb = b if axis == 0 else np.ascontiguousarray(b.T)
        rr = r if axis == 0 else np.ascontiguousarray(r.T)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (301, 1))
        op = cv2.morphologyEx(bb, cv2.MORPH_OPEN, k)
        n, lab, st, cen = cv2.connectedComponentsWithStats(op, 8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if w < 1200 or h > 10:
                continue
            yc = y + h / 2.0
            pts = []
            for xx in range(x + 10, x + w - 10, 10):
                y0 = int(round(yc)) - 7
                if y0 < 0 or y0 + 15 > rr.shape[0]:
                    continue
                prof = rr[y0:y0 + 15, xx].astype(np.float64)
                wgt = np.clip(235 - prof, 0, None)
                if wgt.sum() < 100:
                    continue
                pts.append((xx, y0 + (np.arange(15) * wgt).sum() / wgt.sum()))
            pts = np.array(pts)
            if len(pts) < 20:
                continue
            keep = np.ones(len(pts), bool)
            for _ in range(5):
                p = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
                res = pts[:, 1] - np.polyval(p, pts[:, 0])
                sd = 1.4826 * np.median(np.abs(res[keep])) + 0.02
                keep = np.abs(res) < 3 * sd
            slopes.append(p[0] if axis == 0 else -p[0])
            weights.append(w)
    return np.array(slopes), np.array(weights, float)


def _wmedian(v, w):
    o = np.argsort(v)
    v = v[o]
    w = w[o]
    c = np.cumsum(w)
    return v[np.searchsorted(c, c[-1] / 2.0)]


def deskew(g):
    a = _coarse_angle(g)
    r = rotate(g, a)
    s, w = _line_slopes(r)
    if len(s):
        a = a + float(np.degrees(np.arctan(_wmedian(s, w))))
        r = rotate(g, a)
    return r, a


def _crossings(band, lev, off):
    xs = []
    for i in range(len(band) - 1):
        a, b = band[i], band[i + 1]
        if (a - lev) * (b - lev) < 0:
            xs.append(off + i + (lev - a) / (b - a))
    return xs


def find_scale(r):
    """Locate the graphic scale bar (0-4 black, 4-8 white, 8-16 black, 16-32 white)
    and return pixels per foot plus diagnostic info."""
    b = (r < 128).astype(np.uint8)
    op = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(op, 8)
    solids = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 12 <= h <= 60 and w >= 30 and a > 0.85 * w * h and w > 2 * h:
            solids.append((x, y, w, h))
    best = None
    for (x1, y1, w1, h1) in solids:
        for (x2, y2, w2, h2) in solids:
            if x2 <= x1:
                continue
            if abs(y1 - y2) > 3 or abs(h1 - h2) > 3:
                continue
            # second solid twice as long, starting one segment-length after first ends
            if not (1.8 < w2 / float(w1) < 2.2):
                continue
            gap = x2 - (x1 + w1)
            if not (0.8 * w1 < gap < 1.2 * w1):
                continue
            cand = (x1, y1, w1, h1, x2, w2)
            if best is None or w1 > best[2]:
                best = cand
    if best is None:
        return _check_bar_scale(r, solids)
    x1, y1, w1, h1, x2, w2 = best
    rr = r.astype(np.float64)
    band = rr[y1 + 4:y1 + h1 - 4, max(0, x1 - 40):x1 + 9 * w1].mean(0)
    off = max(0, x1 - 40)
    lo = np.percentile(band, 2)
    hi = np.percentile(band, 90)
    xs = _crossings(band, (lo + hi) / 2, off)
    # first four crossings: L0, R1, L2, R2
    if len(xs) < 4:
        return None
    L0, R1, L2, R2 = xs[:4]
    unit8 = L2 - L0          # 8 ft, stroke bias cancels
    unit12 = R2 - R1         # 12 ft, stroke bias cancels
    ppf_a = unit8 / 8.0
    ppf_b = unit12 / 12.0
    # far end: right vertical outline of the 16-32 segment, expected near L0 + 32 ft
    exp_end = L0 + 32 * ppf_a
    colmean = rr[y1 + 3:y1 + h1 - 3, :].mean(0)
    lo_i = int(exp_end - 1.2 * ppf_a)
    hi_i = int(exp_end + 1.2 * ppf_a)
    seg = colmean[lo_i:hi_i]
    ppf = (ppf_a + ppf_b) / 2.0
    if len(seg) > 3:
        i = int(np.argmin(seg))
        if seg[i] < 200:
            # subpixel centroid of the dark line
            j0, j1 = max(0, i - 3), min(len(seg), i + 4)
            wgt = np.clip(np.max(colmean[lo_i - 20:lo_i]) - seg[j0:j1], 0, None) if lo_i > 20 else np.clip(240 - seg[j0:j1], 0, None)
            if wgt.sum() > 0:
                xe = lo_i + j0 + (np.arange(j1 - j0) * wgt).sum() / wgt.sum()
                lw = (R2 - L2) - 8 * ppf_a     # stroke width
                ppf_c = (xe - (L0 + lw / 2.0)) / 32.0
                if abs(ppf_c - ppf) / ppf < 0.01:
                    ppf = (ppf_a + ppf_b + 2 * ppf_c) / 4.0
    return dict(ppf=ppf, ppf_a=ppf_a, ppf_b=ppf_b, bar=(x1, y1, x2 + w2, h1),
                solids=[(x1, y1, w1, h1), (x2, y1, w2, h1)])


def paper_level(g):
    sub = g[::5, ::5].ravel()
    h = np.bincount(sub, minlength=256).astype(np.float64)
    h[:128] = 0
    h = np.convolve(h, np.ones(5) / 5, mode='same')
    return float(np.argmax(h))


def ink_level(r, sc):
    vals = []
    for (x, y, w, h) in sc.get('solids', []):
        patch = r[y + 5:y + h - 5, x + 6:x + w - 6]
        if patch.size:
            vals.append(np.median(patch))
    return float(np.median(vals)) if vals else None


def normalize(r, sc, paper_ref=250.0, ink_ref=24.0):
    """Linear intensity normalisation: paper -> paper_ref, solid ink -> ink_ref."""
    bgv = paper_level(r)
    ink = ink_level(r, sc) if sc else None
    if ink is None or bgv - ink < 60:
        ink = float(np.percentile(r[::3, ::3], 0.05))
    k = (paper_ref - ink_ref) / max(bgv - ink, 1.0)
    G = paper_ref - (bgv - r.astype(np.float32)) * k
    return np.clip(G, 0, 255).astype(np.float32), dict(paper=bgv, ink=ink, gain=k)


def _check_bar_scale(r, solids):
    """Fallback: the title block's 'if this bar does not measure 1"' bar (= 8 ft at 1/8")."""
    H, W = r.shape
    cands = [(x, y, w, h) for (x, y, w, h) in solids if 8 <= w / float(h) <= 16 and x > 0.6 * W]
    if len(cands) != 1:
        return None
    x, y, w, h = cands[0]
    rr = r.astype(np.float64)
    band = rr[y + 3:y + h - 3, max(0, x - 20):x + w + 20].mean(0)
    lo = np.percentile(band, 2); hi = np.percentile(band, 90)
    xs = _crossings(band, (lo + hi) / 2, max(0, x - 20))
    if len(xs) < 2:
        return None
    # filled bar edges include the outline stroke (~1.8 px at 200 dpi, as on the scale bar)
    span = xs[-1] - xs[0]
    span -= 1.8 * span / 196.0
    return dict(ppf=span / 8.0, ppf_a=None, ppf_b=None, bar=(x, y, x + w, h),
                solids=[(x, y, w, h)], fallback='check_bar')
