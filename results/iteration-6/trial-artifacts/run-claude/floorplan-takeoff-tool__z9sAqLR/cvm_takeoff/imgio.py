"""Loading the scanned sheet, deskewing it and reading the graphic scale."""
import numpy as np
import cv2
import fitz


def load_sheet(pdf_path):
    """Return the sheet as a uint8 grayscale array.

    A single scanned image covering the page is read at its native resolution; anything
    else (tiled scans, rotated placement, vector content) is rendered.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    img = None
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        infos = []
    big = [i for i in infos if i.get('xref')]
    if len(big) >= 1:
        best = max(big, key=lambda i: i.get('width', 0) * i.get('height', 0))
        bb = fitz.Rect(best['bbox'])
        tr = best.get('transform', (1, 0, 0, 1, 0, 0))
        upright = abs(tr[1]) < 1e-6 and abs(tr[2]) < 1e-6 and tr[0] > 0 and tr[3] > 0
        covers = bb.width >= 0.9 * page.rect.width and bb.height >= 0.9 * page.rect.height
        if upright and covers and page.rotation % 360 == 0:
            try:
                pix = fitz.Pixmap(doc, best['xref'])
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
                if pix.n != 1:
                    pix = fitz.Pixmap(fitz.csGRAY, pix)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width].copy()
            except Exception:
                img = None
    if img is None:
        dpi = 200.0
        if big:
            best = max(big, key=lambda i: i.get('width', 0) * i.get('height', 0))
            bb = fitz.Rect(best['bbox'])
            if bb.width > 0:
                dpi = float(np.clip(best['width'] / (bb.width / 72.0), 150.0, 300.0))
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width].copy()
    return img


def ink_levels(img):
    """Paper level and the level of solid ink (darkest significant tone) of a scan."""
    sub = img[::4, ::4]
    paper = float(np.percentile(sub, 75))
    ink = float(np.percentile(sub, 0.3))
    if paper - ink < 40:
        ink = paper - 40
    return paper, ink


def estimate_skew(img):
    """Estimate the small rotation of the scan (degrees) from long straight dark lines."""
    small = img
    f = 1.0
    if img.shape[1] > 3600:
        f = 0.5
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    paper, ink = ink_levels(img)
    dark = small < (paper - 0.45 * (paper - ink))
    # keep pixels that belong to long horizontal or vertical structures
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 41))
    d8 = dark.astype(np.uint8)
    hl = cv2.morphologyEx(d8, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(d8, cv2.MORPH_OPEN, vk)
    ys, xs = np.nonzero(hl)
    vy, vx = np.nonzero(vl)
    rng = np.random.default_rng(0)
    if len(ys) > 400000:
        sel = rng.choice(len(ys), 400000, replace=False)
        ys, xs = ys[sel], xs[sel]
    if len(vy) > 400000:
        sel = rng.choice(len(vy), 400000, replace=False)
        vy, vx = vy[sel], vx[sel]
    ys = ys.astype(np.float64); xs = xs.astype(np.float64)
    vy = vy.astype(np.float64); vx = vx.astype(np.float64)

    def score(theta_deg):
        t = np.radians(theta_deg)
        c, s = np.cos(t), np.sin(t)
        sc = 0.0
        if len(ys):
            yr = ys * c - xs * s
            h = np.bincount(np.round(yr - yr.min()).astype(np.int64))
            sc += float((h.astype(np.float64) ** 2).sum())
        if len(vy):
            xr = vx * c + vy * s
            h = np.bincount(np.round(xr - xr.min()).astype(np.int64))
            sc += float((h.astype(np.float64) ** 2).sum())
        return sc

    best = 0.0
    for step, span in ((0.1, 3.0), (0.02, 0.15), (0.004, 0.03)):
        cands = np.arange(best - span, best + span + 1e-9, step)
        scores = [score(t) for t in cands]
        best = float(cands[int(np.argmax(scores))])
    # theta rotates image coordinates so that lines become level:
    # y' = y cos t - x sin t  is constant along a line with slope tan t
    return best


def deskew(img, angle_deg):
    if abs(angle_deg) < 1e-4:
        return img.copy()
    H, W = img.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _crossings(prof, thr):
    out = []
    for i in range(len(prof) - 1):
        a, b = prof[i], prof[i + 1]
        if (a - thr) * (b - thr) < 0:
            out.append((i + (thr - a) / (b - a), -1 if b < a else 1))
    return out


def find_scale(img):
    """Locate the graphic scale bar (0-4-8-16-32 FEET) and return (px_per_ft, info)."""
    paper, ink = ink_levels(img)
    bw = (img < (paper - 0.62 * (paper - ink))).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    rects = []
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if 6 <= h <= 120 and w >= 2.0 * h and ar > 0.85 * w * h:
            rects.append((int(x), int(y), int(w), int(h)))
    best = None
    for A in rects:
        for B in rects:
            if B is A:
                continue
            xa, ya, wa, ha = A
            xb, yb, wb, hb = B
            if abs(ya - yb) > 3 or abs(ha - hb) > 3:
                continue
            if not (1.85 * wa < wb < 2.15 * wa):
                continue
            if not (1.85 * wa < xb - xa < 2.15 * wa):
                continue
            err = abs(wb - 2 * wa) + abs((xb - xa) - 2 * wa)
            if best is None or err < best[0]:
                best = (err, A, B)
    if best is None:
        raise RuntimeError('graphic scale bar not found')
    _, A, B = best
    xa, ya, wa, ha = A
    imgf = img.astype(np.float64)
    seg = wa  # ~4 ft in px
    x0 = max(0, xa - int(0.5 * seg))
    x1 = min(img.shape[1], xa + int(9.0 * seg))
    r0 = ya + max(2, ha // 4)
    r1 = ya + ha - max(2, ha // 4)
    prof = imgf[r0:r1, x0:x1].mean(axis=0)
    lo = np.percentile(prof, 2)
    hi = np.percentile(prof, 98)
    thr = 0.5 * (lo + hi)
    cr = _crossings(prof, thr)
    cr = [(p + x0, d) for p, d in cr]
    # expected positions (in units of the 4ft segment from the 0 mark)
    e0 = xa
    def nearest(target, direction):
        cands = [p for p, d in cr if d == direction]
        if not cands:
            return None
        p = min(cands, key=lambda q: abs(q - target))
        if abs(p - target) > 0.25 * seg:
            return None
        return p
    p0 = nearest(e0, -1)            # white->black at 0
    p4 = nearest(e0 + seg, 1)       # black->white at 4
    p8 = nearest(e0 + 2 * seg, -1)  # white->black at 8
    p16 = nearest(e0 + 4 * seg, 1)  # black->white at 16
    est = []
    if p0 is not None and p8 is not None:
        est.append(((p8 - p0) / 8.0, 8.0))
    if p4 is not None and p16 is not None:
        est.append(((p16 - p4) / 12.0, 12.0))
    if not est:
        ppf = (B[0] + B[2] - xa) / 16.0
        return ppf, dict(method='coarse')
    ppf_inner = sum(v * w for v, w in est) / sum(w for v, w in est)
    # refine with the far end of the bar (thin outline at 32 ft)
    ppf = ppf_inner
    if p0 is not None and p4 is not None:
        half_stroke = 0.5 * ((p4 - p0) - 4.0 * ppf_inner)
        zero = p0 + half_stroke
        target = zero + 32.0 * ppf_inner
        lo_i = int(target - 0.3 * seg - x0)
        hi_i = int(target + 0.3 * seg - x0)
        if 0 < lo_i < hi_i < len(prof):
            sub = prof[lo_i:hi_i]
            d = np.clip(np.median(prof[int(4.6 * seg):int(7.4 * seg)]) - sub, 0, None) if len(prof) > int(7.4 * seg) else np.clip(hi - sub, 0, None)
            if d.max() > 0.25 * (hi - lo):
                k = int(np.argmax(d))
                a0 = max(0, k - 3)
                a1 = min(len(d), k + 4)
                w = d[a0:a1]
                pos = (w * np.arange(a0, a1)).sum() / w.sum() + lo_i + x0
                ppf_far = (pos - zero) / 32.0
                if abs(ppf_far - ppf_inner) < 0.01 * ppf_inner:
                    ppf = (ppf_far * 32.0 + ppf_inner * 10.0) / 42.0
    return ppf, dict(method='bar', zero_x=xa, y=ya, h=ha, seg=wa)


# width of the sheet border in drawing feet at the plan scale (same CAD sheet on every store)
BORDER_WIDTH_FT = 280.9


def scale_from_border(img):
    """Fallback px/ft from the drawn sheet border (when the graphic scale cannot be read)."""
    paper, ink = ink_levels(img)
    dark = img < (paper - 0.45 * (paper - ink))
    H, W = img.shape
    lefts, rights = [], []
    for y in range(int(0.2 * H), int(0.8 * H), 25):
        row = dark[y]
        idx = np.nonzero(row[:int(0.15 * W)])[0]
        if len(idx):
            lefts.append(idx[0])
        idx = np.nonzero(row[int(0.85 * W):])[0]
        if len(idx):
            rights.append(int(0.85 * W) + idx[-1])
    if len(lefts) < 10 or len(rights) < 10:
        return None
    width = float(np.median(rights) - np.median(lefts))
    ppf = width / BORDER_WIDTH_FT
    return ppf if 15.0 < ppf < 40.0 else None
