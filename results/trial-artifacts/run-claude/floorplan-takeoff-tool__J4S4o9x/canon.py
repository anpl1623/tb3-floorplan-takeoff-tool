"""Load a scanned A101 sheet and bring it into a canonical frame:
deskewed, resampled to PPF pixels per foot (from the graphic scale bar),
with normalised paper/ink levels."""
import numpy as np
import cv2
import fitz

PPF = 25.0  # canonical pixels per foot (1/8" = 1'-0" at 200 dpi)


def load_page(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    imgs = page.get_images(full=True)
    if len(imgs) == 1:
        try:
            info = doc.extract_image(imgs[0][0])
            buf = np.frombuffer(info['image'], np.uint8)
            a = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if a is not None and a.shape[0] > 1000:
                # orientation of the placed image must match the page
                bbox = page.get_image_info()[0]['bbox']
                pw, ph = page.rect.width, page.rect.height
                tr = page.get_image_info()[0].get('transform')
                if tr is None or (tr[0] > 0 and tr[3] > 0 and abs(tr[1]) < 1e-6 and abs(tr[2]) < 1e-6):
                    return a
        except Exception:
            pass
    zoom = 200.0 / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def estimate_angle(a, rng=3.0):
    small = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small < 120)
    if len(xs) > 2_000_000:
        sel = np.random.default_rng(0).choice(len(xs), 2_000_000, replace=False)
        xs, ys = xs[sel], ys[sel]
    xs = xs.astype(np.float64) - small.shape[1] / 2
    ys = ys.astype(np.float64) - small.shape[0] / 2

    def score(deg):
        t = np.radians(deg)
        c, s = np.cos(t), np.sin(t)
        yr = -s * xs + c * ys
        xr = c * xs + s * ys
        hy = np.bincount((yr - yr.min()).astype(np.int64)).astype(np.float64)
        hx = np.bincount((xr - xr.min()).astype(np.int64)).astype(np.float64)
        return (hy ** 2).sum() + (hx ** 2).sum()

    cands = np.arange(-rng, rng + 1e-9, 0.1)
    d0 = cands[int(np.argmax([score(d) for d in cands]))]
    for step in (0.02, 0.005, 0.001):
        cands = np.arange(d0 - 6 * step, d0 + 6.0001 * step, step)
        d0 = cands[int(np.argmax([score(d) for d in cands]))]
    return float(d0)


def _rotate(a, ang, scale=1.0):
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    W2, H2 = int(round(w * scale)), int(round(h * scale))
    M[0, 2] += (W2 - w) / 2
    M[1, 2] += (H2 - h) / 2
    return cv2.warpAffine(a, M, (W2, H2), flags=cv2.INTER_LINEAR, borderValue=255), M


def find_scalebar(g):
    """Graphic scale: black 0-4, white 4-8, black 8-16, white 16-32 (feet).
    Returns (pixels per foot, info) in the frame of g."""
    solid = cv2.morphologyEx((g < 110).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(solid, 8)
    blocks = [tuple(int(v) for v in st[i]) for i in range(1, n)
              if 10 <= st[i][3] <= 80 and st[i][2] >= 1.5 * st[i][3] and st[i][4] > 0.85 * st[i][2] * st[i][3]]
    best = None
    for b1 in blocks:
        for b2 in blocks:
            if b1 is b2:
                continue
            x1, y1, w1, h1, _ = b1
            x2, y2, w2, h2, _ = b2
            if abs(y1 - y2) > 4 or abs(h1 - h2) > 4:
                continue
            if abs(w2 - 2 * w1) > 0.12 * w1 or abs((x2 - x1) - 2 * w1) > 0.12 * w1:
                continue
            ym = y1 + h1 // 2
            prof = g[ym - 4:ym + 5, :].astype(np.float64).mean(axis=0)
            lo = np.median(prof[x1 + 3:x1 + w1 - 3])
            seg = prof[x2 + w2 + 10: x2 + w2 + 2 * w1]
            if len(seg) < 5:
                continue
            hi = np.median(seg)
            thr = (lo + hi) / 2
            i = x1 + 3
            while i > 0 and prof[i] < thr:
                i -= 1
            a = i + (thr - prof[i]) / (prof[i + 1] - prof[i])
            j = x2 + w2 + 5
            lim = min(len(prof) - 10, x2 + 6 * w2)
            while j < lim and prof[j] > (hi - 0.25 * (hi - lo)):
                j += 1
            if j >= lim:
                continue
            seg = np.arange(j - 6, j + 8)
            d = np.clip(hi - prof[seg], 0, None)
            d[d < 0.15 * (hi - lo)] = 0
            if d.sum() <= 0:
                continue
            e = (seg * d).sum() / d.sum()
            lw = d.sum() / (hi - lo)
            ppf = (e - a - lw / 2) / 32.0
            # thin-line contrast: outline of the white 16-32 segment
            xs = np.arange(int(x1 + 5 * w1), int(x1 + 7 * w1))
            col = g[max(y1 - 8, 0):y1 + h1 + 8, xs].astype(np.float64).mean(axis=1)
            half = len(col) // 2
            thin = (col[:half].min() + col[half:].min()) / 2
            # consistency: block 1 should be ~4 ft
            if not (0.85 < (w1 / ppf) / 4.0 < 1.15):
                continue
            score = h1 * w1
            if best is None or score > best[0]:
                best = (score, ppf, dict(a=float(a), e=float(e), lw=float(lw), y=int(y1), h=int(h1), black=float(lo), white=float(hi), thin=float(thin)))
    if best is None:
        return None, None
    return best[1], best[2]


def canonical(pdf_path):
    a = load_page(pdf_path)
    ang = estimate_angle(a)
    g, _ = _rotate(a, ang)
    ppf, info = find_scalebar(g)
    if ppf is None:
        raise RuntimeError('graphic scale not found')
    s = PPF / ppf
    c, M = _rotate(a, ang, s)
    # normalise levels: paper -> 248, ink (scale bar black) -> 25
    hist = np.bincount(c[::4, ::4].ravel(), minlength=256)
    white = float(np.argmax(hist[128:]) + 128)
    black = info['black']
    if white - black > 60:
        lut = np.clip(25 + (np.arange(256) - black) * (248.0 - 25.0) / (white - black), 0, 255).astype(np.uint8)
        c = cv2.LUT(c, lut)
    # contrast factor of thin lines relative to the reference scans (thin
    # outline ~119 on paper ~248 after normalisation)
    thin_n = 25 + (info['thin'] - black) * (248.0 - 25.0) / max(white - black, 1)
    k = (248.0 - thin_n) / (248.0 - 119.0)
    k = float(np.clip(k, 0.6, 1.0))
    return c, dict(angle=ang, ppf=ppf, scale=s, white=white, black=black, k=k)
