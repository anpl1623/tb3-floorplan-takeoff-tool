"""Scan loading, normalisation, deskew and scale-bar measurement."""
import numpy as np
import cv2
import fitz


def load_scan(path):
    """Return the page raster of a scanned PDF as (gray uint8, colour BGR or None)."""
    doc = fitz.open(path)
    page = doc[0]
    imgs = page.get_images(full=True)
    best = None
    for info in imgs:
        xref = info[0]
        w, h = info[2], info[3]
        if best is None or w * h > best[1]:
            best = (xref, w * h)
    if best is not None:
        data = doc.extract_image(best[0])
        buf = np.frombuffer(data['image'], np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if im is None:
            pix = fitz.Pixmap(doc, best[0])
            if pix.n - pix.alpha >= 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
            im = arr[:, :, :3][:, :, ::-1].copy() if pix.n >= 3 else arr[:, :, 0].copy()
    else:
        # vector or unknown page: rasterise at 200 dpi
        pix = page.get_pixmap(dpi=200)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        im = arr[:, :, :3][:, :, ::-1].copy()
    if im.ndim == 3:
        if im.shape[2] == 4:
            im = im[:, :, :3]
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return g, im
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return im, None


def _poly_terms(x, y, deg):
    return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def ink_map(g):
    """Background-normalised ink density in [0,1] (0 = paper).

    The paper level is a smooth (cubic) surface fitted robustly to the
    brightest local values, so large solid fills are not mistaken for paper."""
    h, w = g.shape
    f = 16
    small = cv2.resize(g, (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA).astype(np.float32)
    mx = cv2.dilate(small, np.ones((5, 5), np.uint8))
    sh, sw = mx.shape
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    X = (xx / max(1, sw - 1)) * 2 - 1
    Y = (yy / max(1, sh - 1)) * 2 - 1
    A = _poly_terms(X.ravel(), Y.ravel(), 3)
    z = mx.ravel()
    keep = np.ones(z.shape, bool)
    coef = None
    for it in range(5):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        fit = A @ coef
        r = z - fit
        sd = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-3
        keep = r > -2.5 * sd
    # evaluate at quarter resolution then upsample
    q = 4
    gh, gw = max(2, h // q), max(2, w // q)
    yq, xq = np.mgrid[0:gh, 0:gw].astype(np.float32)
    Xq = (xq / max(1, gw - 1)) * 2 - 1
    Yq = (yq / max(1, gh - 1)) * 2 - 1
    bgq = (_poly_terms(Xq.ravel(), Yq.ravel(), 3) @ coef).reshape(gh, gw).astype(np.float32)
    bg = cv2.resize(bgq, (w, h), interpolation=cv2.INTER_LINEAR)
    k = 1.0 - g.astype(np.float32) / np.maximum(bg, 1.0)
    np.clip(k, 0, 1, out=k)
    return k


def rotate(img, ang_deg, border=0, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _profile_sharpness(ys, xs, wts, ang, n):
    t = np.tan(np.deg2rad(ang))
    # rows after rotating by ang (small-angle shear approximation)
    yr = ys - xs * t
    hist = np.bincount(np.round(yr).astype(np.int64) + n, weights=wts, minlength=3 * n)
    xr = xs + ys * t
    hist2 = np.bincount(np.round(xr).astype(np.int64) + n, weights=wts, minlength=3 * n)
    return float(np.dot(hist, hist) + np.dot(hist2, hist2))


def estimate_skew(k):
    """Skew angle (deg, cv2 convention) maximising projection sharpness."""
    h, w = k.shape
    f = 4
    small = cv2.resize(k, (w // f, h // f), interpolation=cv2.INTER_AREA)
    thr = max(0.05, 0.5 * float(np.percentile(small, 99.5)))
    ys, xs = np.nonzero(small > thr)
    ys = ys.astype(np.float64)
    xs = xs.astype(np.float64)
    wts = np.ones_like(ys)
    n = max(h, w) // f + 10
    # sign: rotate(img, a) with cv2 rotates counter-clockwise by a; a point (x,y)
    # goes (approximately) to y' = y - x*tan(a) ... search directly over a.
    angs = np.arange(-3.0, 3.0001, 0.05)
    sc = [_profile_sharpness(ys, xs, wts, a, n) for a in angs]
    a0 = float(angs[int(np.argmax(sc))])
    # full resolution refinement
    thr = max(0.05, 0.5 * float(np.percentile(k[::4, ::4], 99.5)))
    ys, xs = np.nonzero(k > thr)
    if len(ys) > 4_000_000:
        sel = np.random.default_rng(0).choice(len(ys), 4_000_000, replace=False)
        ys, xs = ys[sel], xs[sel]
    ys = ys.astype(np.float64)
    xs = xs.astype(np.float64)
    wts = np.ones_like(ys)
    n = max(h, w) + 10
    best = a0
    for step, span in ((0.01, 0.06), (0.002, 0.012), (0.0005, 0.003)):
        angs = np.arange(best - span, best + span + 1e-9, step)
        sc = [_profile_sharpness(ys, xs, wts, a, n) for a in angs]
        best = float(angs[int(np.argmax(sc))])
    return best


def find_scale_bar(kn):
    """Locate the graphic scale, retrying at lower binarisation thresholds."""
    for thr in (0.45, 0.3, 0.2):
        r = _find_scale_bar(kn, thr)
        if r is not None:
            return r
    return None


def _find_scale_bar(kn, bin_thr=0.45):
    """Locate the graphic scale (0-4-8-16-32 ft bar).  Returns dict or None.

    The bar is drawn as filled black [0,4] and [8,16] ft segments and outlined
    white [4,8] and [16,32] ft segments.  Stroke width biases cancel when the
    centres of the black segments are used; the far stroke centre of the 32 ft
    end gives the longest baseline."""
    h, w = kn.shape
    b = (kn > bin_thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, 8)
    best = None
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if ww < 150 or hh < 8 or ww / max(hh, 1) < 12 or ww / max(hh, 1) > 80:
            continue
        pad = 12
        x0 = max(0, x - pad)
        x1 = min(w, x + ww + pad)
        sub = kn[y:y + hh, x0:x1]
        mid = sub[int(hh * 0.3):int(hh * 0.7) + 1].mean(axis=0)
        mx = float(np.percentile(mid, 98))
        if mx < 0.3:
            continue
        thr = mx / 2
        bb = mid > thr
        ed = []
        for j in range(1, len(mid)):
            if bb[j] != bb[j - 1]:
                a0, c0 = mid[j - 1], mid[j]
                ed.append((x0 + j - 1 + (thr - a0) / (c0 - a0 + 1e-9), bool(bb[j])))
        ups = [e for e, u in ed if u]
        downs = [e for e, u in ed if not u]
        if len(ups) < 2 or len(downs) < 2:
            continue
        b1 = (ups[0], downs[0])
        b2 = (ups[1], downs[1])
        if not (b1[0] < b1[1] < b2[0] < b2[1]):
            continue
        L = float(ww)
        r1 = (b1[1] - b1[0]) / L
        r2 = (b2[1] - b2[0]) / L
        g1 = (b2[0] - b1[1]) / L
        if not (0.08 < r1 < 0.18 and 0.18 < r2 < 0.32 and 0.08 < g1 < 0.18):
            continue
        # the 16-32 part is white apart from the end stroke (thin, may be faint)
        tail0 = int(b2[1] - x0 + 0.3 * L)
        tail = mid[tail0:]
        if len(tail) < 0.2 * L:
            continue
        tmax = float(tail.max())
        if tmax < 0.15 * mx:
            continue
        j = int(np.argmax(tail))
        # rightmost strong peak
        cand_idx = np.nonzero(tail > 0.6 * tmax)[0]
        j = int(cand_idx.max())
        while j + 1 < len(tail) and tail[j + 1] >= tail[j]:
            j += 1
        while j - 1 >= 0 and tail[j - 1] > tail[j]:
            j -= 1
        lo_, hi_ = max(0, j - 4), min(len(tail), j + 5)
        wv = np.clip(tail[lo_:hi_] - 0.1 * tmax, 0, None)
        e_c = x0 + tail0 + lo_ + float(np.sum(np.arange(hi_ - lo_) * wv) / max(wv.sum(), 1e-6))
        inner = mid[int(b2[1] - x0 + 4):int(e_c - x0 - 4)]
        if len(inner) < 0.3 * L or float(np.mean(inner > thr)) > 0.1:
            continue
        e_up = e_dn = e_c
        c1 = (b1[0] + b1[1]) / 2.0
        c2 = (b2[0] + b2[1]) / 2.0
        x32 = (e_up + e_dn) / 2.0
        s_c = (c2 - c1) / 10.0
        s_e = (x32 - c1) / 30.0
        s = s_e if abs(s_e - s_c) < 0.01 * s_c else s_c
        cand = dict(x=int(x), y=int(y), w=int(ww), h=int(hh), ppf=float(s),
                    s_c=float(s_c), s_e=float(s_e), x0=float(c1 - 2 * s))
        if best is None or ww > best['w']:
            best = cand
    return best
