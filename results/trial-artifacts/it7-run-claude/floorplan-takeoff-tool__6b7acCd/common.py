"""Image loading, normalisation and deskew for scanned A101 sheets."""
import numpy as np

# global calibration: k = (measured wall stroke darkness) / (nominal 0.85)
CAL = {'k': 1.0}
import cv2
import fitz  # PyMuPDF


def load_page_image(path):
    """Return the scanned page as an 8-bit array (gray or BGR)."""
    doc = fitz.open(path)
    page = doc[0]
    imgs = page.get_images(full=True)
    im = None
    if imgs:
        xref = max(imgs, key=lambda t: t[2] * t[3])[0]
        try:
            info = doc.extract_image(xref)
            buf = np.frombuffer(info['image'], np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        except Exception:
            im = None
        if im is not None:
            # check placement: if the image is rotated/flipped on the page, render instead
            try:
                rects = page.get_image_rects(xref, transform=True)
                if rects:
                    _, m = rects[0]
                    if abs(m.b) > 1e-6 or abs(m.c) > 1e-6 or m.a < 0 or m.d < 0 or page.rotation:
                        im = None
            except Exception:
                pass
    if im is None:
        zoom = 200.0 / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
        im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2BGR)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return im


def to_gray(im):
    if im.ndim == 2:
        return im.astype(np.float32)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)


def background_map(g, step=16, win=25):
    """Smooth estimate of paper brightness (high percentile in local windows)."""
    h, w = g.shape
    small = cv2.resize(g, (w // step, h // step), interpolation=cv2.INTER_AREA)
    # local high-percentile via dilation (max) after mild smoothing, then smoothing
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    bg = cv2.dilate(cv2.medianBlur(small.astype(np.uint8), 5), k).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), win / 2.0)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def darkness(g):
    """Map gray image to ink darkness: 0 = paper, 1 = full ink."""
    bg = background_map(g)
    d = (bg - g) / np.maximum(bg, 1.0)
    # estimate the level of solid ink: darkest robust percentile
    sample = d[::4, ::4]
    ink = np.percentile(sample, 99.95)
    ink = max(ink, 0.3)
    d = d / ink
    return np.clip(d, 0.0, 1.2).astype(np.float32)


def estimate_skew(dk):
    """Estimate rotation (degrees) that makes long lines axis aligned."""
    h, w = dk.shape
    f = 4
    small = cv2.resize(dk, (w // f, h // f), interpolation=cv2.INTER_AREA)
    best = (None, 0.0)
    sh, sw = small.shape
    for ang in np.arange(-5.0, 5.0001, 0.1):
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), ang, 1.0)
        r = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_LINEAR)
        sc = (r.sum(1) ** 2).sum() + (r.sum(0) ** 2).sum()
        if best[0] is None or sc > best[0]:
            best = (sc, ang)
    a0 = best[1]
    best = (None, a0)
    for ang in np.arange(a0 - 0.12, a0 + 0.1201, 0.01):
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), ang, 1.0)
        r = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_LINEAR)
        sc = (r.sum(1) ** 2).sum() + (r.sum(0) ** 2).sum()
        if best[0] is None or sc > best[0]:
            best = (sc, ang)
    a1 = best[1]
    # refine by fitting long straight lines in the coarsely-deskewed image
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a1, 1.0)
    r = cv2.warpAffine(dk, M, (w, h), flags=cv2.INTER_LINEAR)
    ink = (r > 0.45).astype(np.uint8)
    angs = []
    for horiz in (True, False):
        if horiz:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (301, 1))
            dil = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        else:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 301))
            dil = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        op = cv2.morphologyEx(dil, cv2.MORPH_OPEN, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(op)
        for i in range(1, n):
            x, y, ww, hh, a = stats[i]
            L = ww if horiz else hh
            if L < 1200:
                continue
            sub = (lab[y:y + hh, x:x + ww] == i) & (ink[y:y + hh, x:x + ww] > 0)
            ys, xs = np.nonzero(sub)
            if len(xs) < 500:
                continue
            if horiz:
                p = np.polyfit(xs, ys, 1)
                angs.append((L, np.degrees(np.arctan(p[0]))))
            else:
                p = np.polyfit(ys, xs, 1)
                angs.append((L, -np.degrees(np.arctan(p[0]))))
    if angs:
        angs.sort(reverse=True)
        top = angs[:12]
        vals = np.array([a for _, a in top])
        wts = np.array([l for l, _ in top], float)
        med = np.median(vals)
        keep = np.abs(vals - med) < 0.05
        corr = float((vals[keep] * wts[keep]).sum() / wts[keep].sum())
        return a1 + corr
    return a1


def rotate(img, ang, border):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def prepare(path):
    raw = load_page_image(path)
    g = to_gray(raw)
    dk = darkness(g)
    ang = estimate_skew(dk)
    dk = rotate(dk, ang, 0.0)
    return dk, ang


def hollow_dark_fills(dk, level=0.75, core_level=0.3):
    """Lighten the interior of wide solid dark regions (e.g. solid wall poche) so that their
    outlines read as strokes; thin strokes and text are unchanged."""
    sm = cv2.GaussianBlur(dk, (0, 0), 0.7)
    dark = (sm > level).astype(np.uint8)
    core = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if not core.any():
        return dk
    out = dk.copy()
    m = core > 0
    out[m] = np.minimum(out[m], core_level)
    return out
