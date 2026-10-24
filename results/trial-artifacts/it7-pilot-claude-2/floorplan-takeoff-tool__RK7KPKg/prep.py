"""Image loading, deskew and ink normalisation."""
import numpy as np
import cv2


def load_image(pdf_path):
    """Return the scanned sheet as a uint8 grayscale image (one pixel per scan pixel)."""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[0]
    img = None
    imgs = page.get_images(full=True)
    if len(imgs) >= 1:
        # use the largest embedded image (the scan)
        best = max(imgs, key=lambda t: t[2] * t[3])
        try:
            info = doc.extract_image(best[0])
            data = np.frombuffer(info['image'], np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception:
            img = None
        if img is not None:
            # make sure the image fills the page reasonably (it is a scan)
            if min(img.shape[:2]) < 1500:
                img = None
    if img is None:
        # render the page at ~200 dpi
        zoom = 200.0 / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Portrait scans: rotate to landscape (sheets are landscape D-size)
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # very high resolution scans: work at about 200 dpi (36" sheet -> 7200 px)
    if img.shape[1] > 8000:
        f = 7200.0 / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return img


def _rough_ink(img):
    med = float(np.median(img))
    lo = float(np.percentile(img, 0.5))
    thr = med - 0.3 * (med - lo)
    return (img < thr).astype(np.float32)


def estimate_skew(img):
    ink = _rough_ink(img)
    f = 0.5
    small = cv2.resize(ink, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h, w = small.shape

    def score(a):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), a, 1.0)
        r = cv2.warpAffine(small, M, (w, h))
        return float(np.var(r.sum(1)) + np.var(r.sum(0)))

    best = max(np.arange(-3.0, 3.0001, 0.05), key=score)
    best = max(np.arange(best - 0.05, best + 0.0501, 0.01), key=score)
    best = max(np.arange(best - 0.01, best + 0.01001, 0.002), key=score)
    return float(best)


def deskew(img):
    a = estimate_skew(img)
    if abs(a) < 0.003:
        return img, a
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), a, 1.0)
    bv = int(np.median(img))
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    return out, a


def normalize(img):
    """Return ink map in [0,1]: 0 = paper, 1 = darkest ink.

    The paper level is estimated on a coarse grid with a very large neighbourhood so that
    large solid fills (poche, carpet, walk-off mats) are not mistaken for paper."""
    im = img.astype(np.float32)
    h, w = im.shape
    b = 32
    hb, wb = h // b, w // b
    blocks = im[:hb * b, :wb * b].reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb, wb, b * b)
    p90 = np.percentile(blocks, 90, axis=2).astype(np.float32)
    k = 31
    bgb = cv2.dilate(p90, np.ones((k, k), np.uint8))
    bgb = cv2.GaussianBlur(bgb, (0, 0), 6)
    bg = cv2.resize(bgb, (w, h), interpolation=cv2.INTER_LINEAR)
    # the block p90 is slightly above the mean paper level (noise): correct with the
    # median offset over paper-like pixels
    paper = im > np.percentile(im, 40)
    off = float(np.median((bg - im)[paper])) if paper.any() else 0.0
    off = max(0.0, min(off, 30.0))
    bgl = bg - off
    rough = im < (np.median(im) - 0.5 * (np.median(im) - np.percentile(im, 0.3)))
    if rough.sum() > 1000:
        dark = float(np.percentile(im[rough], 5))
    else:
        dark = float(np.percentile(im, 0.2))
    ink = (bgl - im) / np.maximum(bgl - dark, 1.0)
    np.clip(ink, 0, 1, out=ink)
    return ink


def noise_sigma(img, ink):
    """Noise level of the paper, in ink units (estimated on quiet paper blocks)."""
    im = img.astype(np.float32)
    h, w = im.shape
    b = 64
    hb, wb = h // b, w // b
    blocks = im[:hb * b, :wb * b].reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb * wb, b * b)
    sd = blocks.std(1)
    mn = blocks.mean(1)
    # paper level: typical mean of the brightest quiet blocks
    paper = float(np.percentile(mn, 75))
    ok = (sd > 0.4) & (sd < 20) & (np.abs(mn - paper) < 12)
    if ok.sum() < 20:
        # no noisy paper (binary or very clean image): minimal noise level
        return 0.003
    sdo = sd[ok]
    sigma_gray = float(np.percentile(sdo, 20))
    # quantisation (JPEG) can flatten the noise while leaving block artefacts: also look at
    # the spread of the quietest blocks' extreme values
    q = blocks[ok]
    qs = np.argsort(sd[ok])[:max(5, len(q) // 5)]
    ext = np.percentile(np.abs(q[qs] - np.median(q[qs], axis=1, keepdims=True)), 99, axis=1)
    sigma_gray = max(sigma_gray, float(np.median(ext)) / 3.0)
    rough = im < (np.median(im) - 0.5 * (np.median(im) - np.percentile(im, 0.3)))
    dark = float(np.percentile(im[rough], 5)) if rough.sum() > 1000 else float(np.percentile(im, 0.2))
    scale = max(paper - dark, 1.0)
    return max(sigma_gray / scale, 0.003)


def texture_ink(ink, sigma):
    """Ink map with the paper noise floor removed (for texture statistics)."""
    f = 3.0 * sigma
    out = ink - f
    np.clip(out, 0, 1, out=out)
    return out
