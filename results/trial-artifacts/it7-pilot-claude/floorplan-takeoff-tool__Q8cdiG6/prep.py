"""Image loading, deskew and normalisation."""
import numpy as np
import cv2
import fitz


def load_page(pdf_path):
    """Return the scanned sheet as a uint8 grayscale image at (about) native resolution."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    rect = page.rect
    dpi = 200.0
    best = None
    try:
        for info in page.get_images(full=True):
            w, h = info[2], info[3]
            if best is None or w * h > best[0] * best[1]:
                best = (w, h)
    except Exception:
        best = None
    if best is not None and rect.width > 0:
        dpi = 72.0 * max(best[0] / rect.width, best[1] / rect.height)
    dpi = float(min(max(dpi, 150.0), 300.0))
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    img = np.ascontiguousarray(img)
    if img.shape[0] > img.shape[1]:
        # sheet is landscape; rotate portrait renders
        img = np.ascontiguousarray(np.rot90(img))
    return img, dpi


def _profile_score(img, angle):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    p = r.sum(axis=1, dtype=np.float64)
    q = r.sum(axis=0, dtype=np.float64)
    return float((np.diff(p) ** 2).sum() + (np.diff(q) ** 2).sum())


def estimate_skew(gray):
    inv = (255 - gray).astype(np.float32)
    h, w = inv.shape
    s4 = cv2.resize(inv, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    angles = np.arange(-3.0, 3.0001, 0.1)
    scores = [_profile_score(s4, a) for a in angles]
    a0 = float(angles[int(np.argmax(scores))])
    s2 = cv2.resize(inv, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    angles = np.arange(a0 - 0.12, a0 + 0.12001, 0.01)
    scores = [_profile_score(s2, a) for a in angles]
    i = int(np.argmax(scores))
    a1 = float(angles[i])
    # parabolic refinement
    if 0 < i < len(angles) - 1:
        y0, y1, y2 = scores[i - 1], scores[i], scores[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            a1 += 0.01 * 0.5 * (y0 - y2) / den
    return a1


def deskew(gray, angle):
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def normalise(gray):
    """Return darkness image D in [0,1] (float32): 0 = paper, 1 = full ink."""
    g = gray.astype(np.float32)
    h, w = g.shape
    bs = 96
    nh, nw = (h + bs - 1) // bs, (w + bs - 1) // bs
    pad = np.pad(g, ((0, nh * bs - h), (0, nw * bs - w)), mode='edge')
    blocks = pad.reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3).reshape(nh, nw, -1)
    bg = np.percentile(blocks, 90, axis=2).astype(np.float32)
    # robust smoothing of background map (blocks full of ink are outliers)
    bg = cv2.dilate(bg, np.ones((3, 3), np.uint8))
    bg = cv2.medianBlur(bg, 5)
    bg = cv2.GaussianBlur(bg, (0, 0), 1.5)
    bgf = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    ink = float(np.percentile(g, 0.3))
    paper = float(np.median(bg))
    rng = max(paper - ink, 40.0)
    D = (bgf - g) / rng
    np.clip(D, 0.0, 1.0, out=D)
    return D, paper, ink
