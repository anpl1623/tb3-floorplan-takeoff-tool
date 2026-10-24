"""Image loading, normalisation, deskew, scale bar and building outline."""
import numpy as np
import cv2
from scipy import ndimage as ndi


def load_scan(pdf_path):
    """Return the scan as a float32 grayscale array in [0,1] (1 = white)."""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[0]
    img = None
    imgs = page.get_images(full=True)
    if len(imgs) == 1:
        try:
            info = doc.extract_image(imgs[0][0])
            buf = np.frombuffer(info['image'], np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
        except Exception:
            img = None
        if img is not None:
            # make sure the image covers (roughly) the whole page
            r = page.rect
            if img.shape[1] < 3000:
                img = None
    if img is None:
        zoom = 200.0 / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n >= 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            img = img[:, :, 0]
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # very high resolution scans: work at ~200 dpi (36" sheet -> 7200 px)
    h, w = gray.shape
    if max(h, w) > 8000:
        f = 7200.0 / max(h, w)
        gray = cv2.resize(gray, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
    # halftoned / dithered greys (some photocopies): smooth them back to grey
    if halftone_fraction(gray) > 0.22:
        gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return gray


def halftone_fraction(gray, n=2048):
    """Share of spectral power at periods of 2.2-5.5 px in a central crop."""
    h, w = gray.shape
    n = min(n, h, w)
    n -= n % 2
    c = gray[h // 2 - n // 2:h // 2 + n // 2, w // 2 - n // 2:w // 2 + n // 2].astype(np.float32) / 255.0
    c = c - c.mean()
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    P = np.abs(np.fft.rfft2(c * win)) ** 2
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    band = (r > 1 / 5.5) & (r < 1 / 2.2)
    tot = P[r > 0.01].sum()
    return float(P[band].sum() / tot) if tot > 0 else 0.0


def normalise(gray):
    """Flatten the paper background; returns float32 image, paper = 1.0.

    The paper level varies only slowly, so it is estimated from coarse block
    percentiles (so large poche/hatch fills are not mistaken for paper).
    """
    h, w = gray.shape
    bs = 200
    nby, nbx = max(h // bs, 1), max(w // bs, 1)
    P = np.zeros((nby, nbx), np.float32)
    for i in range(nby):
        for j in range(nbx):
            blk = gray[i * bs:(i + 1) * bs if i < nby - 1 else h, j * bs:(j + 1) * bs if j < nbx - 1 else w]
            P[i, j] = np.percentile(blk, 97)
    P = ndi.maximum_filter(P, size=5, mode='nearest')
    P = ndi.gaussian_filter(P, 1.5, mode='nearest')
    bg = cv2.resize(P, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = gray.astype(np.float32) / np.maximum(bg, 1.0)
    # most of the sheet is blank paper: map its median level to 1.0
    med = float(np.median(norm[::7, ::7]))
    if 0.5 < med < 1.0:
        norm = norm / med
    norm = np.clip(norm, 0.0, 1.0)
    # contrast: the darkest ink on the sheet (scale bar blocks, columns, bold
    # text, wall lines) should be black even on faded prints
    ink = 1.0 - norm
    q = float(np.percentile(ink[::3, ::3], 99.9))
    if 0.15 < q < 0.92:
        norm = 1.0 - np.clip(ink / q, 0.0, 1.0)
    return norm.astype(np.float32)


def estimate_skew(norm):
    """Coarse skew angle (degrees) of the dominant horizontal/vertical lines."""
    h, w = norm.shape
    small = cv2.resize(norm, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    bw = (small < 0.75).astype(np.uint8) * 255
    angs, wts = [], []
    for thr, minlen in ((400, 500), (200, 250), (100, 120)):
        lines = cv2.HoughLinesP(bw, 1, np.pi / 3600, threshold=thr,
                                minLineLength=minlen, maxLineGap=4)
        if lines is None:
            continue
        for x1, y1, x2, y2 in lines[:, 0]:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            L = np.hypot(x2 - x1, y2 - y1)
            if abs(a) < 6:
                angs.append(a); wts.append(L)
            elif abs(abs(a) - 90) < 6:
                angs.append(a - 90 if a > 0 else a + 90); wts.append(L)
        if len(angs) >= 8:
            break
    if not angs:
        return 0.0
    angs = np.array(angs); wts = np.array(wts)
    med = np.median(angs)
    sel = np.abs(angs - med) < 0.2
    return float(np.average(angs[sel], weights=wts[sel]))


def refine_skew(norm, angle):
    """Refine skew by fitting long near-horizontal lines with subpixel accuracy."""
    h, w = norm.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(norm, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
    dark = (rot < 0.55).astype(np.uint8)
    hk = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, 301), np.uint8))
    lab, n = ndi.label(hk)
    slopes, wts = [], []
    for i, sl in enumerate(ndi.find_objects(lab)):
        ys, xs = sl
        L = xs.stop - xs.start
        if L < 1500 or ys.stop - ys.start > 8:
            continue
        y0 = max(ys.start - 4, 0); y1 = min(ys.stop + 4, h)
        sub = 1.0 - rot[y0:y1, xs.start:xs.stop]
        sub = np.where(sub > 0.25, sub, 0)
        cols = np.arange(xs.start, xs.stop)
        s = sub.sum(axis=0)
        ok = s > 0.5
        if ok.sum() < 500:
            continue
        yc = (sub * np.arange(y0, y1)[:, None]).sum(axis=0)[ok] / s[ok]
        xc = cols[ok]
        p = np.polyfit(xc, yc, 1)
        res = yc - np.polyval(p, xc)
        keep = np.abs(res) < 1.0
        if keep.sum() > 300:
            p = np.polyfit(xc[keep], yc[keep], 1)
        slopes.append(p[0]); wts.append(L)
    if len(slopes) < 2:
        return angle
    slopes = np.array(slopes); wts = np.array(wts)
    med = np.median(slopes)
    sel = np.abs(slopes - med) < 3e-4
    corr = np.degrees(np.arctan(np.average(slopes[sel], weights=wts[sel])))
    return angle + corr


def deskew(norm):
    a = estimate_skew(norm)
    for _ in range(6):
        a1 = refine_skew(norm, a)
        done = abs(a1 - a) < 0.002
        a = a1
        if done:
            break
    h, w = norm.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    rot = cv2.warpAffine(norm, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
    return rot, a


def _edge_pos(profile, i0, i1, rising):
    """Subpixel position of the steepest transition inside profile[i0:i1]."""
    i0 = max(i0, 1); i1 = min(i1, len(profile) - 1)
    seg = profile[i0:i1]
    d = np.diff(seg)
    if rising:
        k = int(np.argmax(d))
    else:
        k = int(np.argmin(d))
    lo = seg.min(); hi = seg.max()
    mid = 0.5 * (lo + hi)
    # find crossing of mid near k
    j = k
    a, b = seg[j], seg[j + 1]
    if (a - mid) * (b - mid) <= 0 and a != b:
        return i0 + j + (mid - a) / (b - a) + 0.5 - 0.5
    return i0 + k + 0.5


def find_scale_bar(rot):
    """Locate the graphic scale bar; returns dict with px-per-foot and bbox."""
    h, w = rot.shape
    dark = (rot < 0.5).astype(np.uint8)
    op = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    lab, n = ndi.label(op)
    cands = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        hh = sl[0].stop - sl[0].start; ww = sl[1].stop - sl[1].start
        if 10 <= hh <= 90 and ww >= 30 and ww > 1.8 * hh:
            area = (lab[sl] == i + 1).sum()
            if area / (hh * ww) > 0.85:
                cands.append((sl[1].start, sl[0].start, ww, hh))
    best = None
    for a in cands:
        for b in cands:
            if b is a:
                continue
            if abs(a[1] - b[1]) < 6 and abs(a[3] - b[3]) < 6 and 1.75 < b[2] / a[2] < 2.25 \
                    and 1.75 < (b[0] - a[0]) / a[2] < 2.25:
                score = a[2]
                if best is None or score > best[0]:
                    best = (score, a, b)
    if best is None:
        raise RuntimeError('scale bar not found')
    _, a, b = best
    y0 = a[1]; hh = a[3]
    # horizontal profile through the middle of the black blocks
    ys = slice(y0 + max(2, hh // 4), y0 + hh - max(2, hh // 4))
    x_lo = max(a[0] - 60, 0)
    x_hi = min(int(a[0] + a[2] * 9.5), w)
    prof = rot[ys, x_lo:x_hi].mean(axis=0)
    u = a[2]  # approx 4ft in px
    L0 = x_lo + _edge_pos(prof, a[0] - x_lo - 8, a[0] - x_lo + 8, rising=False)
    R4 = x_lo + _edge_pos(prof, a[0] + a[2] - x_lo - 8, a[0] + a[2] - x_lo + 8, rising=True)
    L8 = x_lo + _edge_pos(prof, b[0] - x_lo - 8, b[0] - x_lo + 8, rising=False)
    R16 = x_lo + _edge_pos(prof, b[0] + b[2] - x_lo - 8, b[0] + b[2] - x_lo + 8, rising=True)
    ppf_a = (L8 - L0) / 8.0
    ppf_b = (R16 - R4) / 12.0
    ppf = 0.5 * (ppf_a + ppf_b)
    # the 32 ft end: thin vertical stroke of the outline near L0 + 32*ppf
    x32 = L0 + 32 * ppf
    # use rows spanning the bar height, look for dark minimum
    band = rot[y0 + 2:y0 + hh - 2, int(x32 - 15):int(x32 + 16)].mean(axis=0)
    ink = np.clip(band.max() - band, 0, None)
    if ink.max() > 0.2:
        xs = np.arange(int(x32 - 15), int(x32 + 16))
        k = int(np.argmax(ink))
        sel = slice(max(k - 4, 0), k + 5)
        xpk = (ink[sel] * xs[sel]).sum() / ink[sel].sum()
        stroke = max((R16 - L0) - 16 * ppf, 0)
        ppf32 = (xpk - (L0 + stroke / 2)) / 32.0
        if abs(ppf32 - ppf) / ppf < 0.01:
            ppf = (ppf_a * 8 + ppf_b * 12 + ppf32 * 32) / 52.0
    return {'ppf': float(ppf), 'bbox': (int(a[0]), int(y0), int(L0 + 32 * ppf - a[0]), int(hh)),
            'ppf_a': ppf_a, 'ppf_b': ppf_b}


def line_segments(rot, horizontal, thr=0.55, minlen=60, region=None):
    """Return dark straight segments: list of (pos, start, end, thickness)."""
    dark = (rot < thr).astype(np.uint8)
    if region is not None:
        x0, y0, x1, y1 = region
        mask = np.zeros_like(dark); mask[y0:y1, x0:x1] = 1
        dark &= mask
    if horizontal:
        k = np.ones((1, minlen), np.uint8)
    else:
        k = np.ones((minlen, 1), np.uint8)
    op = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    lab, n = ndi.label(op)
    segs = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        ys, xs = sl
        if horizontal:
            th = ys.stop - ys.start
            y0 = max(ys.start - 2, 0); y1 = ys.stop + 2
            sub = 1 - rot[y0:y1, xs.start:xs.stop]
            prof = sub.mean(axis=1)
            prof = np.clip(prof - prof.min(), 0, None)
            pos = (prof * np.arange(y0, y1)).sum() / max(prof.sum(), 1e-6)
            segs.append((pos, xs.start, xs.stop, th))
        else:
            th = xs.stop - xs.start
            x0 = max(xs.start - 2, 0); x1 = xs.stop + 2
            sub = 1 - rot[ys.start:ys.stop, x0:x1]
            prof = sub.mean(axis=0)
            prof = np.clip(prof - prof.min(), 0, None)
            pos = (prof * np.arange(x0, x1)).sum() / max(prof.sum(), 1e-6)
            segs.append((pos, ys.start, ys.stop, th))
    return segs


def merge_collinear(segs, pos_tol=2.0, gap=40):
    """Merge segments (pos, s, e, th) that are collinear and close."""
    segs = sorted(segs, key=lambda s: (round(s[0]), s[1]))
    out = []
    used = [False] * len(segs)
    segs_sorted = sorted(segs, key=lambda s: s[0])
    groups = []
    for s in segs_sorted:
        if groups and abs(s[0] - groups[-1][-1][0]) <= pos_tol:
            groups[-1].append(s)
        else:
            groups.append([s])
    for g in groups:
        g.sort(key=lambda s: s[1])
        cur = list(g[0])
        wsum = (cur[2] - cur[1]); psum = cur[0] * wsum
        for s in g[1:]:
            if s[1] <= cur[2] + gap:
                L = s[2] - s[1]
                psum += s[0] * L; wsum += L
                cur[2] = max(cur[2], s[2]); cur[3] = max(cur[3], s[3])
            else:
                cur[0] = psum / max(wsum, 1)
                out.append(tuple(cur))
                cur = list(s); wsum = (cur[2] - cur[1]); psum = cur[0] * wsum
        cur[0] = psum / max(wsum, 1)
        out.append(tuple(cur))
    return out


def _ridge_segments(rot, horizontal, thr, minlen, region, k=3, L=9):
    """Line segments from a directional ridge map (works for dark-poche walls)."""
    D = 1.0 - rot
    if horizontal:
        Dx = cv2.blur(D, (L, 1))
        R = Dx - 0.5 * (np.roll(Dx, k, axis=0) + np.roll(Dx, -k, axis=0))
    else:
        Dy = cv2.blur(D, (1, L))
        R = Dy - 0.5 * (np.roll(Dy, k, axis=1) + np.roll(Dy, -k, axis=1))
    fake = np.where(R > thr, 0.0, 1.0).astype(np.float32)
    return line_segments(fake, horizontal, 0.5, minlen, region)


def find_building(rot, ppf, xmax=None):
    """Find the exterior wall rectangle (outer/inner face pixel positions)."""
    try:
        return _find_building(rot, ppf, xmax, 'dark')
    except RuntimeError:
        return _find_building(rot, ppf, xmax, 'ridge')


def _find_building(rot, ppf, xmax, mode):
    h, w = rot.shape
    if xmax is None:
        xmax = w
    region = (0, 0, xmax, h)
    minlen = int(3 * ppf)
    if mode == 'dark':
        Hs = line_segments(rot, True, 0.6, minlen, region)
        Vs = line_segments(rot, False, 0.6, minlen, region)
    else:
        Hs = _ridge_segments(rot, True, 0.2, minlen, region)
        Vs = _ridge_segments(rot, False, 0.2, minlen, region)
    H = merge_collinear(Hs, 2.5, int(25 * ppf))
    V = merge_collinear(Vs, 2.5, int(25 * ppf))
    t_lo, t_hi = 0.35 * ppf, 1.1 * ppf

    def pairs(segs):
        out = []
        segs = [s for s in segs if s[2] - s[1] > 25 * ppf]
        for a in segs:
            for b in segs:
                d = b[0] - a[0]
                if t_lo <= d <= t_hi:
                    ov = min(a[2], b[2]) - max(a[1], b[1])
                    if ov > 0.6 * min(a[2] - a[1], b[2] - b[1]):
                        out.append((a, b))
        return out
    HP = pairs(H)
    VP = pairs(V)
    best = None
    for top in HP:
        for bot in HP:
            if bot[0][0] - top[1][0] < 40 * ppf:
                continue
            for left in VP:
                for right in VP:
                    if right[0][0] - left[1][0] < 40 * ppf:
                        continue
                    x0, x1 = left[0][0], right[1][0]
                    y0, y1 = top[0][0], bot[1][0]
                    W = x1 - x0; Hh = y1 - y0
                    # each side must span the rectangle
                    tol = 0.08
                    ok = True
                    for (a, b), lo, hi in ((top, x0, x1), (bot, x0, x1), (left, y0, y1), (right, y0, y1)):
                        span = min(a[2], b[2], hi) - max(a[1], b[1], lo)
                        ext = max(a[2], b[2]) - min(a[1], b[1])
                        if span < (1 - tol) * (hi - lo) * 0.5:
                            ok = False; break
                        if ext > (hi - lo) * 1.15:
                            ok = False; break
                    if not ok:
                        continue
                    # corners: line ends near the rectangle corners
                    area = W * Hh
                    if best is None or area > best[0]:
                        best = (area, top, bot, left, right)
    if best is None:
        raise RuntimeError('building not found')
    _, top, bot, left, right = best
    res = {
        'outer': [left[0][0], top[0][0], right[1][0], bot[1][0]],
        'inner': [left[1][0], top[1][0], right[0][0], bot[0][0]],
    }
    res = refine_building(rot, res)
    return res


def _line_center(prof, c, half=4):
    """Subpixel centre of a dark line near index c in a 1D profile (1 = white):
    position of the intensity minimum, refined by a parabola through its
    neighbours (unbiased by a lighter poche on one side of the line)."""
    c = int(round(c))
    lo = max(c - half, 1); hi = min(c + half + 1, len(prof) - 1)
    if hi <= lo:
        return float(c)
    k = lo + int(np.argmin(prof[lo:hi]))
    a, b, d = prof[k - 1], prof[k], prof[k + 1]
    den = a - 2 * b + d
    off = 0.5 * (a - d) / den if den > 1e-6 else 0.0
    return float(k + np.clip(off, -0.5, 0.5))


def refine_building(rot, res):
    """Refine the face positions using median of many profile measurements."""
    ox0, oy0, ox1, oy1 = res['outer']
    ix0, iy0, ix1, iy1 = res['inner']
    out = {}
    def meas_h(y, xs):
        vals = []
        for x in xs:
            col = rot[int(y) - 10:int(y) + 11, x - 1:x + 2].mean(axis=1)
            vals.append(int(y) - 10 + _line_center(col, 10, 3))
        vals = np.array(vals)
        med = np.median(vals)
        vals = vals[np.abs(vals - med) < 1.0]
        return float(np.median(vals)) if len(vals) else float(med)
    def meas_v(x, ys):
        vals = []
        for y in ys:
            row = rot[y - 1:y + 2, int(x) - 10:int(x) + 11].mean(axis=0)
            vals.append(int(x) - 10 + _line_center(row, 10, 3))
        vals = np.array(vals)
        med = np.median(vals)
        vals = vals[np.abs(vals - med) < 1.0]
        return float(np.median(vals)) if len(vals) else float(med)
    xs = np.linspace(ox0 + 40, ox1 - 40, 120).astype(int)
    ys = np.linspace(oy0 + 40, oy1 - 40, 120).astype(int)
    oy0 = meas_h(oy0, xs); iy0 = meas_h(iy0, xs)
    oy1 = meas_h(oy1, xs); iy1 = meas_h(iy1, xs)
    ox0 = meas_v(ox0, ys); ix0 = meas_v(ix0, ys)
    ox1 = meas_v(ox1, ys); ix1 = meas_v(ix1, ys)
    return {'outer': [ox0, oy0, ox1, oy1], 'inner': [ix0, iy0, ix1, iy1]}
