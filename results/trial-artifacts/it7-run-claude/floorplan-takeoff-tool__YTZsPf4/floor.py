"""Floor finish areas: room polygons from the wall model, hatch classification."""
import numpy as np
import cv2
from scipy import ndimage as ndi
from shapely.geometry import box as sbox, Polygon, MultiPolygon
from shapely.ops import unary_union

from walls import THICK

LAG = 40


def line_strength(rot, sigma=1.2):
    """Isotropic dark-line strength (largest Hessian eigenvalue of ink)."""
    D = (1.0 - rot).astype(np.float32)
    G = cv2.GaussianBlur(D, (0, 0), sigma)
    dxx = cv2.Sobel(G, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(G, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(G, cv2.CV_32F, 1, 1, ksize=3)
    tr = 0.5 * (dxx + dyy)
    df = np.sqrt((0.5 * (dxx - dyy)) ** 2 + dxy ** 2)
    return np.clip(-(tr - df), 0, None)


def strength_reference(S, bldg):
    ox0, oy0, ox1, oy1 = bldg['outer']
    vals = []
    for x in np.linspace(ox0 + 60, ox1 - 60, 200).astype(int):
        vals.append(S[int(oy0) - 3:int(oy0) + 4, x].max())
        vals.append(S[int(oy1) - 3:int(oy1) + 4, x].max())
    return float(np.median(vals))


# ------------------------------------------------------------------ rooms

def room_polygons(runs, W, H):
    t = 2.0 / 3.0
    interior = sbox(t, t, W - t, H - t)
    rects = []
    for r in runs:
        th = THICK[r['type']]
        lo, hi = r['lo'], r['hi']
        lo -= min(th, r.get('lo_th', th)) / 2 if r.get('lo_met') else 0
        hi += min(th, r.get('hi_th', th)) / 2 if r.get('hi_met') else 0
        if r['h']:
            rects.append(sbox(lo, r['c'] - th / 2, hi, r['c'] + th / 2))
        else:
            rects.append(sbox(r['c'] - th / 2, lo, r['c'] + th / 2, hi))
    walls = unary_union(rects) if rects else None
    floor = interior.difference(walls) if walls is not None else interior
    if isinstance(floor, Polygon):
        rooms = [floor]
    else:
        rooms = [g for g in floor.geoms if isinstance(g, Polygon)]
    rooms = [g for g in rooms if g.area > 1.0]
    return rooms


def rasterize_rooms(rooms, shape, tf):
    """Label image of rooms (1..n) in pixel space. tf maps (xft,yft)->(xpx,ypx)."""
    lab = np.zeros(shape, np.int32)
    SH = 4
    for i, poly in enumerate(rooms):
        ext = np.array([tf(x, y) for x, y in poly.exterior.coords], np.float64)
        pts = [np.round(ext * (1 << SH)).astype(np.int32)]
        cv2.fillPoly(lab, pts, i + 1, lineType=cv2.LINE_8, shift=SH)
        for hole in poly.interiors:
            hp = np.array([tf(x, y) for x, y in hole.coords], np.float64)
            cv2.fillPoly(lab, [np.round(hp * (1 << SH)).astype(np.int32)], 0, lineType=cv2.LINE_8, shift=SH)
    return lab


# ------------------------------------------------------------------ texture

def masked_acf(ink, mask, lag=LAG):
    """Normalised autocorrelation of ink over mask; returns (acf, mean, var)."""
    msk = mask.astype(np.float32)
    n = msk.sum()
    if n < 50:
        return None, 0.0, 0.0
    m = float((ink * msk).sum() / n)
    Z = (ink - m) * msk
    H, W = Z.shape
    fh = cv2.getOptimalDFTSize(H + lag + 1)
    fw = cv2.getOptimalDFTSize(W + lag + 1)
    F = np.fft.rfft2(Z, s=(fh, fw))
    Fm = np.fft.rfft2(msk, s=(fh, fw))
    ac = np.fft.irfft2(F * np.conj(F), s=(fh, fw))
    cnt = np.fft.irfft2(Fm * np.conj(Fm), s=(fh, fw))
    ys = np.r_[fh - lag:fh, 0:lag + 1]
    xs = np.r_[fw - lag:fw, 0:lag + 1]
    ac = ac[np.ix_(ys, xs)]
    cnt = cnt[np.ix_(ys, xs)]
    ac = ac / np.maximum(cnt, 1.0)
    var = float(ac[lag, lag])
    if var <= 1e-9:
        return np.zeros_like(ac), m, 0.0
    return ac / var, m, var



def line_feature(Rh, Rv, mask):
    """Strongest mean H/V line response along a single row/column of the masked
    window (long straight hatch lines -> high; stipple/dots/text -> low)."""
    m = mask.astype(np.float32)
    best = 0.0
    for R, ax in ((Rh, 1), (Rv, 0)):
        num = (np.clip(R, 0, None) * m).sum(axis=ax)
        cnt = m.sum(axis=ax)
        full = mask.shape[ax]
        ok = cnt >= 0.5 * full
        if ok.any():
            best = max(best, float((num[ok] / cnt[ok]).max()))
    return best


def region_signature(ink, mask, max_windows=10, win=200, rng=None, lines=None):
    """ACF signature of a (possibly large) masked region, averaged over windows."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    if (y1 - y0) * (x1 - x0) <= (2 * win) ** 2:
        a, m, v = masked_acf(ink[y0:y1, x0:x1], mask[y0:y1, x0:x1])
        if a is None:
            return None
        sig = {'acf': a, 'm': m, 'v': v, 'n': int(mask[y0:y1, x0:x1].sum())}
        if lines is not None:
            sig['L'] = line_feature(lines[0][y0:y1, x0:x1], lines[1][y0:y1, x0:x1], mask[y0:y1, x0:x1])
        return sig
    if rng is None:
        rng = np.random.RandomState(0)
    # tile the bbox, pick windows with most coverage
    tiles = []
    for ty in range(y0, y1, win):
        for tx in range(x0, x1, win):
            c = int(mask[ty:ty + win, tx:tx + win].sum())
            if c > 0.3 * win * win:
                tiles.append((c, ty, tx))
    if not tiles:
        a, m, v = masked_acf(ink[y0:y1, x0:x1], mask[y0:y1, x0:x1])
        if a is None:
            return None
        return {'acf': a, 'm': m, 'v': v, 'n': int(mask.sum())}
    tiles.sort(reverse=True)
    if len(tiles) > max_windows:
        idx = rng.choice(len(tiles), max_windows, replace=False)
        tiles = [tiles[i] for i in sorted(idx)]
    accs, ms, vs, ns = [], [], [], []
    for c, ty, tx in tiles:
        a, m, v = masked_acf(ink[ty:ty + win, tx:tx + win], mask[ty:ty + win, tx:tx + win])
        if a is None:
            continue
        accs.append(a * c); ms.append(m * c); vs.append(v * c); ns.append(c)
    if not ns:
        return None
    N = float(sum(ns))
    return {'acf': sum(accs) / N, 'm': sum(ms) / N, 'v': sum(vs) / N, 'n': int(N)}


def _acf_vec(sig):
    if '_vec' in sig:
        return sig['_vec'], sig['_E']
    A = sig['acf'].copy()
    c = LAG
    A[c - 2:c + 3, c - 2:c + 3] = 0
    v = A.ravel()
    E = float(np.sqrt(np.mean(v * v)))
    sig['_vec'] = v; sig['_E'] = E
    return v, E


def sig_distance(a, b):
    va, Ea = _acf_vec(a)
    vb, Eb = _acf_vec(b)
    # structure energy (flat vs structured)
    de = np.log((Ea + 0.03) / (Eb + 0.03))
    d = 1.5 * de * de
    # shape similarity when both are structured
    w = min(Ea, Eb) / (min(Ea, Eb) + 0.05)
    cos = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-9))
    d += 3.0 * w * (1.0 - cos)
    mm = max(a['m'], b['m'])
    dm = (a['m'] - b['m']) / max(0.015, 0.3 * mm)
    d += 0.3 * dm * dm
    if 'L' in a and 'L' in b:
        dl = (a['L'] - b['L']) / max(0.04, 0.35 * max(a['L'], b['L']))
        d += 0.4 * dl * dl
    return d, de, dm


def detect_grid_lines(Rh, Rv, bldg, ppf, ref):
    """Column-grid lines: thin straight lines that continue outside the building
    on both sides towards the grid bubbles (no hatch out there, so a low ridge
    threshold is safe). Returns (xs, ys) pixel positions."""
    ox0, oy0, ox1, oy1 = [int(round(v)) for v in bldg['outer']]
    out = {}
    for name, R, bands, span in (
            ('v', Rv, [(oy0 - int(4 * ppf), oy0 - int(0.6 * ppf)), (oy1 + int(0.6 * ppf), oy1 + int(4 * ppf))],
             (ox0 + 5, ox1 - 5)),
            ('h', Rh, [(ox0 - int(4 * ppf), ox0 - int(0.6 * ppf)), (ox1 + int(0.6 * ppf), ox1 + int(4 * ppf))],
             (oy0 + 5, oy1 - 5))):
        votes = None
        for a, b in bands:
            a = max(a, 0)
            if name == 'v':
                sub = R[a:b, span[0]:span[1]] > 0.1 * ref
                frac = sub.mean(axis=0)
            else:
                sub = R[span[0]:span[1], a:b] > 0.1 * ref
                frac = sub.mean(axis=1)
            frac = ndi.maximum_filter1d(frac, 3)
            votes = frac if votes is None else np.minimum(votes, frac)
        pos = []
        if votes is not None:
            cand = np.where(votes > 0.35)[0]
            groups = []
            for c_ in cand:
                if groups and c_ - groups[-1][-1] <= 2:
                    groups[-1].append(c_)
                else:
                    groups.append([c_])
            for g in groups:
                pos.append(span[0] + float(np.mean(g)))
        out[name] = pos
    return out['v'], out['h']


def window_signatures(ink, mask, win, min_cov=0.5, max_windows=40, rng=None, lines=None):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    H = y1 - y0; Wd = x1 - x0
    wy = min(win, H); wx = min(win, Wd)
    tiles = []
    ny = max(1, int(round(H / float(wy)))); nx = max(1, int(round(Wd / float(wx))))
    for iy in range(ny):
        ty = y0 + int(iy * (H - wy) / max(ny - 1, 1)) if ny > 1 else y0
        for ix in range(nx):
            tx = x0 + int(ix * (Wd - wx) / max(nx - 1, 1)) if nx > 1 else x0
            m = mask[ty:ty + wy, tx:tx + wx]
            c = m.mean()
            if c >= min_cov:
                tiles.append((ty, tx))
    if rng is None:
        rng = np.random.RandomState(1)
    if len(tiles) > max_windows:
        idx = rng.choice(len(tiles), max_windows, replace=False)
        tiles = [tiles[i] for i in sorted(idx)]
    sigs = []
    for ty, tx in tiles:
        a, m, v = masked_acf(ink[ty:ty + wy, tx:tx + wx], mask[ty:ty + wy, tx:tx + wx])
        if a is None:
            continue
        sig = {'acf': a, 'm': m, 'v': v, 'n': int(mask[ty:ty + wy, tx:tx + wx].sum()),
               'pos': (ty, tx, wy, wx)}
        if lines is not None:
            sig['L'] = line_feature(lines[0][ty:ty + wy, tx:tx + wx], lines[1][ty:ty + wy, tx:tx + wx],
                                    mask[ty:ty + wy, tx:tx + wx])
        sigs.append(sig)
    return sigs


def classify_region(ink, mask, swatches, ppf, blank_m=None, lines=None):
    """Robust classification: median distance over windows + global signature."""
    area_sf = float(mask.sum()) / (ppf * ppf)
    win_ft = min(12.0, max(5.0, np.sqrt(max(area_sf, 1.0)) / 4.0))
    win = int(win_ft * ppf)
    sigs = window_signatures(ink, mask, win, lines=lines)
    if len(sigs) < 2:
        # heavily masked regions (fixtures, tags): smaller windows / lower coverage
        for w2, cov in ((win // 2, 0.5), (win // 2, 0.3)):
            if w2 < 2 * LAG + 8:
                break
            s2 = window_signatures(ink, mask, w2, min_cov=cov, lines=lines)
            if len(s2) >= 2:
                sigs = s2
                break
    if not sigs:
        return None
    K = len(swatches)
    D = np.array([[sig_distance(s, sw)[0] for sw in swatches] for s in sigs])
    # per-window best
    med = np.median(D, axis=0)
    lowq = np.array([np.mean(np.sort(D[:, k])[:max(1, int(np.ceil(0.6 * len(sigs))))]) for k in range(K)])
    score = 0.5 * (med + lowq)
    order = np.argsort(score)
    votes = np.bincount(np.argmin(D, axis=1), minlength=K)
    return {'best': int(order[0]), 'second': int(order[1]), 'score': score,
            'margin': float(score[order[1]] - score[order[0]]),
            'ratio': float(score[order[0]] / max(score[order[1]], 1e-6)),
            'votes': votes, 'nwin': len(sigs), 'm': float(np.median([s['m'] for s in sigs]))}


def strong_line_mask(S, sref, thr=0.55, min_extent=12):
    """Binary mask of strong lines; drops tiny specks (dots, stipple)."""
    B = (S > thr * sref).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(B, connectivity=8)
    ext = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = ext >= min_extent
    keep[0] = False
    return keep[lab]


TEX_BLUR = 1.0
SALES_FLOOR_CODES = (0, 1)  # LVT-1, PC-1 (legend order)


def texture_ink(rot):
    """Ink image used for hatch texture analysis (lightly denoised)."""
    ink = (1.0 - rot).astype(np.float32)
    if TEX_BLUR > 0:
        ink = cv2.GaussianBlur(ink, (0, 0), TEX_BLUR)
    return ink


def swatch_signatures(tink, boxes, Btex, lines=None):
    sw = []
    for (x0, y0, x1, y1) in boxes:
        x0, y0, x1, y1 = int(round(x0)) + 5, int(round(y0)) + 5, int(round(x1)) - 4, int(round(y1)) - 4
        ln = None if lines is None else (lines[0][y0:y1, x0:x1], lines[1][y0:y1, x0:x1])
        sw.append(region_signature(tink[y0:y1, x0:x1], ~Btex[y0:y1, x0:x1], lines=ln))
    return sw


def finish_labels(ink, ppf, rooms, lab, swatches, S, sref, Bseg, Btex, debug=None, lines=None):
    """Assign finish class per room pixel (ink = texture_ink image)."""
    from skimage.segmentation import watershed
    out = []
    sf = ppf * ppf
    objs = ndi.find_objects(lab)
    for i, poly in enumerate(rooms):
        sl = objs[i]
        if sl is None:
            out.append({'area': poly.area, 'fractions': {}})
            continue
        pad = 2
        y0 = max(sl[0].start - pad, 0); y1 = sl[0].stop + pad
        x0 = max(sl[1].start - pad, 0); x1 = sl[1].stop + pad
        M = lab[y0:y1, x0:x1] == (i + 1)
        inkr = ink[y0:y1, x0:x1]
        # texture is sampled away from the room perimeter (wall face residue)
        er = max(int(0.25 * ppf), 2)
        Me = cv2.erode(M.astype(np.uint8), np.ones((2 * er + 1, 2 * er + 1), np.uint8)).astype(bool)
        Btr = Btex[y0:y1, x0:x1] | ~Me
        Bsr = Bseg[y0:y1, x0:x1]
        lnr = None if lines is None else (lines[0][y0:y1, x0:x1], lines[1][y0:y1, x0:x1])
        whole = classify_region(inkr, M & ~Btr, swatches, ppf, lines=lnr)
        if whole is not None and poly.area < 1000:
            # semantic prior: plank / polished concrete are sales-floor finishes
            sc = whole['score'].copy()
            for k in SALES_FLOOR_CODES:
                sc[k] += 0.3
            order = np.argsort(sc)
            whole = dict(whole, score=sc, best=int(order[0]), second=int(order[1]),
                         ratio=float(sc[order[0]] / max(sc[order[1]], 1e-6)))
        if poly.area < 400:
            if whole is None:
                out.append({'area': poly.area, 'fractions': {}})
            else:
                out.append({'area': poly.area, 'fractions': {whole['best']: 1.0}, 'whole': whole})
            continue
        sub = M & ~Bsr
        cl, nc = ndi.label(sub)
        markers = np.zeros(M.shape, np.int32)
        info = []
        cobjs = ndi.find_objects(cl)
        for k, csl in enumerate(cobjs):
            if csl is None:
                continue
            cm = cl[csl] == (k + 1)
            area_sf = cm.sum() / sf
            if area_sf < 8:
                continue
            res = classify_region(inkr[csl], cm & ~Btr[csl], swatches, ppf,
                                  lines=None if lnr is None else (lnr[0][csl], lnr[1][csl]))
            if res is None:
                continue
            info.append((area_sf, res, csl, k))
            if (res['ratio'] < 0.6 and res['score'][res['best']] < 2.5) or area_sf >= 300:
                er = ndi.binary_erosion(cm, iterations=2)
                markers[csl][er] = res['best'] + 1
        labs_present = np.unique(markers[markers > 0])
        if len(labs_present) == 0:
            if whole is None:
                out.append({'area': poly.area, 'fractions': {}})
            else:
                out.append({'area': poly.area, 'fractions': {whole['best']: 1.0}, 'whole': whole})
            continue
        if len(labs_present) == 1:
            out.append({'area': poly.area, 'fractions': {int(labs_present[0]) - 1: 1.0}, 'info': info})
            continue
        elev = cv2.GaussianBlur(S[y0:y1, x0:x1], (0, 0), 1.0)
        ws = watershed(elev, markers, mask=M)
        cnt = np.bincount(ws[M].ravel(), minlength=len(swatches) + 1)
        tot = float(cnt[1:].sum())
        fr = {k - 1: cnt[k] / tot for k in range(1, len(cnt)) if cnt[k] > 0}
        out.append({'area': poly.area, 'fractions': fr, 'info': info})
        if debug is not None:
            debug.append((y0, x0, ws, markers))
    return out
