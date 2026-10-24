"""Floor finish areas: rooms from the wall model, finish classification against the legend."""
import numpy as np
import cv2


class Canvas:
    """A cropped pixel window around the building."""

    def __init__(self, fr, margin_ft=2.0, shape=None):
        x0 = int(np.floor(fr.ix(-margin_ft)))
        y0 = int(np.floor(fr.iy(-margin_ft)))
        x1 = int(np.ceil(fr.ix(fr.W + margin_ft)))
        y1 = int(np.ceil(fr.iy(fr.H + margin_ft)))
        if shape is not None:
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(shape[1], x1)
            y1 = min(shape[0], y1)
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.w = x1 - x0
        self.h = y1 - y0

    def crop(self, img):
        return img[self.y0:self.y1, self.x0:self.x1]


def wall_mask(fr, lines, cv):
    """Rasterise wall pieces (centreline extents, face-to-face thickness).  A pixel is wall
    if its centre lies inside a wall rectangle."""
    M = np.zeros((cv.h, cv.w), np.uint8)
    xs = np.arange(cv.w) + cv.x0 + 0.5 - 0.5   # pixel centre coordinates (pixel index = coordinate)
    ys = np.arange(cv.h) + cv.y0
    for L in lines.values():
        for (p0, p1, kind, tpx, votes) in L.pieces:
            # thickness: piece thickness if available, else line thickness
            if tpx is not None:
                t_ft = tpx / (fr.py if L.o == 'h' else fr.px)
            else:
                t_ft = L.t
            c0 = L.c - t_ft / 2.0
            c1 = L.c + t_ft / 2.0
            if L.o == 'h':
                ax0, ax1 = fr.ix(p0), fr.ix(p1)
                cy0, cy1 = fr.iy(c0), fr.iy(c1)
                i0 = int(np.ceil(cy0 - cv.y0))
                i1 = int(np.floor(cy1 - cv.y0))
                j0 = int(np.ceil(ax0 - cv.x0))
                j1 = int(np.floor(ax1 - cv.x0))
            else:
                ax0, ax1 = fr.iy(p0), fr.iy(p1)
                cx0, cx1 = fr.ix(c0), fr.ix(c1)
                j0 = int(np.ceil(cx0 - cv.x0))
                j1 = int(np.floor(cx1 - cv.x0))
                i0 = int(np.ceil(ax0 - cv.y0))
                i1 = int(np.floor(ax1 - cv.y0))
            i0 = max(i0, 0)
            j0 = max(j0, 0)
            i1 = min(i1, cv.h - 1)
            j1 = min(j1, cv.w - 1)
            if i1 >= i0 and j1 >= j0:
                M[i0:i1 + 1, j0:j1 + 1] = 1
    return M


def rooms(fr, lines, cv):
    W = wall_mask(fr, lines, cv)
    free = (W == 0).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(free, connectivity=4)
    # the outside component touches the canvas border
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    inside = np.zeros(n, bool)
    for i in range(1, n):
        if i not in border:
            inside[i] = True
    return W, lab, st, inside


# ----------------------------------------------------------------------------
# texture windows (sizes adapt to the drawing scale)
# ----------------------------------------------------------------------------
WS = 48       # window size (px), set by configure()
LAG = 16      # max autocorrelation lag (px)
_P = 64
LAGS = []
_LY = None
_LX = None
_CNTS = {}


def configure(ppf):
    """Window ~2 ft square, lags up to ~1.1 ft (captures 1 ft tile periods)."""
    global WS, LAG, _P, LAGS, _LY, _LX, _CNTS
    WS = int(round(2.0 * ppf))
    WS += WS % 2
    LAG = int(round(1.1 * ppf))
    LAG += LAG % 2
    _P = cv2.getOptimalDFTSize(WS + LAG + 1)
    LAGS = []
    for dy in range(0, LAG + 1, 2):
        for dx in range(-LAG, LAG + 1, 2):
            if dy == 0 and dx < 0:
                continue
            LAGS.append((dy, dx))
    _LY = np.array([dy % _P for dy, dx in LAGS])
    _LX = np.array([dx % _P for dy, dx in LAGS])
    _CNTS = {}


configure(24.5)


def _counts(h, w):
    key = (h, w)
    if key not in _CNTS:
        m = np.zeros((_P, _P), np.float32)
        m[:h, :w] = 1
        F = np.fft.rfft2(m)
        c = np.fft.irfft2(F * np.conj(F), s=(_P, _P))
        _CNTS[key] = np.maximum(c[_LY, _LX], 1.0).astype(np.float32)
    return _CNTS[key]


def window_feats(wins):
    """ACF features (sqrt-compressed) of (N, h, w) windows (h, w <= WS)."""
    N, h, w = wins.shape
    out = np.zeros((N, len(LAGS)), np.float32)
    cnt = _counts(h, w)
    bs = 512
    for i0 in range(0, N, bs):
        ww = wins[i0:i0 + bs]
        pad = np.zeros((ww.shape[0], _P, _P), np.float32)
        pad[:, :h, :w] = ww
        F = np.fft.rfft2(pad)
        c = np.fft.irfft2(F * np.conj(F), s=(_P, _P))
        out[i0:i0 + bs] = c[:, _LY, _LX] / cnt
    return np.sqrt(np.maximum(out, 0))


def swatch_refs(tink, boxes, inset=6, stride=6):
    """Reference ACF features from crops of every legend swatch.  Besides full windows,
    partial (2/3 size) windows are used so that sparse patterns whose lines fall in every
    full window are still represented by line-free / single-line samples."""
    feats, labels = [], []
    pw = int(round(WS * 0.67))
    shapes = ((WS, WS), (pw, WS), (WS, pw))
    for k, (x, y, w, h) in enumerate(boxes):
        sw = tink[y + inset:y + h - inset, x + inset:x + w - inset].astype(np.float32)
        for (hh, ww) in shapes:
            if hh > sw.shape[0] or ww > sw.shape[1]:
                hh = min(hh, sw.shape[0])
                ww = min(ww, sw.shape[1])
            wins = []
            for yy in range(0, sw.shape[0] - hh + 1, stride):
                for xx in range(0, sw.shape[1] - ww + 1, stride):
                    wins.append(sw[yy:yy + hh, xx:xx + ww])
            if not wins:
                continue
            feats.append(window_feats(np.array(wins, np.float32)))
            labels += [k] * len(wins)
    return np.vstack(feats), np.array(labels)


def knn(F, R, RL, k=5, ncls=8):
    out_pred = np.zeros(F.shape[0], np.int32)
    out_d = np.zeros(F.shape[0], np.float32)
    out_dk = np.zeros((F.shape[0], ncls), np.float32)
    BB = (R * R).sum(1)
    for i0 in range(0, F.shape[0], 2048):
        A = F[i0:i0 + 2048]
        d = (A * A).sum(1)[:, None] + BB[None, :] - 2 * A @ R.T
        d = np.sqrt(np.maximum(d, 0))
        nn = np.argsort(d, axis=1)[:, :k]
        lab = RL[nn]
        votes = np.zeros((A.shape[0], ncls), np.int32)
        for j in range(k):
            np.add.at(votes, (np.arange(A.shape[0]), lab[:, j]), 1)
        out_pred[i0:i0 + 2048] = votes.argmax(1)
        out_d[i0:i0 + 2048] = d.min(1)
        for c in range(ncls):
            sel = RL == c
            out_dk[i0:i0 + 2048, c] = d[:, sel].min(1) if sel.any() else 9.0
    return out_pred, out_d, out_dk


# ----------------------------------------------------------------------------
# line strength
# ----------------------------------------------------------------------------

def oriented_strength(x):
    """Integrated (3 px) line strength, max over 4 orientations."""
    outs = [cv2.blur(x, (7, 3)) * 3, cv2.blur(x, (3, 7)) * 3]
    kd = np.zeros((7, 7), np.float32)
    for i in range(7):
        for d in (-1, 0, 1):
            j = i + d
            if 0 <= j < 7:
                kd[i, j] = 1
    kd /= 7.0
    outs.append(cv2.filter2D(x, -1, kd))
    outs.append(cv2.filter2D(x, -1, np.fliplr(kd).copy()))
    return np.max(outs, axis=0)


def tophat(x, k=7):
    return x - cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


NAMES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']


def _hysteresis(S, lo, hi):
    strong = (S > hi).astype(np.uint8)
    weak = (S > lo).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong > 0])] = True
    keep[0] = False
    return keep[lab]


def oriented_strength5(x):
    """Integrated (5 px) line strength, max over 4 orientations (robust to blur)."""
    outs = [cv2.blur(x, (7, 5)) * 5, cv2.blur(x, (5, 7)) * 5]
    kd = np.zeros((7, 7), np.float32)
    for i in range(7):
        for d in (-2, -1, 0, 1, 2):
            j = i + d
            if 0 <= j < 7:
                kd[i, j] = 1
    kd /= 7.0
    outs.append(cv2.filter2D(x, -1, kd))
    outs.append(cv2.filter2D(x, -1, np.fliplr(kd).copy()))
    return np.max(outs, axis=0)


def calibrate_lines(ink, legend):
    """Line-strength thresholds from the legend.
    Returns (barrier_S5, heavy_S3): finish-transition lines must exceed barrier_S5; window
    sampling avoids anything stronger than heavy_S3.  Both stay clear of the strongest hatch
    line of the office's swatches and scale with the wall outline weight."""
    wall5 = []
    wall3 = []
    for (x, y, w, h) in legend['parts']:
        reg = ink[y - 8:y + h + 8, x + 10:x + w - 10].astype(np.float32)
        S5 = oriented_strength5(reg)
        S3 = oriented_strength(reg)
        wall5.append(float(np.median(S5, axis=1).max()))
        wall3.append(float(np.median(S3, axis=1).max()))
    h3 = []
    h5 = []
    for (x, y, w, h) in legend['floor']:
        reg = ink[y - 4:y + h + 4, x - 4:x + w + 4].astype(np.float32)
        th = tophat(reg)
        h3.append(float(np.percentile(oriented_strength(th)[12:-12, 12:-12], 99.5)))
        h5.append(float(np.percentile(oriented_strength5(th)[12:-12, 12:-12], 99.5)))
    w5 = float(np.median(wall5)) if wall5 else 3.0
    w3 = float(np.median(wall3)) if wall3 else 2.4
    hm5 = max(h5) if h5 else 0.8
    hm3 = max(h3) if h3 else 0.6
    barrier = max(0.5 * w5, 1.3 * hm5)
    barrier = min(barrier, 0.8 * w5)
    heavy = max(0.6 * w3, 1.25 * hm3)
    heavy = min(heavy, 0.85 * w3)
    global HATCH_MAX5
    HATCH_MAX5 = hm5
    return barrier, heavy


HATCH_MAX5 = 0.8


def _window_grid(valid, cl, H, Wd, step=16):
    cy, cx = np.mgrid[0:H - WS:step, 0:Wd - WS:step]
    cy = cy.ravel()
    cx = cx.ravel()
    ii = cv2.integral(valid.astype(np.uint8))
    tot = ii[cy + WS, cx + WS] - ii[cy, cx + WS] - ii[cy + WS, cx] + ii[cy, cx]
    ok = tot >= 0.97 * WS * WS
    cy, cx = cy[ok], cx[ok]
    cid = cl[cy + WS // 2, cx + WS // 2]
    same = np.ones(len(cy), bool)
    for dy, dx in ((4, 4), (4, WS - 5), (WS - 5, 4), (WS - 5, WS - 5)):
        c2 = cl[cy + dy, cx + dx]
        same &= (c2 == cid) | (c2 == 0)
    sel = same & (cid > 0)
    return cy[sel], cx[sel], cid[sel]


def _split_mixed_cell(cell_mask, x0, y0, wy, wx, wl, S, ncls):
    """Split a cell holding two finishes: seeded watershed on line strength."""
    from skimage.segmentation import watershed
    h, w = cell_mask.shape
    markers = np.zeros((h, w), np.int32)
    for y, x, l in zip(wy, wx, wl):
        yy = y - y0 + WS // 2
        xx = x - x0 + WS // 2
        markers[max(0, yy - 4):yy + 5, max(0, xx - 4):xx + 5] = l + 1
    markers[~cell_mask] = 0
    lab = watershed(S, markers=markers, mask=cell_mask)
    return lab - 1


def _blank_info(tink, legend, tsub, room):
    """Ink threshold for 'blank' windows, the blank-crop fraction of every swatch, and the
    finish most likely to look blank (large-module pattern such as sawcut concrete)."""
    pap = tsub[room & (cv2.blur(tsub, (9, 9)) < 0.01)]
    bthr = max(0.03, float(np.percentile(pap, 99.9)) if pap.size > 1000 else 0.03)
    dens = []
    bfr = []
    pw = int(round(WS * 0.67))
    for (x, y, w, h) in legend['floor']:
        sw = tink[y + 8:y + h - 8, x + 8:x + w - 8].astype(np.float32)
        dens.append(float(sw.mean()))
        nb = 0
        nt = 0
        for (hh, ww) in ((pw, pw), (pw, WS), (WS, pw)):
            for yy in range(0, max(1, sw.shape[0] - hh + 1), 4):
                for xx in range(0, max(1, sw.shape[1] - ww + 1), 4):
                    c = sw[yy:yy + hh, xx:xx + ww]
                    nt += 1
                    nb += (c > bthr).sum() < 10 * c.size / float(WS * WS)
        bfr.append(nb / max(nt, 1))
    blank_cls = int(max(range(len(dens)), key=lambda k: (round(bfr[k], 2), -dens[k])))
    return bthr, bfr, blank_cls


def _classify_windows(tsub, cy, cx, R, RL, ncls, bthr, gate=None):
    """kNN classes of texture windows, blank flags and vote weights (gated by distance to the
    class, scaled by the decision margin)."""
    if len(cy) == 0:
        z = np.zeros(0)
        return np.zeros(0, np.int32), z, np.zeros(0, bool), gate
    wins = np.stack([tsub[y:y + WS, x:x + WS] for y, x in zip(cy, cx)])
    pred, dmin, dk = knn(window_feats(wins), R, RL, ncls=ncls)
    blank = (wins > bthr).reshape(len(wins), -1).sum(1) < 10
    srt = np.sort(dk, axis=1)
    margin = np.clip((srt[:, 1] - srt[:, 0]) / np.maximum(srt[:, 0], 1e-6), 0.0, 1.0)
    if gate is None:
        # contaminated windows: much farther from their class than its typical clean match
        gp20 = float(np.percentile(dmin[~blank], 20)) if (~blank).any() else float(np.percentile(dmin, 20))
        gate = np.zeros(ncls)
        for k in range(ncls):
            selk = (pred == k) & ~blank
            pk = float(np.percentile(dmin[selk], 20)) if selk.sum() >= 10 else gp20
            gate[k] = 4.0 * pk + 1e-6
    wts = np.where(dmin > gate[pred], 0.0, 1.0)
    wts[blank] = 0.0
    wts = wts * (0.1 + 0.9 * margin)
    return pred, wts, blank, gate


def _decide_cells(ids, cell_area, cmask_of, cy, cx, cid, pred, wts, blank, ncls, blank_cls, bfr,
                  S5, pa, dbg=None):
    """Finish of every cell (majority of weighted votes, blank-dominated cells -> blank finish)
    and texture splits of cells that hold two finishes (transition line not closed)."""
    nmax = int(max(ids)) + 1 if len(ids) else 1
    score = np.zeros((nmax, ncls))
    np.add.at(score, (cid, pred), wts)
    cnt = np.zeros((nmax, ncls))
    np.add.at(cnt, (cid, pred), (wts > 0).astype(float))
    nblank = np.zeros(nmax)
    np.add.at(nblank, cid, blank.astype(float))
    tw = cnt.sum(1)
    cls = {}
    for i in ids:
        top_share = score[i].max() / max(score[i].sum(), 1e-9)
        if nblank[i] >= 3 * tw[i] and nblank[i] >= 10 and top_share < 0.7 and cell_area[i] * pa >= 20:
            cls[i] = blank_cls
        elif tw[i] >= 2.0 or (tw[i] >= 1.0 and cell_area[i] < 30 * WS * WS):
            cls[i] = int(np.argmax(score[i]))
        elif nblank[i] >= 3 and nblank[i] > 4 * tw[i] and cell_area[i] * pa >= 20:
            cls[i] = blank_cls
    splits = {}
    step = 16
    for i, ci in cls.items():
        maj = int(np.argmax(score[i])) if score[i].sum() > 0 else ci
        # blank windows stand for the blank finish only if the majority finish never shows
        # blank windows itself
        use_blank = bfr[maj] < 0.05 and blank_cls != maj
        seli = (cid == i) & ((wts > 0) | (blank & use_blank))
        if seli.sum() < 30:
            continue
        wy, wx = cy[seli], cx[seli]
        wl = np.where(blank[seli], blank_cls, pred[seli])
        gy = (wy - wy.min()) // step
        gx = (wx - wx.min()) // step
        G = np.full((gy.max() + 1, gx.max() + 1), -1, np.int32)
        G[gy, gx] = wl
        area_sf = cell_area[i] * pa
        thr_area = max(80.0, 0.04 * area_sf)
        cores = {}
        for k in np.unique(wl):
            mk = (G == k).astype(np.uint8)
            mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            core = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            need = thr_area
            if k == blank_cls and use_blank:
                # blank areas are weak evidence (gaps in sparse patterns): need a big region
                need = max(150.0, 0.15 * area_sf)
            if core.sum() * step * step * pa >= need:
                cores[int(k)] = core
        if len(cores) < 2:
            continue
        sy_l, sx_l, sl_l = [], [], []
        for k, core in cores.items():
            sy, sx = np.nonzero(core)
            sy_l.append(sy * step + wy.min())
            sx_l.append(sx * step + wx.min())
            sl_l.append(np.full(len(sy), k))
        x, y, w, h, cm = cmask_of(i)
        labs = _split_mixed_cell(cm, x, y, np.concatenate(sy_l), np.concatenate(sx_l),
                                 np.concatenate(sl_l), S5[y:y + h, x:x + w], ncls)
        # a real finish transition is a drawn line: the split boundary must follow one
        bnd = np.zeros_like(cm)
        bnd[:, 1:] |= (labs[:, 1:] != labs[:, :-1]) & (labs[:, 1:] >= 0) & (labs[:, :-1] >= 0)
        bnd[1:, :] |= (labs[1:, :] != labs[:-1, :]) & (labs[1:, :] >= 0) & (labs[:-1, :] >= 0)
        Sd = cv2.dilate(S5[y:y + h, x:x + w], np.ones((3, 3), np.uint8))
        bstr = float(np.median(Sd[bnd])) if bnd.any() else 0.0
        if dbg is not None:
            dbg.setdefault('split_bnd', []).append((i, bstr))
        if bstr < 0.6 * HATCH_MAX5:
            continue
        splits[i] = (x, y, w, h, labs)
    return cls, splits, score, cnt, nblank


def finish_areas(ink, tink, fr, lines, legend, tau, dbg=None, occluder=None):
    """Area (sf) of every legend finish."""
    configure(0.5 * (fr.px + fr.py))
    cv = Canvas(fr, shape=ink.shape)
    W, lab, st, inside = rooms(fr, lines, cv)
    room = inside[lab]
    sub = cv.crop(ink)
    tsub = cv.crop(tink).astype(np.float32)
    th = tophat(sub)
    S3 = oriented_strength(th)
    S5 = oriented_strength5(th)
    tb, theavy = calibrate_lines(ink, legend)
    B = _hysteresis(S5, 0.72 * tb, tb) & room
    B = cv2.morphologyEx(B.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    free = (room & ~B & (W == 0)).astype(np.uint8)
    n, cl, cst, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    heavy3 = (S3 > theavy).astype(np.uint8)
    heavy = cv2.dilate(heavy3 | B.astype(np.uint8), np.ones((7, 7), np.uint8))
    valid = room & (heavy == 0)
    if occluder is not None:
        valid &= ~occluder
    valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    H, Wd = sub.shape
    pa = 1.0 / (fr.px * fr.py)
    R, RL = swatch_refs(tink, legend['floor'])
    ncls = len(legend['floor'])
    bthr, bfr, blank_cls = _blank_info(tink, legend, tsub, room)

    # ---- pass 1: cells bounded by walls and drawn transition lines
    cy, cx, cid = _window_grid(valid, cl, H, Wd)
    pred, wts, blank, gate = _classify_windows(tsub, cy, cx, R, RL, ncls, bthr)
    if len(cy) and max(bfr) < 0.1:
        # no swatch looks blank: in cells dominated by blank windows, the textured votes
        # tell which finish is the blank one
        cnt0 = np.zeros((n, ncls))
        np.add.at(cnt0, (cid, pred), (wts > 0).astype(float))
        nb0 = np.zeros(n)
        np.add.at(nb0, cid, blank.astype(float))
        acc = np.zeros(ncls)
        for i in range(1, n):
            tv = cnt0[i].sum()
            if tv >= 20 and nb0[i] >= tv:
                acc += cnt0[i]
        if acc.sum() >= 50:
            blank_cls = int(np.argmax(acc))

    def cmask1(i):
        x, y, w, h, a = cst[i]
        return x, y, w, h, cl[y:y + h, x:x + w] == i
    ids1 = list(range(1, n))
    cls1, splits1, score1, cnt1, nblank1 = _decide_cells(
        ids1, cst[:, 4], cmask1, cy, cx, cid, pred, wts, blank, ncls, blank_cls, bfr, S5, pa, dbg)
    cls_map = np.full(lab.shape, -1, np.int32)
    cell_cls = np.full(n, -1)
    for i, k in cls1.items():
        cell_cls[i] = k
    good = (cl > 0) & (cell_cls[cl] >= 0)
    cls_map[good] = cell_cls[cl[good]]
    for i, (x, y, w, h, labs) in splits1.items():
        reg = cls_map[y:y + h, x:x + w]
        m = (cl[y:y + h, x:x + w] == i) & (labs >= 0)
        reg[m] = labs[m]

    # ---- pass 2: rooms poorly covered by classified cells (fragmented by lines) are
    #      classified as one region with the same rules
    nlab = lab.max() + 1
    area_r = np.bincount(lab[room].ravel(), minlength=nlab)
    # coverage by reliably classified cells (enough votes, not tiny fragments)
    reliable = np.zeros(n, bool)
    for i in cls1:
        if (cnt1[i].sum() >= 5 or nblank1[i] >= 10) and cst[i, 4] * pa >= 10:
            reliable[i] = True
    rel_px = (cl > 0) & reliable[cl]
    cov_r = np.bincount(lab[rel_px & room].ravel(), minlength=nlab)
    weak = [int(r) for r in np.unique(lab[room]) if area_r[r] > 0 and cov_r[r] < 0.5 * area_r[r]]
    if weak and len(R):
        rvalid = room & (cv2.dilate(heavy3, np.ones((5, 5), np.uint8)) == 0)
        if occluder is not None:
            rvalid &= ~occluder
        rvalid = cv2.erode(rvalid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        wk = np.zeros(nlab, bool)
        wk[weak] = True
        cl2 = np.where(rvalid & wk[lab], lab, 0).astype(np.int32)
        cy2, cx2, cid2 = _window_grid(rvalid, cl2, H, Wd, step=12)
        p2, w2, b2, _ = _classify_windows(tsub, cy2, cx2, R, RL, ncls, bthr, None)
        area2 = area_r.astype(float)

        def cmask2(r):
            ys_, xs_ = np.nonzero((lab == r) & room)
            y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
            return x0, y0, x1 - x0, y1 - y0, ((lab[y0:y1, x0:x1] == r) & room[y0:y1, x0:x1])
        cls2, splits2, _, _, _ = _decide_cells(
            weak, area2, cmask2, cy2, cx2, cid2, p2, w2, b2, ncls, blank_cls, bfr, S5, pa, None)
        for r, k in cls2.items():
            cls_map[(lab == r) & room] = k
        for r, (x, y, w, h, labs) in splits2.items():
            reg = cls_map[y:y + h, x:x + w]
            m = ((lab[y:y + h, x:x + w] == r) & room[y:y + h, x:x + w]) & (labs >= 0)
            reg[m] = labs[m]

    # ---- remaining pixels (lines, symbols, unclassified cells): nearest classified pixel,
    #      preferably in the same room
    labelled = (cls_map >= 0) & room
    if labelled.any():
        dist, nl = cv2.distanceTransformWithLabels((~labelled).astype(np.uint8), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
        yy, xx = np.nonzero(labelled)
        lut_c = np.full(nl.max() + 1, -1, np.int32)
        lut_r = np.zeros(nl.max() + 1, np.int32)
        lut_c[nl[yy, xx]] = cls_map[yy, xx]
        lut_r[nl[yy, xx]] = lab[yy, xx]
        near_cls = lut_c[nl]
        near_room = lut_r[nl]
        todo = room & (cls_map < 0)
        same_room = near_room == lab
        cls_map[todo & same_room] = near_cls[todo & same_room]
        room_has = np.zeros(nlab, bool)
        room_has[np.unique(lab[labelled])] = True
        rest = room & (cls_map < 0) & ~room_has[lab]
        cls_map[rest] = near_cls[rest]
        left = room & (cls_map < 0)
        if left.any():
            lab_in = (cls_map >= 0) & room
            d2, nl2 = cv2.distanceTransformWithLabels((~lab_in).astype(np.uint8), cv2.DIST_L2, 5,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
            yy, xx = np.nonzero(lab_in)
            lut2 = np.zeros(nl2.max() + 1, np.int32)
            lut2[nl2[yy, xx]] = cls_map[yy, xx]
            cls_map[left] = lut2[nl2[left]]
    areas = {}
    for k in range(ncls):
        areas[k] = float((cls_map == k).sum() * pa)
    if dbg is not None:
        dbg.update({'cv': cv, 'W': W, 'lab': lab, 'cl': cl, 'cst': cst, 'cell_cls': cell_cls,
                    'score': score1, 'cls_map': cls_map, 'B': B, 'S': S5, 'pred': pred,
                    'win': (cy, cx, cid), 'room': room, 'split': list(splits1.keys()), 'tb': tb,
                    'wts': wts, 'bfr': bfr, 'blank_cls': blank_cls, 'weak_rooms': weak})
    return areas
