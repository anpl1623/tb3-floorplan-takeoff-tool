"""Door detection anchored on wall segments."""
import numpy as np
import cv2


def _sample_rows(G, rows, cols):
    """G[rows, cols] with clipping (rows, cols int arrays broadcastable)."""
    H, W = G.shape
    r = np.clip(rows, 0, H - 1)
    c = np.clip(cols, 0, W - 1)
    v = G[r, c]
    oob = (rows < 0) | (rows >= H) | (cols < 0) | (cols >= W)
    return np.where(oob, 0.0, v)


def swing_scores(G, s, side, d, w):
    """Arc score for every hinge position along wall segment s.
    G: dilated darkness (ROI coords). side: -1 (towards f1 / smaller coord) or +1.
    d: sweep direction along the wall (+1/-1). w: door width in px.
    Returns (positions, arc_score, leaf_score)."""
    f = s['f1'] if side < 0 else s['f2']
    a0 = int(np.floor(s['a'] - w))
    a1 = int(np.ceil(s['b'] + w))
    pos = np.arange(a0, a1)
    thetas = np.deg2rad(np.linspace(10, 80, 15))
    arc = np.zeros(len(pos))
    ring = np.zeros(len(pos))
    for th in thetas:
        for rr, acc in ((w, 'a'), (w - 7, 'i'), (w + 7, 'o')):
            dx = d * rr * np.cos(th)
            dy = side * rr * np.sin(th)
            along = np.round(pos + dx).astype(int)
            across = int(round(f + dy))
            if s['orient'] == 'H':
                v = _sample_rows(G, np.full(len(pos), across), along)
            else:
                v = _sample_rows(G, along, np.full(len(pos), across))
            if acc == 'a':
                arc += v
            else:
                ring = np.maximum(ring, v) if False else ring + v * 0.5
    arc /= len(thetas)
    ring /= len(thetas)
    # leaf: line perpendicular to the wall at the hinge, length w
    ts = np.linspace(0.15, 0.9, 10) * w
    leaf = np.zeros(len(pos))
    for t in ts:
        across = int(round(f + side * t))
        best = np.zeros(len(pos))
        for off in range(-3 * d, 4 * d, d) if d > 0 else range(3, -4, -1):
            along = pos + off
            if s['orient'] == 'H':
                v = _sample_rows(G, np.full(len(pos), across), along)
            else:
                v = _sample_rows(G, along, np.full(len(pos), across))
            best = np.maximum(best, v)
        leaf += best
    leaf /= len(ts)
    return pos, arc - ring, leaf


def opening_is_open(D, s, o0, o1):
    """True unless the band interior over [o0,o1] looks like the wall's own (non-blank) poche."""
    a, band, cen, off = band_profiles(D, s)
    seg_fill = band.mean(axis=0)
    i0 = int(max(0, o0 - a + 3))
    i1 = int(min(band.shape[1], o1 - a - 3))
    if i1 - i0 < 5:
        return True
    med = float(np.median(seg_fill))
    op = band[:, i0:i1]
    # wall poche statistics outside the opening
    mask = np.ones(band.shape[1], bool)
    mask[max(0, i0 - 10):i1 + 10] = False
    if mask.sum() < 10:
        return True
    rest = band[:, mask]
    rmean, omean = float(rest.mean()), float(op.mean())
    rstd, ostd = float(rest.std()), float(op.std())
    if rmean < 0.1 and rstd < 0.08:
        return True  # blank-looking poche: cannot tell
    return not (abs(omean - rmean) < 0.35 * rmean and abs(ostd - rstd) < 0.35 * max(rstd, 0.02))


def find_swings(G, segs, ppf, widths_ft=(2.5, 3.0, 3.5, 4.0), arc_thr=None, leaf_thr=0.35, D=None, exclude=()):
    if arc_thr is None:
        allc = find_swings(G, segs, ppf, widths_ft, 0.12, leaf_thr, D, exclude)
        sc = sorted([c['score'] for c in allc], reverse=True)
        if not sc:
            return []
        top = sc[:min(8, len(sc))]
        thr = max(0.15, min(0.28, 0.62 * float(np.median(top))))
        return [c for c in allc if c['score'] >= thr]
    cands = []
    for si, s in enumerate(segs):
        for side in (-1, 1):
            for d in (-1, 1):
                best = None
                for wf in widths_ft:
                    w = wf * ppf
                    pos, A, L = swing_scores(G, s, side, d, w)
                    sc = np.where(L > leaf_thr, A, -1.0)
                    if best is None:
                        best = (sc, np.full(len(sc), wf), pos)
                    else:
                        # align by position (pos ranges differ by w); map onto common axis
                        p0 = best[2]
                        common = np.full(len(p0), -1.0)
                        idx = pos - p0[0]
                        ok = (idx >= 0) & (idx < len(p0))
                        common[idx[ok]] = sc[ok]
                        better = common > best[0]
                        b0 = best[0].copy()
                        b0[better] = common[better]
                        b1 = best[1].copy()
                        b1[better] = wf
                        best = (b0, b1, p0)
                sc, wfs, pos = best
                sc = sc.astype(np.float32)
                k = max(1, int(0.75 * ppf))
                mx = cv2.dilate(sc.reshape(1, -1).astype(np.float32), np.ones((1, 2 * k + 1), np.uint8)).ravel()
                for i in np.nonzero((sc >= mx) & (sc > arc_thr))[0]:
                    # hinge must be within the wall extent (opening lies in the wall band)
                    h = float(pos[i])
                    w = wfs[i] * ppf
                    o0, o1 = sorted((h, h + d * w))
                    if o0 < s['a'] - 0.3 * ppf or o1 > s['b'] + 0.3 * ppf:
                        continue
                    if any(e['seg'] == si and min(e['o1'], o1) - max(e['o0'], o0) > 0.3 * w for e in exclude):
                        continue
                    if D is not None and not opening_is_open(D, s, o0, o1):
                        continue
                    cands.append(dict(score=float(sc[i]), seg=si, side=side, d=d, hinge=h, w=w,
                                      o0=o0, o1=o1, orient=s['orient'], c=s['c']))
    # NMS: same segment, overlapping openings
    cands.sort(key=lambda c: -c['score'])
    keep = []
    for c in cands:
        dup = False
        for k in keep:
            if k['seg'] == c['seg'] and min(k['o1'], c['o1']) - max(k['o0'], c['o0']) > 0.5 * min(k['w'], c['w']) \
                    and (k['side'] == c['side'] or True) and k['d'] == c['d'] and abs(k['hinge'] - c['hinge']) < 0.5 * ppf:
                dup = True
                break
        if not dup:
            keep.append(c)
    return keep


def group_pairs(swings, ppf):
    """Combine mirrored swings sharing one opening into pair doors."""
    used = set()
    pairs = []
    singles = []
    for i, a in enumerate(swings):
        if i in used:
            continue
        for j, b in enumerate(swings):
            if j <= i or j in used:
                continue
            if a['seg'] != b['seg'] or a['side'] != b['side'] or a['d'] == b['d']:
                continue
            left, right = (a, b) if a['hinge'] < b['hinge'] else (b, a)
            if left['d'] != 1 or right['d'] != -1:
                continue
            gap = right['hinge'] - left['hinge']
            if abs(gap - (left['w'] + right['w'])) < 0.6 * ppf:
                pairs.append(dict(o0=left['hinge'], o1=right['hinge'], seg=a['seg'], orient=a['orient'], c=a['c']))
                used.add(i)
                used.add(j)
                break
    for i, a in enumerate(swings):
        if i not in used:
            singles.append(a)
    return singles, pairs


def band_profiles(D, s):
    """Along-wall profiles inside the wall band: centre-line max, off-centre mean, row-max darkness."""
    c = s['c']
    f1, f2 = s['f1'], s['f2']
    a, b = int(s['a']), int(s['b'])
    rows = np.arange(int(np.ceil(f1 + 2.5)), int(np.floor(f2 - 2.5)) + 1)
    if len(rows) < 2:
        rows = np.arange(int(round(c)) - 1, int(round(c)) + 2)
    if s['orient'] == 'H':
        band = D[rows[0]:rows[-1] + 1, a:b]
    else:
        band = D[a:b, rows[0]:rows[-1] + 1].T
    ci = int(round(c)) - rows[0]
    lo = max(0, ci - 1)
    hi = min(band.shape[0], ci + 2)
    cen = band[lo:hi].max(axis=0)
    offmask = np.ones(band.shape[0], bool)
    offmask[max(0, ci - 2):ci + 3] = False
    off = band[offmask].mean(axis=0) if offmask.any() else np.zeros(band.shape[1])
    return a, band, cen, off


def _runs(mask):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _close_runs(mask, gap):
    """Fill gaps of False shorter than `gap` between True runs."""
    m = mask.copy()
    rs = _runs(mask)
    for (a0, a1), (b0, b1) in zip(rs[:-1], rs[1:]):
        if b0 - a1 <= gap:
            m[a1:b0] = True
    return m


def find_special(D, segs, ppf, swings, pairs):
    """Overhead coiling (dashed centre line, wide) and automatic sliding (panels) openings."""
    ohd = []
    slide = []
    k = max(3, int(0.5 * ppf))
    ker = np.ones(k) / k
    for si, s in enumerate(segs):
        a, band, cen, off = band_profiles(D, s)
        if band.shape[1] < 2 * k:
            continue
        cm = np.convolve(cen, ker, mode='same')
        om = np.convolve(off, ker, mode='same')
        # panels: some interior row is dark along the wall
        kk = max(3, int(1.0 * ppf))
        rowsm = cv2.blur(band.astype(np.float32), (kk, 1))
        rmax = rowsm.max(axis=0)
        rmin = rowsm.min(axis=0)
        fillmed = float(np.median(band.mean(axis=0)))
        # exclude swing-door openings on this segment
        busy = np.zeros(band.shape[1], bool)
        for sw in swings + pairs:
            if sw['seg'] == si:
                busy[max(0, int(sw['o0']) - a):max(0, int(sw['o1']) - a)] = True
        nrows = band.shape[0]
        if nrows >= 6:
            ci = int(round(s['c'])) - int(np.ceil(s['f1'] + 2.5))
            ci = int(np.clip(ci, 1, nrows - 2))
            near = [r for r in range(1, nrows - 1) if abs(r - ci) >= max(2, nrows // 4)]
            if near:
                nf = np.convolve(band[near].mean(axis=0), ker, mode='same')
                dashed = (cm > 0.12) & (cm > 2.0 * nf) & (nf < 0.2) & ~busy
                dashed = _close_runs(dashed, int(0.3 * ppf))
                for x0, x1 in _runs(dashed):
                    if (x1 - x0) >= 5.0 * ppf:
                        ohd.append(dict(seg=si, o0=a + x0, o1=a + x1, orient=s['orient'], c=s['c']))
        panel = (rmax > max(0.55, fillmed + 0.3)) & ~busy
        panel = _close_runs(panel, int(0.5 * ppf))
        for x0, x1 in _runs(panel):
            if 3.0 * ppf <= (x1 - x0) <= 10.0 * ppf:
                slide.append(dict(seg=si, o0=a + x0, o1=a + x1, orient=s['orient'], c=s['c']))
    return ohd, slide
