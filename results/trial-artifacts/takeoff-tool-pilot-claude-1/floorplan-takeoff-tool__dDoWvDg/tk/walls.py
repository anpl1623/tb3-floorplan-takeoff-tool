"""Wall detection: paired parallel heavy face lines with a wall-type fill."""
import numpy as np
import cv2


def heavy_ridges(G, bg, wthr, cmin=40.0, k=21, strip=512):
    """Ridges of horizontal dark lines (local minima across the line direction) whose
    integrated darkness over 7 rows exceeds wthr and whose contrast against the rows
    3 px away exceeds cmin.  Processed in strips to bound memory."""
    H, W = G.shape
    R = np.zeros((H, W), bool)
    pad = 8
    qthr = bg - wthr / 7.0
    for y0 in range(0, H, strip):
        a = max(0, y0 - pad); b = min(H, y0 + strip + pad)
        Sh = cv2.blur(G[a:b], (k, 1))
        Qok = cv2.blur(Sh, (1, 7)) < qthr
        c = Sh[3:-3]
        r = (c < Sh[2:-4]) & (c <= Sh[4:-2])
        r &= (np.minimum(Sh[:-6], Sh[6:]) - c) > cmin
        r &= Qok[3:-3]
        # r[i] corresponds to row a + 3 + i
        y1 = min(H, y0 + strip)
        out_lo = max(y0, a + 3); out_hi = min(y1, b - 3)
        if out_hi > out_lo:
            R[out_lo:out_hi] = r[out_lo - (a + 3):out_hi - (a + 3)]
    R[:3] = False; R[-3:] = False
    return R


def _pieces_one(G, s, bg, wthr):
    Hh, Ww = G.shape
    R = heavy_ridges(G, bg, wthr)
    dmin, dmax = int(0.33 * s), int(0.78 * s) + 1
    nxt = np.full(Ww, 30000, np.int32)
    dist = np.zeros((Hh, Ww), np.int16)
    for y in range(Hh - 1, -1, -1):
        dist[y] = np.minimum(nxt - y, 30000)
        nxt = np.where(R[y], y, nxt)
    top = R & (dist >= dmin) & (dist <= dmax)
    m = cv2.dilate(top.astype(np.uint8), np.ones((3, 1), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 0.4 * s:
            continue
        sub = top[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == i)
        ys, xs = np.nonzero(sub)
        dd = dist[y:y + h, x:x + w][sub]
        out.append(dict(a0=int(x), a1=int(x + w - 1), t=float(np.median(ys + y)), d=float(np.median(dd))))
    return out


def _fill_columns(G, p):
    yt = int(round(p['t'])); d = int(round(p['d']))
    fill = G[yt + 3:yt + d - 2, p['a0']:p['a1'] + 1]
    cm = fill.mean(0)
    cmin = fill.min(0)
    return cm, cmin


def _col_classes(cm, cmin, white=238.0):
    """Per-column fill class: 0 white(opening) 1 light 2 gray 3 hatch 4 dark/invalid."""
    cls = np.full(len(cm), 1, np.int8)
    cls[(cm > 120) & (cm < 195)] = 2
    cls[(cls == 1) & (cmin < 200)] = 3
    cls[cm > white] = 0
    cls[cmin < 105] = 4
    return cls


def detect_pieces(G, s, bg, wthr):
    GT = np.ascontiguousarray(G.T)
    res = []
    for o, img in (('H', G), ('V', GT)):
        for p in _pieces_one(img, s, bg, wthr):
            if (p['a1'] - p['a0']) < 0.4 * s:
                continue
            cm, cmin = _fill_columns(img, p)
            cls = _col_classes(cm, cmin)
            p['colmean'] = cm
            p['o'] = o
            p['c'] = p['t'] + p['d'] / 2.0
            p['colcls'] = cls
            res.append(p)
    return res


def piece_class(p, s):
    """Wall appearance class from thickness and mean fill (both robust to blur):
    thin walls are the light-filled stud partition; thick walls are either the
    solid gray rated partition or hatched CMU."""
    cls = p['colcls']
    n = len(cls)
    cnt = np.bincount(cls, minlength=5)
    valid = cnt[1] + cnt[2] + cnt[3]
    if valid < 0.3 * n or cnt[4] > 0.35 * n:
        return None
    if p['d'] < 0.52 * s:
        return 'light'
    cm = p.get('colmean')
    if cm is not None and len(cm):
        okc = cls != 0
        med = float(np.median(cm[okc])) if okc.any() else float(np.median(cm))
        return 'gray' if med < 196.0 else 'hatch'
    if cnt[2] > 0.5 * valid:
        return 'gray'
    return 'hatch'


def merge_pieces(pieces, s, tol_c=3.0, tol_d=3.0, max_gap=0.9):
    """Merge collinear pieces of the same class separated by small gaps (junctions)."""
    out = []
    for o in ('H', 'V'):
        ps = sorted([p for p in pieces if p['o'] == o], key=lambda p: (p['c'], p['a0']))
        used = [False] * len(ps)
        groups = []
        for i, p in enumerate(ps):
            if used[i]:
                continue
            grp = [p]; used[i] = True
            for j in range(i + 1, len(ps)):
                q = ps[j]
                if q['c'] - p['c'] > tol_c:
                    break
                if not used[j] and abs(q['d'] - p['d']) <= tol_d and q['cls'] == p['cls']:
                    grp.append(q); used[j] = True
            groups.append(grp)
        for grp in groups:
            grp.sort(key=lambda p: p['a0'])
            cur = None
            for p in grp:
                if cur is None:
                    cur = dict(o=o, a0=p['a0'], a1=p['a1'], c=p['c'], d=p['d'], cls=p['cls'], parts=[p])
                    continue
                if p['a0'] - cur['a1'] <= max_gap * s:
                    cur['a1'] = max(cur['a1'], p['a1'])
                    cur['parts'].append(p)
                else:
                    out.append(cur)
                    cur = dict(o=o, a0=p['a0'], a1=p['a1'], c=p['c'], d=p['d'], cls=p['cls'], parts=[p])
            if cur is not None:
                out.append(cur)
    for w in out:
        L = np.array([p['a1'] - p['a0'] for p in w['parts']], float)
        w['c'] = float(np.average([p['c'] for p in w['parts']], weights=L))
        w['d'] = float(np.average([p['d'] for p in w['parts']], weights=L))
        w['t0'] = w['c'] - w['d'] / 2.0
        w['t1'] = w['c'] + w['d'] / 2.0
    return out


def snap_ends(walls, s, tol=None):
    """Extend/trim each wall end to the centreline of a perpendicular wall it runs into."""
    if tol is None:
        tol = max(4.0, 0.35 * s)
    for w in walls:
        w['e0'] = float(w['a0']); w['e1'] = float(w['a1'])
        w['j0'] = None; w['j1'] = None
    for w in walls:
        perp = [q for q in walls if q['o'] != w['o']]
        for end in (0, 1):
            a = w['a0'] if end == 0 else w['a1']
            best = None
            span_tol = w['d'] / 2.0 + max(tol, 0.45 * s)
            for q in perp:
                # q must reach w's centreline (q may itself stop at w's face)
                if not (q['a0'] - span_tol <= w['c'] <= q['a1'] + span_tol):
                    continue
                # w's end must lie at q (inside its thickness band, or just short of it)
                if not (q['t0'] - tol <= a <= q['t1'] + tol):
                    continue
                dd = abs(a - q['c'])
                if best is None or dd < best[0]:
                    best = (dd, q)
            if best is not None:
                q = best[1]
                if end == 0:
                    w['e0'] = q['c']; w['j0'] = q
                else:
                    w['e1'] = q['c']; w['j1'] = q
    return walls


def face_line_evidence(G, bg, w, g0, g1, s):
    """Fraction of the gap [g0,g1] along wall w where lines exist at both faces."""
    if g1 - g0 < 3:
        return 1.0
    t0, t1 = w['t0'], w['t1']
    pad = 6
    lo = int(t0) - pad; hi = int(t1) + pad + 1
    a0 = int(g0); a1 = int(g1) + 1
    if w['o'] == 'H':
        patch = G[lo:hi, a0:a1]
    else:
        patch = G[a0:a1, lo:hi].T
    if patch.size == 0:
        return 0.0
    patch = cv2.blur(patch.astype(np.float32), (5, 1))
    rows = np.arange(lo, hi)
    def has_line(tf):
        sel = np.abs(rows - tf) <= 3
        dark = bg - patch[sel].min(0)
        return dark > 60
    e0 = has_line(t0); e1 = has_line(t1)
    return float((e0 & e1).mean())


def bridge_gaps(walls, G, bg, s, max_gap_ft=16.0, min_evidence=0.6):
    """Merge collinear same-class walls across door openings."""
    changed = True
    walls = list(walls)
    while changed:
        changed = False
        walls.sort(key=lambda w: (w['o'], w['c'], w['a0']))
        for i in range(len(walls)):
            w = walls[i]
            best = None
            for j in range(len(walls)):
                q = walls[j]
                if j == i or q['o'] != w['o'] or q['cls'] != w['cls']:
                    continue
                if abs(q['c'] - w['c']) > 3.0 or abs(q['d'] - w['d']) > 3.0:
                    continue
                gap = q['a0'] - w['a1']
                if gap <= 0 or gap > max_gap_ft * s:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, j)
            if best is None:
                continue
            q = walls[best[1]]
            ev = face_line_evidence(G, bg, w, w['a1'], q['a0'], s)
            if ev >= min_evidence:
                L1 = w['a1'] - w['a0']; L2 = q['a1'] - q['a0']
                w['c'] = (w['c'] * L1 + q['c'] * L2) / (L1 + L2)
                w['d'] = (w['d'] * L1 + q['d'] * L2) / (L1 + L2)
                w['t0'] = w['c'] - w['d'] / 2; w['t1'] = w['c'] + w['d'] / 2
                w['a1'] = q['a1']
                w['parts'] = w['parts'] + q['parts']
                w.setdefault('bridged', []).append((float(w['a1'] - L2 - best[0]), float(q['a0'])))
                del walls[best[1]]
                changed = True
                break
    return walls


def remove_nested(walls, s):
    """Drop walls lying inside the band of a larger wall of the same orientation."""
    keep = []
    for w in walls:
        nested = False
        for q in walls:
            if q is w or q['o'] != w['o']:
                continue
            if (q['a1'] - q['a0']) <= (w['a1'] - w['a0']):
                continue
            if q['t0'] - 2 <= w['c'] <= q['t1'] + 2 and q['a0'] - 3 <= w['a0'] and w['a1'] <= q['a1'] + 3:
                nested = True
                break
        if not nested:
            keep.append(w)
    return keep


def footprint(walls, s, shape, scale=4):
    """Building footprint mask (downsampled) from the heavy hatch walls."""
    H, W = shape
    m = np.zeros((H // scale + 1, W // scale + 1), np.uint8)
    for w in walls:
        if w['cls'] != 'hatch' or w['d'] < 0.55 * s or (w['a1'] - w['a0']) < 3.0 * s:
            continue
        if w['o'] == 'H':
            x0, x1, y0, y1 = w['a0'], w['a1'], w['t0'], w['t1']
        else:
            y0, y1, x0, x1 = w['a0'], w['a1'], w['t0'], w['t1']
        cv2.rectangle(m, (int(x0 // scale), int(y0 // scale)), (int(x1 // scale), int(y1 // scale)), 1, -1)
    k = int(max(3, 16 * s / scale)) | 1
    mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    # fill holes
    ff = mc.copy()
    pad = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 2)
    filled = ((ff != 2)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(filled, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8)


def inside(fp, x, y, scale=4):
    yi = int(y // scale); xi = int(x // scale)
    if yi < 0 or xi < 0 or yi >= fp.shape[0] or xi >= fp.shape[1]:
        return False
    return fp[yi, xi] > 0


def classify_exterior(walls, fp, s, scale=4):
    """Mark walls whose one face borders the outside of the footprint."""
    for w in walls:
        w['ext'] = False
        if w['cls'] != 'hatch':
            continue
        n_out = 0; n = 0
        for a in np.linspace(w['a0'] + 0.1 * (w['a1'] - w['a0']), w['a1'] - 0.1 * (w['a1'] - w['a0']), 9):
            for off in (-(w['d'] / 2 + 0.75 * s), (w['d'] / 2 + 0.75 * s)):
                if w['o'] == 'H':
                    x, y = a, w['c'] + off
                else:
                    x, y = w['c'] + off, a
                n += 1
                if not inside(fp, x, y, scale):
                    n_out += 1
        w['ext'] = n_out >= 6
    return walls


def wall_fill_classes(G, w, bg=250.0):
    """Per-position fill class along wall w (see _col_classes).  For thick walls the
    centre rows are ignored so that dashed thresholds inside openings read as white,
    and hatch is judged by the density of dark pixels in a small window (robust to
    isolated noise); thin walls use only their centre rows."""
    a0, a1 = int(w['a0']), int(w['a1'])
    yt = w['t0']; d = w['d']
    cen = yt + d / 2.0
    if d >= 13:
        rows = np.arange(int(np.ceil(yt + 3.5)), int(np.floor(yt + d - 3.5)) + 1)
        rows = rows[np.abs(rows - cen) > 1.6]
    else:
        rows = np.arange(int(np.ceil(cen - 1.5)), int(np.floor(cen + 1.5)) + 1)
    if len(rows) == 0:
        rows = np.array([int(round(cen))])
    if w['o'] == 'H':
        fill = G[rows][:, a0:a1 + 1]
    else:
        fill = G[:, rows][a0:a1 + 1, :].T
    cm = fill.mean(0)
    cmin = fill.min(0)
    cls = _col_classes(cm, cmin, white=min(238.0, bg - 9.0))
    if d >= 13:
        dark = (fill < 205).astype(np.float32).mean(0)
        frac = np.convolve(dark, np.ones(9) / 9.0, mode='same')
        hatch_like = frac > 0.08
        # light columns need dense dark pixels to count as hatch; white otherwise
        cls[(cls == 3) & ~hatch_like] = np.where(cm[(cls == 3) & ~hatch_like] > bg - 25, 0, 1)
    return cls


def _band_rows(w):
    yt = w['t0']; d = w['d']
    cen = yt + d / 2.0
    if d >= 13:
        rows = np.arange(int(np.ceil(yt + 3.5)), int(np.floor(yt + d - 3.5)) + 1)
        rows = rows[np.abs(rows - cen) > (2.6 if d >= 14.5 else 1.6)]
    else:
        rows = np.arange(int(np.ceil(cen - 1.5)), int(np.floor(cen + 1.5)) + 1)
    if len(rows) == 0:
        rows = np.array([int(round(cen))])
    return rows


def band_profile(G, w):
    """Smoothed mean fill along the wall band (centre rows excluded on thick walls)."""
    a0, a1 = int(w['a0']), int(w['a1'])
    rows = _band_rows(w)
    if w['o'] == 'H':
        fill = G[rows][:, a0:a1 + 1]
    else:
        fill = G[:, rows][a0:a1 + 1, :].T
    cm = fill.astype(np.float64).mean(0)
    # running median along the wall: ignores dust specks touching a few columns
    if len(cm) >= 9:
        pad = np.pad(cm, 4, mode='edge')
        win = np.lib.stride_tricks.sliding_window_view(pad, 9)
        return np.median(win, axis=1)
    return cm


def typical_fills(G, walls):
    """Typical band fill per wall class, pooled over the whole plan (length weighted)."""
    pools = {}
    for w in walls:
        pools.setdefault((w['cls'], bool(w.get('ext'))), []).append(band_profile(G, w))
    out = {}
    for k, v in pools.items():
        out[k] = float(np.median(np.concatenate(v)))
    return out


def find_openings(G, w, s, bg=None, min_ft=1.6, typical=None):
    """Openings = stretches where the band's mean fill departs from the typical fill of
    this wall class (whiter for door openings, darker for sliding-door panels)."""
    if bg is None:
        bg = 250.0
    cms = band_profile(G, w)
    F = float(np.median(cms))
    if typical is not None:
        Fp = typical.get((w['cls'], bool(w.get('ext'))))
        if Fp is not None:
            F = Fp
    tau = max(10.0, 0.45 * (bg - F))
    dev = np.abs(cms - F) > tau
    k = max(3, int(0.2 * s)) | 1
    mm = np.convolve(dev.astype(float), np.ones(k) / k, mode='same') > 0.5
    runs = []
    for i0, i1 in _runs(mm):
        if (i1 - i0) >= min_ft * s:
            runs.append((w['a0'] + i0, w['a0'] + i1 - 1))
    return [refine_jambs(G, w, s, g0, g1) for g0, g1 in runs]


def refine_jambs(G, w, s, g0, g1, reach=2.0):
    """Move opening ends to the wall end caps (heavy lines across the whole band)."""
    t0, t1 = w['t0'], w['t1']
    rows = np.arange(int(np.ceil(t0 + 2.0)), int(np.floor(t1 - 2.0)) + 1)
    if len(rows) < 2:
        return g0, g1
    lo = int(max(w['a0'] - 2, g0 - reach * s)); hi = int(min(w['a1'] + 2, g1 + reach * s))
    if w['o'] == 'H':
        band = G[rows][:, lo:hi + 1]
    else:
        band = G[lo:hi + 1][:, rows].T
    if band.shape[1] < 5:
        return g0, g1
    P = band.astype(np.float64).mean(0)
    ref = np.median(P)
    dark = P < ref - 0.45 * (ref - 30.0)
    cand = [lo + i for i in range(1, len(P) - 1) if dark[i] and P[i] <= P[i - 1] and P[i] <= P[i + 1]]
    left = [c for c in cand if g0 - reach * s <= c <= g0 + 0.5 * s]
    right = [c for c in cand if g1 - 0.5 * s <= c <= g1 + reach * s]
    n0 = min(left, key=lambda c: abs(c - g0)) if left else g0
    n1 = min(right, key=lambda c: abs(c - g1)) if right else g1
    if n1 - n0 < 1.2 * s:
        return g0, g1
    return int(n0), int(n1)


def _runs(b):
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def refine_faces(G, walls, s):
    """Subpixel face-line positions from the mean cross-profile along each wall."""
    for w in walls:
        a0, a1 = int(w['a0']), int(w['a1'])
        if a1 - a0 < 6:
            continue
        lo = int(np.floor(w['t0'])) - 5
        hi = int(np.ceil(w['t1'])) + 6
        if lo < 1:
            continue
        if w['o'] == 'H':
            patch = G[lo:hi, a0:a1 + 1]
        else:
            patch = G[a0:a1 + 1, lo:hi].T
        if patch.shape[0] < 8 or patch.shape[1] < 4:
            continue
        try:
            match = wall_fill_classes(G, w) == {'light': 1, 'gray': 2, 'hatch': 3}[w['cls']]
            if w['cls'] == 'light':
                match |= wall_fill_classes(G, w) == 3
        except Exception:
            match = np.ones(patch.shape[1], bool)
        if match.sum() >= 5 and len(match) == patch.shape[1]:
            patch = patch[:, match]
        prof = patch.astype(np.float64).mean(1)
        rows = np.arange(lo, hi)
        new = []
        for tf in (w['t0'], w['t1']):
            sel = np.nonzero(np.abs(rows - tf) <= 2.5)[0]
            if len(sel) == 0:
                new.append(tf); continue
            i = sel[np.argmin(prof[sel])]
            if 0 < i < len(prof) - 1:
                a, b, c = prof[i - 1], prof[i], prof[i + 1]
                den = a - 2 * b + c
                off = 0.5 * (a - c) / den if den > 1e-6 else 0.0
                off = float(np.clip(off, -0.5, 0.5))
                new.append(rows[i] + off)
            else:
                new.append(tf)
        if new[1] - new[0] > 0.25 * s:
            w['t0'], w['t1'] = new
            w['c'] = (new[0] + new[1]) / 2.0
            w['d'] = new[1] - new[0]
    return walls


def drop_implausible(walls, s):
    """After face refinement: drop walls thinner than any partition type, and short
    walls whose band touches a longer parallel wall (door panels, thresholds)."""
    out = []
    for w in walls:
        if w['d'] < 0.34 * s:
            continue
        bad = False
        for q in walls:
            if q is w or q['o'] != w['o'] or (q['a1'] - q['a0']) <= (w['a1'] - w['a0']):
                continue
            if min(w['a1'], q['a1']) - max(w['a0'], q['a0']) <= 0:
                continue
            gap = max(q['t0'] - w['t1'], w['t0'] - q['t1'])
            if gap < 0.25 * s:
                bad = True
                break
        if not bad:
            out.append(w)
    return out


def extend_free_ends(walls, G, bg, s, max_gap_ft=16.0, min_evidence=0.6):
    """A wall end that stops short of any wall is extended to the perpendicular wall
    ahead of it when door-frame lines continue along both faces (a door set against a
    corner leaves no wall stub between the jamb and the corner)."""
    changed = False
    for w in walls:
        for end in (0, 1):
            if (w['j0'] if end == 0 else w['j1']) is not None:
                continue
            a = w['a0'] if end == 0 else w['a1']
            best = None
            for q in walls:
                if q['o'] == w['o']:
                    continue
                if not (q['a0'] - w['d'] <= w['c'] <= q['a1'] + w['d']):
                    continue
                if end == 1:
                    gap = q['t0'] - a
                else:
                    gap = a - q['t1']
                if 0.5 * s < gap <= max_gap_ft * s and (best is None or gap < best[0]):
                    best = (gap, q)
            if best is None:
                continue
            gap, q = best
            g0, g1 = (a, q['t0']) if end == 1 else (q['t1'], a)
            if face_line_evidence(G, bg, w, g0, g1, s) < min_evidence:
                continue
            if end == 1:
                w['a1'] = int(np.floor(q['t0']))
            else:
                w['a0'] = int(np.ceil(q['t1']))
            changed = True
    return changed
