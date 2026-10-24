"""Wall detection: stroke tracks, pairing into wall bands, typing, centreline lengths."""
import numpy as np
import cv2
from common import CAL


def stroke_tracks(dk, ppf, th=0.5, min_len_ft=0.8, smooth=5, ridge=0.12, rr=3):
    """Horizontal stroke centre tracks.

    Returns list of dicts: y (sub-pixel centre), x0, x1, peak (median darkness)."""
    th = th * CAL['k']
    ridge = ridge * CAL['k']
    sm = cv2.blur(dk, (smooth, 1))
    H, W = sm.shape
    up = sm[:-2]
    mid = sm[1:-1]
    dn = sm[2:]
    pk = (mid > th) & (mid >= up) & (mid > dn)
    peak = np.zeros(sm.shape, bool)
    peak[1:-1] = pk
    # ridge condition: darker than both sides at distance rr
    side_up = np.zeros_like(sm); side_up[rr:] = sm[:-rr]
    side_dn = np.zeros_like(sm); side_dn[:-rr] = sm[rr:]
    peak &= (sm - side_up > ridge) & (sm - side_dn > ridge)
    ys, xs = np.nonzero(peak)
    u = up[ys - 1, xs]; m = mid[ys - 1, xs]; d = dn[ys - 1, xs]
    den = (u - 2 * m + d)
    off = np.where(np.abs(den) > 1e-6, 0.5 * (u - d) / np.where(np.abs(den) > 1e-6, den, 1), 0.0)
    off = np.clip(off, -0.5, 0.5)
    yc = ys + off
    yb = np.round(yc * 2).astype(np.int32)
    S2 = np.zeros((2 * H + 2, W), np.uint8)
    S2[yb, xs] = 1
    S2 = cv2.morphologyEx(S2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(S2, connectivity=8)
    min_len = max(8, int(min_len_ft * ppf))
    good = np.zeros(n, bool)
    good[1:] = st[1:, cv2.CC_STAT_WIDTH] >= min_len
    lab_at = lab[yb, xs]
    sel = good[lab_at]
    tracks = []
    if not sel.any():
        return tracks
    la = lab_at[sel]; yv = yc[sel]; xv = xs[sel]; pv = sm[ys[sel], xs[sel]]
    order = np.argsort(la, kind='stable')
    la = la[order]; yv = yv[order]; xv = xv[order]; pv = pv[order]
    bounds = np.nonzero(np.diff(la))[0] + 1
    for grp in np.split(np.arange(len(la)), bounds):
        if len(grp) < min_len * 0.6:
            continue
        for sg in _split_levels(xv[grp], yv[grp], run=max(4, int(0.4 * ppf))):
            g2 = grp[sg]
            if len(g2) < min_len * 0.6:
                continue
            x0 = int(xv[g2].min()); x1 = int(xv[g2].max()) + 1
            if x1 - x0 < min_len:
                continue
            tracks.append(dict(y=float(np.median(yv[g2])), x0=x0, x1=x1,
                               peak=float(np.median(pv[g2])), n=len(g2)))
    return tracks


def _split_levels(xs, ys, jump=1.2, run=4):
    """Split a stroke track where its centre position steps (two collinear strokes merged)."""
    order = np.argsort(xs, kind='stable')
    xs_o = xs[order]
    ys_o = ys[order]
    ux, inv = np.unique(xs_o, return_inverse=True)
    if len(ux) < 3 * run:
        return [order]
    ycol = np.zeros(len(ux))
    np.add.at(ycol, inv, ys_o)
    ycol /= np.bincount(inv)
    segs = []
    start = 0
    ref = np.median(ycol[:run])
    i = run
    while i < len(ux):
        if abs(ycol[i] - ref) > jump:
            # check persistence
            j = min(len(ux), i + run)
            if np.all(np.abs(ycol[i:j] - ref) > jump):
                segs.append((start, i))
                start = i
                ref = np.median(ycol[i:j])
                i = j
                continue
        # slowly update reference
        lo = max(start, i - 30)
        ref = np.median(ycol[lo:i + 1])
        i += 1
    segs.append((start, len(ux)))
    if len(segs) == 1:
        return [order]
    out = []
    for a, b in segs:
        sel = (inv >= a) & (inv < b)
        out.append(order[sel])
    return out


def pair_tracks(tracks, ppf, tmin_ft=0.25, tmax_ft=0.95, min_overlap_ft=0.5):
    """Pair parallel stroke tracks into wall band candidates."""
    tr = sorted(tracks, key=lambda t: t['y'])
    out = []
    for i, a in enumerate(tr):
        for j in range(i + 1, len(tr)):
            b = tr[j]
            d = b['y'] - a['y']
            if d > tmax_ft * ppf:
                break
            if d < tmin_ft * ppf:
                continue
            o0, o1 = max(a['x0'], b['x0']), min(a['x1'], b['x1'])
            if o1 - o0 < min_overlap_ft * ppf:
                continue
            out.append(dict(ya=a['y'], yb=b['y'], x0=o0, x1=o1, t=d, ym=0.5 * (a['y'] + b['y']),
                            pa=a['peak'], pb=b['peak'], ia=i, ib=j))
    return out


def band_descriptor(band):
    """Descriptor of a wall poche band (rows across thickness, cols along wall)."""
    band = np.asarray(band, np.float32)
    h, w = band.shape
    res = dict(m=float(band.mean()) if band.size else 0.0, G=0.0, a=0.0, b=0.0)
    if h >= 3 and w >= 5:
        gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
        gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
        G = float((gx * gx + gy * gy).mean())
        if G > 1e-9:
            res['G'] = G
            res['a'] = float(((gx + gy) ** 2 - (gx - gy) ** 2).mean() / (2 * G))
            res['b'] = float((gy * gy - gx * gx).mean() / G)
    return res


def legend_wall_profiles(dk, swatches, ppf):
    """Measure stroke positions and poché descriptors of the partition legend swatches."""
    out = []
    for (x, y, w, h) in swatches:
        pad = 4
        sub = dk[max(0, y - pad):y + h + pad, x + 6:x + w - 6]
        prof = sub.mean(1)
        # two strongest peaks
        pk = [i for i in range(1, len(prof) - 1) if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1]]
        pk = sorted(pk, key=lambda i: -prof[i])[:2]
        pk.sort()
        def sub_pix(i):
            a, b, c = prof[i - 1], prof[i], prof[i + 1]
            den = a - 2 * b + c
            return i + (0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0)
        ya, yb = sub_pix(pk[0]), sub_pix(pk[1])
        r0 = int(np.ceil(ya + 2.5)); r1 = int(np.floor(yb - 2.5)) + 1
        band = sub[r0:r1, 4:-4]
        d = band_descriptor(band)
        d['t'] = yb - ya
        d['peak'] = float(min(prof[pk[0]], prof[pk[1]]))
        out.append(d)
    return out


def candidate_descriptor(dk_aligned, c, ppf, block_ft=2.0):
    """Median block descriptor of a wall candidate in the aligned image."""
    r0 = int(np.ceil(c['ya'] + 2.5))
    r1 = int(np.floor(c['yb'] - 2.5)) + 1
    if r1 - r0 < 1:
        r0 = int(round(c['ym'])); r1 = r0 + 1
    x0, x1 = c['x0'] + 2, c['x1'] - 2
    bl = max(12, int(block_ft * ppf))
    descs = []
    xs = list(range(x0, max(x0 + 1, x1 - bl + 1), bl)) or [x0]
    for xa in xs:
        xb = min(x1, xa + bl)
        if xb - xa < 8:
            continue
        descs.append(band_descriptor(dk_aligned[r0 - 1:r1 + 1, xa:xb][1:-1] if r1 - r0 >= 1 else dk_aligned[r0:r1, xa:xb]))
    if not descs:
        descs = [band_descriptor(dk_aligned[r0:r1, x0:x1])]
    keys = descs[0].keys()
    med = {k: float(np.median([d[k] for d in descs])) for k in keys}
    med['nblk'] = len(descs)
    return med, descs


FEATS = ('m', 'lg', 'a')


def _feat(d):
    G = d.get('G', 0.0)
    w = G / (G + 0.05)
    return dict(m=d['m'], lg=float(np.log(G + 0.01)), a=d['a'] * w, t=d.get('t', 0.0))


def type_distance(dp, dl):
    fp, fl = _feat(dp), _feat(dl)
    s = ((fp['m'] - fl['m']) / 0.05) ** 2
    s += ((fp['lg'] - fl['lg']) / 0.6) ** 2
    s += ((fp['a'] - fl['a']) / 0.35) ** 2
    s += ((fp['t'] - fl['t']) / 1.6) ** 2
    return float(s)


def detect_wall_pieces(dk, ppf, legend_walls, x_limit):
    """Find wall band pieces (H and V) and classify them against legend types.

    Returns list of dicts with: o ('H'/'V'), c (centre coord across), t, lo, hi (extent along),
    type (best interior type), dist (dict of distances), pk (stroke darkness)."""
    sub = np.ascontiguousarray(dk[:, :x_limit])
    leg = dict(zip(['EXT', 'P3', 'P2', 'P1'], legend_walls))
    t_thin = leg['P1']['t']
    t_thick = max(leg['EXT']['t'], leg['P3']['t'], leg['P2']['t'])
    pieces = []
    for o, img in (('H', sub), ('V', np.ascontiguousarray(sub.T))):
        tr = stroke_tracks(img, ppf)
        cs = pair_tracks(tr, ppf, tmin_ft=0.6 * t_thin / ppf, tmax_ft=1.3 * t_thick / ppf)
        for c in cs:
            d, _ = candidate_descriptor(img, c, ppf)
            d['t'] = c['t']
            dist = {k: type_distance(d, leg[k]) for k in ('P3', 'P2', 'P1')}
            best = min(dist, key=dist.get)
            pieces.append(dict(o=o, c=c['ym'], t=c['t'], lo=c['x0'], hi=c['x1'], ya=c['ya'], yb=c['yb'],
                               type=best, dist=dist, pk=min(c['pa'], c['pb']), desc=d))
    return pieces


def classify_blocks(img, c, ppf, leg, block_ft=1.5, dmax=8.0, types=('P3', 'P2', 'P1')):
    """Classify a candidate by voting over blocks; returns (type, votes, n_unknown)."""
    _, descs = candidate_descriptor(img, c, ppf, block_ft=block_ft)
    votes = {}
    unk = 0
    for d in descs:
        d = dict(d)
        d['t'] = c['t']
        dist = {k: type_distance(d, leg[k]) for k in types}
        b = min(dist, key=dist.get)
        if dist[b] > dmax:
            unk += 1
            continue
        votes[b] = votes.get(b, 0) + 1
    if not votes:
        return None, votes, unk
    return max(votes, key=votes.get), votes, unk


def detect_walls(dk, ppf, legend_walls, x_limit):
    sub = np.ascontiguousarray(dk[:, :x_limit])
    leg = dict(zip(['EXT', 'P3', 'P2', 'P1'], legend_walls))
    t_thin = leg['P1']['t']
    t_thick = max(leg['EXT']['t'], leg['P3']['t'], leg['P2']['t'])
    # if the office draws the exterior wall like the 8" CMU partition, treat them as one class here
    d_ep = type_distance(dict(leg['EXT']), dict(leg['P3']))
    types = ('P3', 'P2', 'P1') if d_ep < 4.0 else ('EXT', 'P3', 'P2', 'P1')
    raw = []
    for o, img in (('H', sub), ('V', np.ascontiguousarray(sub.T))):
        tr = stroke_tracks(img, ppf)
        cs = pair_tracks(tr, ppf, tmin_ft=0.6 * t_thin / ppf, tmax_ft=1.3 * t_thick / ppf)
        for c in cs:
            if c['x1'] - c['x0'] < 1.0 * ppf:
                continue
            L_ft = (c['x1'] - c['x0']) / ppf
            typ, votes, unk = classify_blocks(img, c, ppf, leg, block_ft=float(np.clip(L_ft / 6.0, 0.6, 1.5)),
                                              types=types)
            if typ is None:
                # fallback: overall descriptor + thickness agreement
                d, _ = candidate_descriptor(img, c, ppf, block_ft=max(1.0, L_ft))
                d['t'] = c['t']
                dist = {k: type_distance(d, leg[k]) for k in types}
                b = min(dist, key=dist.get)
                if dist[b] < 40.0 and abs(c['t'] - leg[b]['t']) < 1.5 and L_ft >= 2.0:
                    typ, votes, unk = b, {b: 0}, -1
                else:
                    continue
            raw.append(dict(o=o, c=c['ym'], t=c['t'], lo=float(c['x0']), hi=float(c['x1']),
                            ya=c['ya'], yb=c['yb'], type=typ, votes=votes, unk=unk,
                            pk=min(c['pa'], c['pb'])))
    if not raw:
        return [], raw
    ref = np.median([p['pk'] for p in sorted(raw, key=lambda p: -(p['hi'] - p['lo']))[:8]])
    raw = [p for p in raw if p['pk'] >= 0.7 * ref]
    # drop pieces contained in another piece's band
    keep = []
    for p in raw:
        inside = False
        for q in raw:
            if q is p or q['o'] != p['o']:
                continue
            ov = min(p['hi'], q['hi']) - max(p['lo'], q['lo'])
            if ov < 0.6 * (p['hi'] - p['lo']):
                continue
            if abs(p['c'] - q['c']) < 0.5 * q['t'] and p['t'] < q['t'] - 1.0 and \
                    (q['hi'] - q['lo']) >= (p['hi'] - p['lo']):
                inside = True
                break
        if not inside:
            keep.append(p)
    raw = keep
    # merge collinear same-type pieces
    walls = []
    used = [False] * len(raw)
    order = sorted(range(len(raw)), key=lambda i: (raw[i]['o'], raw[i]['c'], raw[i]['lo']))
    for i in order:
        if used[i]:
            continue
        p = raw[i]
        grp = [p]
        used[i] = True
        for j in order:
            if used[j]:
                continue
            q = raw[j]
            if q['o'] != p['o'] or q['type'] != p['type']:
                continue
            if abs(q['c'] - p['c']) > 2.0 or abs(q['t'] - p['t']) > 2.0:
                continue
            grp.append(q)
            used[j] = True
        # split group into chains by gaps
        grp.sort(key=lambda q: q['lo'])
        img = sub if p['o'] == 'H' else None
        chain = [grp[0]]
        for q in grp[1:]:
            last_hi = max(r['hi'] for r in chain)
            gap = q['lo'] - last_hi
            if gap <= 1.2 * ppf or (gap <= 16 * ppf and _face_evidence(sub, p, last_hi, q['lo'])):
                chain.append(q)
            else:
                walls.append(_mk_wall(chain))
                chain = [q]
        walls.append(_mk_wall(chain))
    # drop short bands lying inside a thicker wall band (door panels, symbols drawn in openings)
    final = []
    for p in walls:
        inside = False
        L = p['hi'] - p['lo']
        if L < 16.0 * ppf:
            for q in walls:
                if q is p or q['o'] != p['o'] or (q['hi'] - q['lo']) <= max(L, 2.0 * L - 5 * ppf):
                    continue
                ov = min(p['hi'], q['hi']) - max(p['lo'], q['lo'])
                if ov >= 0.8 * L and abs(p['c'] - q['c']) < 0.5 * max(q['t'], p['t']):
                    inside = True
                    break
        if not inside:
            final.append(p)
    return final, raw


def _face_evidence(sub, p, a, b, th=0.3, frac=0.6):
    a, b = int(a), int(b)
    if b - a < 2:
        return True
    best = 0.0
    for yy in (p['ya'], p['yb']):
        y = int(round(yy))
        if p['o'] == 'H':
            strip = sub[max(0, y - 1):y + 2, a:b]
            cov = (strip.max(0) > th).mean() if strip.size else 0
        else:
            strip = sub[a:b, max(0, y - 1):y + 2]
            cov = (strip.max(1) > th).mean() if strip.size else 0
        best = max(best, cov)
    return best >= frac


def _mk_wall(chain):
    L = np.array([q['hi'] - q['lo'] for q in chain])
    return dict(o=chain[0]['o'], c=float(np.average([q['c'] for q in chain], weights=L)),
                t=float(np.average([q['t'] for q in chain], weights=L)),
                lo=float(min(q['lo'] for q in chain)), hi=float(max(q['hi'] for q in chain)),
                type=chain[0]['type'], n=len(chain))


def wall_rect(w):
    """Axis-aligned band rectangle (x0, y0, x1, y1) in image coords (float)."""
    if w['o'] == 'H':
        return (w['lo'], w['c'] - w['t'] / 2, w['hi'], w['c'] + w['t'] / 2)
    return (w['c'] - w['t'] / 2, w['lo'], w['c'] + w['t'] / 2, w['hi'])


def mark_exterior(walls, shape, ppf):
    """Flag walls on the building perimeter (adjacent to outside)."""
    f = 4
    h, w = shape[0] // f + 1, shape[1] // f + 1
    M = np.zeros((h, w), np.uint8)
    for wl in walls:
        x0, y0, x1, y1 = wall_rect(wl)
        cv2.rectangle(M, (int(x0 / f), int(y0 / f)), (int(np.ceil(x1 / f)), int(np.ceil(y1 / f))), 1, -1)
    r = max(1, int(round(3.0 * ppf / f)))
    Mc = cv2.morphologyEx(M, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    inv = (1 - Mc).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    outside = np.isin(lab, list(border))
    for wl in walls:
        x0, y0, x1, y1 = wall_rect(wl)
        d = wl['t'] / 2 + 0.6 * ppf
        n_out = 0
        n_tot = 0
        for s in np.linspace(0.1, 0.9, 9):
            if wl['o'] == 'H':
                x = wl['lo'] + s * (wl['hi'] - wl['lo'])
                pts = [(x, wl['c'] - d), (x, wl['c'] + d)]
            else:
                y = wl['lo'] + s * (wl['hi'] - wl['lo'])
                pts = [(wl['c'] - d, y), (wl['c'] + d, y)]
            for (px, py) in pts:
                ix, iy = int(px / f), int(py / f)
                if 0 <= ix < w and 0 <= iy < h:
                    n_tot += 1
                    n_out += bool(outside[iy, ix])
        wl['ext'] = n_out >= 4
    return outside


def centreline_lengths(walls, ppf, tol_ft=0.45):
    """Extend wall ends to the centreline of walls they run into; return per-wall (a, b)."""
    tol = tol_ft * ppf
    for wl in walls:
        ends = []
        for side in ('lo', 'hi'):
            e = wl[side]
            best = None
            for ot in walls:
                if ot is wl or ot['o'] == wl['o']:
                    continue
                # perpendicular wall: its centre coordinate is along our axis
                if abs(e - ot['c']) <= ot['t'] / 2 + tol and ot['lo'] - tol - wl['t'] / 2 <= wl['c'] <= ot['hi'] + tol + wl['t'] / 2:
                    dd = abs(e - ot['c'])
                    if best is None or dd < best[0]:
                        best = (dd, ot['c'])
            ends.append(best[1] if best is not None else e)
        wl['a'], wl['b'] = ends
        wl['len'] = (wl['b'] - wl['a']) / ppf
    return walls


def retype_interior(walls, dk, ppf, legend_walls):
    """Interior walls that were typed like the exterior wall get the nearest partition type."""
    leg = dict(zip(['EXT', 'P3', 'P2', 'P1'], legend_walls))
    for w in walls:
        if w.get('ext') or w['type'] in ('P3', 'P2', 'P1'):
            continue
        img = dk if w['o'] == 'H' else dk.T
        c = dict(ya=w['c'] - w['t'] / 2, yb=w['c'] + w['t'] / 2, ym=w['c'], x0=int(w['lo']), x1=int(w['hi']),
                 t=w['t'])
        d, _ = candidate_descriptor(img, c, ppf, block_ft=1.5)
        d['t'] = w['t']
        dist = {k: type_distance(d, leg[k]) for k in ('P3', 'P2', 'P1')}
        w['type'] = min(dist, key=dist.get)
    return walls
