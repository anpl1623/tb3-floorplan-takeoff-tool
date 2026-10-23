"""Wall detection: heavy parallel face lines with a characteristic fill in between."""
import numpy as np
import cv2
from scipy import ndimage as ndi

# wall type thickness classes (feet, between face line centres)
THIN_MAX_FT = 0.52        # P1 (3-5/8" stud + GWB) ~0.40 ft;  thick walls ~0.62-0.68 ft
T_MIN_FT = 0.32
T_MAX_FT = 0.85
HEAVY_WEIGHT = 400.0      # integrated darkness of wall face pens (fixture pens < ~330)


def ridge_maps(D, axis, k=9, r=4):
    """Ridge strength for lines running along an image axis.

    axis=0: horizontal lines (profile taken along y); axis=1: vertical lines.
    Returns S (smoothed darkness), mask of local maxima across the line, and subpixel offsets.
    """
    if axis == 0:
        S = ndi.uniform_filter1d(D, size=k, axis=1)
        Sp = np.pad(S, ((r, r), (0, 0)), mode='edge')
        a1 = Sp[0:-2 * r]
        a2 = Sp[2 * r:]
        u1 = Sp[r - 1:-r - 1]
        u2 = Sp[r + 1:Sp.shape[0] - r + 1]
    else:
        S = ndi.uniform_filter1d(D, size=k, axis=0)
        Sp = np.pad(S, ((0, 0), (r, r)), mode='edge')
        a1 = Sp[:, 0:-2 * r]
        a2 = Sp[:, 2 * r:]
        u1 = Sp[:, r - 1:-r - 1]
        u2 = Sp[:, r + 1:Sp.shape[1] - r + 1]
    R = S - 0.5 * (a1 + a2)
    pk = (S > u1) & (S >= u2)
    den = (u1 - 2 * S + u2)
    with np.errstate(divide='ignore', invalid='ignore'):
        off = np.where(den < -1e-3, 0.5 * (u1 - u2) / den, 0.0).astype(np.float32)
    off = np.clip(off, -0.5, 0.5)
    return S, R, pk, off


def extract_segments(D, axis, thrS=100.0, thrR=60.0, minlen=12):
    S, R, pk, off = ridge_maps(D, axis)
    m = pk & (S > thrS) & (R > thrR)
    del R, pk
    lab, n = ndi.label(m, structure=np.ones((3, 3), dtype=bool))
    objs = ndi.find_objects(lab)
    segs = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        ys, xs = sl
        ext = (xs.stop - xs.start) if axis == 0 else (ys.stop - ys.start)
        if ext < minlen:
            continue
        sub = lab[sl] == (i + 1)
        yy, xx = np.nonzero(sub)
        yy = yy + ys.start
        xx = xx + xs.start
        w = S[yy, xx]
        if axis == 0:
            pos = yy + off[yy, xx]
            a, b = int(xs.start), int(xs.stop - 1)
            spread = int(yy.max() - yy.min())
        else:
            pos = xx + off[yy, xx]
            a, b = int(ys.start), int(ys.stop - 1)
            spread = int(xx.max() - xx.min())
        if spread > 4:
            # not a straight axis-aligned line; split by robust position
            med = np.median(pos)
            keep = np.abs(pos - med) <= 2.0
            if keep.sum() < minlen:
                continue
            pos = pos[keep]; w = w[keep]
            along = (xx if axis == 0 else yy)[keep]
            a, b = int(along.min()), int(along.max())
        segs.append(dict(pos=float((pos * w).sum() / w.sum()), a=a, b=b, s=float(w.mean())))
    return segs


def seg_weight(D, sg, axis, core=4, bgw=(6, 9)):
    """Integrated darkness across the line (blur and phase independent line weight).

    The background is taken from the cleaner side so that a neighbouring parallel line
    (the other face of a thin wall, a fill, a hatch) does not eat into the weight.
    """
    p = int(round(sg['pos']))
    H, Wd = D.shape
    n = sg['b'] - sg['a'] + 1
    if n > 9:
        idx = np.arange(sg['a'] + 3, sg['b'] - 2)
    else:
        idx = np.arange(sg['a'], sg['b'] + 1)
    lo, hi = p - bgw[1], p + bgw[1] + 1
    if lo < 0 or (axis == 0 and hi > H) or (axis == 1 and hi > Wd):
        return 0.0
    if axis == 0:
        prof = D[lo:hi][:, idx]
    else:
        prof = D[idx][:, lo:hi].T
    c0 = bgw[1]
    core_sum = prof[c0 - core:c0 + core + 1].sum(0)
    b1 = prof[c0 - bgw[1]:c0 - bgw[0] + 1].mean(0)
    b2 = prof[c0 + bgw[0]:c0 + bgw[1] + 1].mean(0)
    bg = np.minimum(b1, b2) * (2 * core + 1)
    return float(np.median(core_sum - bg))


def band_array(D, axis, p1, p2, lo, hi, margin=2.5):
    r0 = int(np.ceil(p1 + margin))
    r1 = int(np.floor(p2 - margin))
    if r1 < r0:
        r0 = r1 = int(round(0.5 * (p1 + p2)))
    if axis == 0:
        return D[r0:r1 + 1, lo:hi + 1]
    else:
        return D[lo:hi + 1, r0:r1 + 1].T


def interval_features(band):
    """band: rows across the wall, columns along the wall (darkness values)."""
    R, C = band.shape
    m = float(band.mean())
    feats = dict(mean=m, rows=R, cols=C)
    if R >= 3 and C >= 6:
        a = band[:-1, 1:-1].astype(np.float64)
        b_p = band[1:, 2:].astype(np.float64)
        b_m = band[1:, :-2].astype(np.float64)
        b_0 = band[1:, 1:-1].astype(np.float64)
        def ncc(x, y):
            x = x - x.mean(); y = y - y.mean()
            den = np.sqrt((x * x).sum() * (y * y).sum())
            return float((x * y).sum() / den) if den > 1e-6 else 0.0
        cp = ncc(a, b_p); cm = ncc(a, b_m); c0 = ncc(a, b_0)
        feats['diag'] = abs(cp - cm)
        feats['c0'] = c0
        # horizontal-line structure: darkness concentrated in a few rows
        rowm = band.mean(1)
        feats['rowmax'] = float(rowm.max())
        feats['rowmin'] = float(rowm.min())
        feats['colstd'] = float(band.std(1).mean())
    else:
        feats['diag'] = 0.0
        feats['c0'] = 0.0
        rowm = band.mean(1) if R else np.zeros(1)
        feats['rowmax'] = float(rowm.max()) if R else 0.0
        feats['rowmin'] = float(rowm.min()) if R else 0.0
        feats['colstd'] = float(band.std(1).mean()) if R and C > 1 else 0.0
    return feats


def classify_interval(f, thin):
    """Return 'H' (hatched CMU), 'P2' (gray fill), 'P1' (light fill), 'open' or '?'."""
    m = f['mean']
    if thin:
        if m < 12 or (f['rowmin'] < 10 and m < 22):
            return 'open'
        if 14 <= m <= 70:
            return 'P1'
        return '?'
    # hatched CMU: 45 degree structure over a mostly white band.  Heavy blur smears the hatch
    # towards a flat light gray, but every row of the band still crosses hatch lines (so no
    # row is white, unlike a door opening with a threshold or dashed line); the gray stud
    # fill is far darker.
    if 12 < m < 60 and (f['diag'] > 0.30 or (f['rowmin'] > 0.6 * m and f['rowmin'] > 10)):
        return 'H'
    # gray filled metal stud partition: uniform mid gray
    if 50 <= m <= 130 and f['rowmin'] > 40:
        return 'P2'
    if m < 12 or (f['rowmin'] < 8 and f['diag'] < 0.25):
        return 'open'
    return '?'


def find_jambs(band, thr=120.0):
    """Columns where the whole band is dark: jamb lines / door frames."""
    if band.shape[0] == 0:
        return np.zeros(band.shape[1], bool)
    colmin = band.min(0)
    return colmin > thr


def split_candidate(D, cand, ppf):
    """Split a wall candidate into intervals at jamb lines and classify them."""
    axis = cand['axis']
    lo, hi = cand['a'], cand['b']
    band = band_array(D, axis, cand['p1'], cand['p2'], lo, hi)
    thin = cand['t'] < THIN_MAX_FT * ppf
    jam = find_jambs(band)
    # intervals between jamb runs
    cuts = []
    i = 0
    n = len(jam)
    while i < n:
        if jam[i]:
            j = i
            while j + 1 < n and jam[j + 1]:
                j += 1
            cuts.append((i, j))
            i = j + 1
        else:
            i += 1
    pieces = []
    prev = 0
    for (j0, j1) in cuts:
        if j0 > prev:
            pieces.append((prev, j0 - 1))
        prev = j1 + 1
    if prev <= n - 1:
        pieces.append((prev, n - 1))
    out = []
    for (p0, p1) in pieces:
        if p1 - p0 + 1 < 3:
            continue
        sub = band[:, p0:p1 + 1]
        f = interval_features(sub)
        c = classify_interval(f, thin)
        out.append(dict(a=lo + p0, b=lo + p1, cls=c, feats=f))
    return out, [(lo + j0, lo + j1) for (j0, j1) in cuts]


def detect_walls(D, ppf):
    """Return (wall candidates, heavy line segments) for both orientations."""
    result = []
    heavy_all = {0: [], 1: []}
    for axis in (0, 1):
        segs = extract_segments(D, axis)
        heavy = []
        for sg in segs:
            if sg['b'] - sg['a'] < 0.35 * ppf:
                continue
            sg['w'] = seg_weight(D, sg, axis)
            if sg['w'] > HEAVY_WEIGHT:
                heavy.append(sg)
        heavy.sort(key=lambda q: q['pos'])
        heavy_all[axis] = heavy
        tmin, tmax = T_MIN_FT * ppf, T_MAX_FT * ppf
        for i, A in enumerate(heavy):
            for B in heavy[i + 1:]:
                d = B['pos'] - A['pos']
                if d < tmin:
                    continue
                if d > tmax:
                    break
                lo = max(A['a'], B['a'])
                hi = min(A['b'], B['b'])
                if hi - lo < 0.35 * ppf:
                    continue
                cand = dict(axis=axis, c=0.5 * (A['pos'] + B['pos']), t=d, a=lo, b=hi, p1=A['pos'], p2=B['pos'],
                            w1=A['w'], w2=B['w'])
                ivs, jambs = split_candidate(D, cand, ppf)
                cand['intervals'] = ivs
                cand['jambs'] = jambs
                result.append(cand)
    return result, heavy_all


WALL_CLASSES = ('H', 'P1', 'P2')


def _face_spans(heavy, axis, pos, a, b, tol=2.0):
    """True if a heavy line at `pos` covers the stretch [a, b] (allowing small breaks)."""
    if not heavy:
        return False
    cover = []
    for s in heavy.get(axis, []):
        if abs(s['pos'] - pos) <= tol and s['b'] >= a and s['a'] <= b:
            cover.append((max(a, s['a']), min(b, s['b'])))
    if not cover:
        return False
    cover.sort()
    got = 0.0
    end = a
    for (u, v) in cover:
        if v <= end:
            continue
        got += v - max(u, end)
        end = v
    return got >= 0.85 * (b - a)


def merge_runs(cands, ppf, heavy=None):
    """Merge collinear wall candidates into continuous runs.

    Pieces separated by more than a junction gap are still joined (up to the widest door)
    when one of the wall's face lines runs straight across the gap - a door opening.
    """
    thin_lim = THIN_MAX_FT * ppf
    groups = []
    # candidates without any wall-like interval (e.g. a stretch of face lines across a door
    # opening) still take part so that they can bridge the wall pieces on either side
    usable = [c for c in cands if c['intervals']]
    usable.sort(key=lambda c: (c['axis'], c['t'] < thin_lim, c['c']))
    for c in usable:
        placed = False
        for g in groups:
            if g['axis'] == c['axis'] and (g['t'] < thin_lim) == (c['t'] < thin_lim) and abs(g['c'] - c['c']) <= 2.5 \
                    and abs(g['t'] - c['t']) <= 1.6:
                g['members'].append(c)
                placed = True
                break
        if not placed:
            groups.append(dict(axis=c['axis'], c=c['c'], t=c['t'], members=[c]))
    runs = []
    gap_max = T_MAX_FT * ppf + 4
    open_max = 16.0 * ppf
    for g in groups:
        mem = sorted(g['members'], key=lambda c: c['a'])
        cur = None
        for c in mem:
            joined = cur is not None and c['a'] <= cur['b'] + gap_max
            if not joined and cur is not None and c['a'] - cur['b'] <= open_max:
                p1 = np.median([m['p1'] for m in cur['members']])
                p2 = np.median([m['p2'] for m in cur['members']])
                if _face_spans(heavy, g['axis'], p1, cur['b'], c['a']) or \
                        _face_spans(heavy, g['axis'], p2, cur['b'], c['a']):
                    joined = True
                    # record the bridged stretch as an opening
                    c = dict(c)
                    gap_iv = dict(a=cur['b'] + 1, b=c['a'] - 1, cls='open',
                                  feats=dict(mean=0.0, rows=0, cols=0, diag=0.0, c0=0.0, rowmax=0.0,
                                             rowmin=0.0, colstd=0.0))
                    c['intervals'] = [gap_iv] + list(c['intervals'])
            if joined:
                cur['b'] = max(cur['b'], c['b'])
                cur['members'].append(c)
            else:
                if cur is not None:
                    runs.append(cur)
                cur = dict(axis=g['axis'], a=c['a'], b=c['b'], members=[c])
        if cur is not None:
            runs.append(cur)
    out = []
    for r in runs:
        ws = np.array([m['b'] - m['a'] for m in r['members']], float)
        r['c'] = float(np.average([m['c'] for m in r['members']], weights=ws))
        r['t'] = float(np.average([m['t'] for m in r['members']], weights=ws))
        r['p1'] = float(np.average([m['p1'] for m in r['members']], weights=ws))
        r['p2'] = float(np.average([m['p2'] for m in r['members']], weights=ws))
        ivs = []
        for m in r['members']:
            ivs.extend(m['intervals'])
        ivs.sort(key=lambda iv: iv['a'])
        # de-duplicate overlapping intervals (overlapping candidates)
        clean = []
        for iv in ivs:
            if clean and iv['a'] <= clean[-1]['b']:
                prev = clean[-1]
                # keep the longer one's class
                if (iv['b'] - iv['a']) > (prev['b'] - prev['a']):
                    clean[-1] = dict(iv)
                    clean[-1]['a'] = min(prev['a'], iv['a'])
                else:
                    prev['b'] = max(prev['b'], iv['b'])
                continue
            clean.append(dict(iv))
        r['intervals'] = clean
        out.append(r)
    # thin walls: judge each interval against the run's own fill (robust to blur, which
    # bleeds the face lines into the narrow band between them)
    for r in out:
        if r['t'] >= thin_lim:
            continue
        fills = [iv for iv in r['intervals'] if iv['cls'] == 'P1' and (iv['b'] - iv['a']) >= 1.0 * ppf]
        if not fills:
            continue
        wts = np.array([iv['b'] - iv['a'] + 1 for iv in fills], float)
        order = np.argsort([iv['feats']['mean'] for iv in fills])
        cum = np.cumsum(wts[order])
        k = order[int(np.searchsorted(cum, 0.5 * cum[-1]))]
        ref_mean = fills[k]['feats']['mean']
        ref_rowmin = float(np.median([iv['feats']['rowmin'] for iv in fills]))
        for iv in r['intervals']:
            fm = iv['feats']
            if iv['cls'] not in ('P1', 'open', '?'):
                continue
            if fm['rowmin'] < 0.45 * ref_rowmin and fm['mean'] < 0.8 * ref_mean:
                iv['cls'] = 'open'
            elif fm['mean'] > 1.45 * ref_mean + 5:
                iv['cls'] = '?'
            elif iv['cls'] != 'P1' and abs(fm['mean'] - ref_mean) < 0.3 * ref_mean and fm['rowmin'] > 0.6 * ref_rowmin:
                iv['cls'] = 'P1'
    # no opening is wider than this: longer stretches without wall fill are not one wall
    max_open = 16.0 * ppf
    pieces = []
    for r in out:
        cur_start = r['a']
        last_wall_end = None
        cut_points = []
        for iv in r['intervals']:
            if iv['cls'] in WALL_CLASSES:
                if last_wall_end is not None and iv['a'] - last_wall_end > max_open:
                    cut_points.append((last_wall_end, iv['a']))
                last_wall_end = iv['b']
        if not cut_points:
            pieces.append(r)
            continue
        bounds = [r['a']] + [x for cp in cut_points for x in cp] + [r['b']]
        for k in range(0, len(bounds), 2):
            sa, sb = bounds[k], bounds[k + 1]
            nr = dict(r)
            nr['a'], nr['b'] = sa, sb
            nr['intervals'] = [iv for iv in r['intervals'] if iv['b'] >= sa and iv['a'] <= sb]
            pieces.append(nr)
    out = pieces
    # split runs at wall-type changes
    final = []
    for r in out:
        seq = [iv for iv in r['intervals'] if iv['cls'] in WALL_CLASSES]
        if not seq:
            continue
        cur_cls = None
        start = r['a']
        segs = []
        for k, iv in enumerate(seq):
            if cur_cls is None:
                cur_cls = iv['cls']
                seg_start = start
                last_end = iv['b']
                continue
            if iv['cls'] != cur_cls and (iv['b'] - iv['a']) > 0.5 * ppf:
                # check that the previous piece is substantial too
                cut = 0.5 * (last_end + iv['a'])
                segs.append((seg_start, cut, cur_cls))
                seg_start = cut
                cur_cls = iv['cls']
            last_end = iv['b']
        segs.append((seg_start, r['b'], cur_cls))
        for (sa, sb, cls) in segs:
            nr = dict(axis=r['axis'], c=r['c'], t=r['t'], p1=r['p1'], p2=r['p2'], a=float(sa), b=float(sb), cls=cls)
            nr['intervals'] = [iv for iv in r['intervals'] if iv['b'] >= sa and iv['a'] <= sb]
            final.append(nr)
    return final


def snap_runs(runs, ppf, tol=None):
    """Snap run ends to the centrelines of perpendicular walls they run into.

    Sets r['sa'], r['sb'] (snapped ends), r['ja'], r['jb'] (index of wall met or None).
    """
    if tol is None:
        tol = max(4.0, 0.25 * ppf)
    n = len(runs)
    for i, W in enumerate(runs):
        for end in ('a', 'b'):
            e = W[end]
            best = None
            for j, X in enumerate(runs):
                if j == i or X['axis'] == W['axis']:
                    continue
                d = abs(X['c'] - e)
                if d > X['t'] / 2.0 + tol:
                    continue
                if not (X['a'] - (W['t'] / 2.0 + tol) <= W['c'] <= X['b'] + (W['t'] / 2.0 + tol)):
                    continue
                if best is None or d < best[0]:
                    best = (d, j)
            if best is not None:
                W['s' + end] = runs[best[1]]['c']
                W['j' + end] = best[1]
            else:
                W['s' + end] = float(e)
                W['j' + end] = None
    return runs


def main_structure(runs, ppf):
    """Keep the largest connected set of walls (the building)."""
    n = len(runs)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, r in enumerate(runs):
        for k in ('ja', 'jb'):
            j = r.get(k)
            if j is not None:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    tot = {}
    for i, r in enumerate(runs):
        tot.setdefault(find(i), 0.0)
        tot[find(i)] += r['sb'] - r['sa']
    if not tot:
        return []
    root = max(tot, key=tot.get)
    return [i for i in range(n) if find(i) == root]


def render_walls(shape, runs, idxs, fill_openings=True):
    """Render wall bodies (pixel centres between the face line centres) into a mask."""
    m = np.zeros(shape, np.uint8)
    for i in idxs:
        r = runs[i]
        lo = min(r['a'], r['sa'])
        hi = max(r['b'], r['sb'])
        p1, p2 = r['p1'], r['p2']
        c0 = int(np.ceil(p1)); c1 = int(np.floor(p2))
        a0 = int(np.floor(lo + 0.5)); a1 = int(np.floor(hi + 0.5))
        if r['axis'] == 0:
            m[c0:c1 + 1, a0:a1 + 1] = 1
        else:
            m[a0:a1 + 1, c0:c1 + 1] = 1
    return m


def exterior_mask(shape, runs, idxs, close=3):
    wm = render_walls(shape, runs, idxs)
    # close tiny gaps at corners
    wmd = cv2.dilate(wm, np.ones((close, close), np.uint8))
    free = (wmd == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    # exterior = components touching the image border
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    ext = np.isin(lab, list(border))
    # the dilation above shrank the exterior; give it back
    ext = (cv2.dilate(ext.astype(np.uint8), np.ones((close, close), np.uint8)) > 0) & (wm == 0)
    return ext, wm


def classify_exterior(shape, runs, idxs, ppf):
    """Mark hatched runs on the building perimeter as EXT. Returns exterior mask too."""
    ext, wm = exterior_mask(shape, runs, idxs)
    # sanity check: the walls must enclose a substantial interior; otherwise a gap in the
    # perimeter let the flood fill in - close larger gaps and try again
    ys, xs = np.nonzero(wm)
    if len(ys):
        bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        inner = float((~ext[ys.min():ys.max() + 1, xs.min():xs.max() + 1]).sum())
        for close in (9, 21, 41):
            if inner >= 0.5 * bbox_area:
                break
            ext, wm = exterior_mask(shape, runs, idxs, close=close)
            inner = float((~ext[ys.min():ys.max() + 1, xs.min():xs.max() + 1]).sum())
    for i in idxs:
        r = runs[i]
        if r['cls'] != 'H':
            r['type'] = r['cls']
            continue
        lo = int(max(r['a'], r['sa']) + 2); hi = int(min(r['b'], r['sb']) - 2)
        if hi <= lo:
            lo, hi = int(r['a']), int(r['b'])
        samp = np.arange(lo, hi + 1, max(1, (hi - lo) // 50 + 1))
        off = 4
        if r['axis'] == 0:
            ya = int(round(r['p1'] - off)); yb = int(round(r['p2'] + off))
            ya = min(max(ya, 0), shape[0] - 1); yb = min(max(yb, 0), shape[0] - 1)
            fa = ext[ya, samp].mean(); fb = ext[yb, samp].mean()
        else:
            xa = int(round(r['p1'] - off)); xb = int(round(r['p2'] + off))
            xa = min(max(xa, 0), shape[1] - 1); xb = min(max(xb, 0), shape[1] - 1)
            fa = ext[samp, xa].mean(); fb = ext[samp, xb].mean()
        r['type'] = 'EXT' if max(fa, fb) > 0.5 else 'P3'
        r['ext_side'] = (-1 if fa > fb else 1) if max(fa, fb) > 0.5 else 0
    return ext


def add_free_standing(runs, idxs, exterior, ppf, min_ft=3.0):
    """Walls standing free on the floor inside the building (touching no other wall)."""
    keep = set(idxs)
    out = list(idxs)
    body = None
    for i, r in enumerate(runs):
        if i in keep or r['cls'] not in WALL_CLASSES:
            continue
        if (r['b'] - r['a']) < min_ft * ppf:
            continue
        if body is None:
            body = cv2.dilate(render_walls(exterior.shape, runs, idxs), np.ones((7, 7), np.uint8)) > 0
        c = int(round(r['c']))
        along = np.linspace(r['a'], r['b'], 15).astype(int)
        if r['axis'] == 0:
            ys, xs = np.full(len(along), c), along
        else:
            ys, xs = along, np.full(len(along), c)
        ok = (ys >= 0) & (ys < exterior.shape[0]) & (xs >= 0) & (xs < exterior.shape[1])
        if not ok.all():
            continue
        on_floor = (~exterior[ys, xs]) & (~body[ys, xs])
        if on_floor.mean() > 0.9:
            r['type'] = 'P3' if r['cls'] == 'H' else r['cls']
            out.append(i)
    return out


def wall_lengths(runs, idxs, ppf):
    tot = {}
    for i in idxs:
        r = runs[i]
        L = (r['sb'] - r['sa']) / ppf
        tot[r['type']] = tot.get(r['type'], 0.0) + L
    return tot
