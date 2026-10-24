"""Wall detection: paired heavy outlines with a poche fill between them."""
import numpy as np
import cv2


def strong_line_map(kn, ppf, horizontal=True, tau=0.45):
    """Binary map of long dark strokes running along x (horizontal=True) or y."""
    ke = max(3, int(round(0.3 * ppf)))
    kb = max(3, int(round(0.5 * ppf)))
    if horizontal:
        E = cv2.erode(kn, np.ones((1, ke), np.uint8))
        M = cv2.blur(E, (kb, 1))
    else:
        E = cv2.erode(kn, np.ones((ke, 1), np.uint8))
        M = cv2.blur(E, (1, kb))
    return M


def _runs(col):
    """Start/end (inclusive) of True runs in a 1-D bool array."""
    d = np.diff(np.r_[0, col.astype(np.int8), 0])
    s = np.nonzero(d == 1)[0]
    e = np.nonzero(d == -1)[0] - 1
    return s, e


def detect_bands(kn, ppf, tau=0.45, horizontal=True):
    """Detect wall-like bands running along x (or y if horizontal=False).

    Returns list of dicts with keys: a0, a1 (extent along the run direction),
    c1, c2 (outline centre positions across), and per-sample arrays."""
    img = kn if horizontal else kn.T
    M = strong_line_map(img, ppf, True, tau)
    HM = (M > tau).astype(np.uint8)
    # bridge small breaks in noisy outlines
    HM = cv2.morphologyEx(HM, cv2.MORPH_CLOSE, np.ones((1, max(3, int(round(0.2 * ppf)))), np.uint8))
    g = int(round(0.95 * ppf))
    F = cv2.morphologyEx(HM, cv2.MORPH_CLOSE, np.ones((g, 1), np.uint8))
    F = cv2.morphologyEx(F, cv2.MORPH_OPEN, np.ones((max(3, int(round(0.3 * ppf))), 1), np.uint8))
    F = cv2.morphologyEx(F, cv2.MORPH_OPEN, np.ones((1, max(3, int(round(0.8 * ppf)))), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(F, 4)
    bands = []
    raw = []
    step = 2
    for i in range(1, n):
        x0, y0, w, h, a = st[i]
        if w < 0.8 * ppf:
            continue
        sub_lab = lab[y0:y0 + h, x0:x0 + w] == i
        sub_hm = HM[y0:y0 + h, x0:x0 + w].astype(bool) & sub_lab
        sub_k = img[y0:y0 + h, x0:x0 + w]
        sub_m = M[y0:y0 + h, x0:x0 + w]
        xs, tops, bots, nruns, thick = [], [], [], [], []
        otops, obots = [], []
        for xx in range(0, w, step):
            col = sub_hm[:, xx]
            if not col.any():
                continue
            s, e = _runs(col)
            # outline centres from ink-weighted centroid of first and last runs
            kcol = sub_k[:, xx]
            def cen_of(a0, a1):
                ys = np.arange(a0, a1 + 1)
                wv = kcol[a0:a1 + 1]
                return float((ys * wv).sum() / max(wv.sum(), 1e-6))
            def xing(i_in, i_out):
                # subpixel threshold crossing between an inside and outside row
                if i_out < 0 or i_out >= len(col):
                    return float(i_in)
                mi, mo = Mcol[i_in], Mcol[i_out]
                f_ = (mi - tau) / max(mi - mo, 1e-6)
                return i_in + (i_out - i_in) * float(np.clip(f_, 0, 1))
            Mcol = sub_m[:, xx]
            ot = xing(s[0], s[0] - 1) + y0
            ob = xing(e[-1], e[-1] + 1) + y0
            otops.append(ot)
            obots.append(ob)
            if len(s) >= 2:
                t = cen_of(s[0], e[0])
                b = cen_of(s[-1], e[-1])
                tops.append(t + y0)
                bots.append(b + y0)
                nruns.append(len(s))
                thick.append((e[0] - s[0] + 1 + e[-1] - s[-1] + 1) / 2.0)
                xs.append(xx + x0)
            else:
                # single thick run (solid poche) : use edges
                tops.append(s[0] + y0 - 0.5)
                bots.append(e[0] + y0 + 0.5)
                nruns.append(1)
                thick.append(float(e[0] - s[0] + 1))
                xs.append(xx + x0)
        if len(xs) < 3:
            continue
        raw.append((x0, y0, w, h, np.array(xs, float), np.array(tops), np.array(bots),
                    np.array(nruns), np.array(thick), np.array(otops), np.array(obots)))
    # global outline stroke width from outlined (two-run) columns of long bands
    allw = [r_[8][r_[7] >= 2] for r_ in raw if (r_[2] > 3 * ppf)]  # noqa
    allw = np.concatenate(allw) if allw else np.zeros(0)
    gw = float(np.median(allw)) if len(allw) >= 20 else max(2.0, 0.12 * ppf)
    for (x0, y0, w, h, xs, tops, bots, nruns, thick, otops, obots) in raw:
        two = nruns >= 2
        # solid-poche columns (one dark run): outer edges -> outline centres
        wline = gw
        one = ~two
        tops = tops.copy()
        bots = bots.copy()
        tops[one] += wline / 2.0 - 0.5
        bots[one] -= wline / 2.0 - 0.5
        # split where centre / thickness change (collinear walls of other types)
        cc = (tops + bots) / 2.0
        dd = bots - tops
        k = 5
        if len(cc) >= k:
            pad = k // 2
            ccm = np.array([np.median(cc[max(0, i - pad):i + pad + 1]) for i in range(len(cc))])
            ddm = np.array([np.median(dd[max(0, i - pad):i + pad + 1]) for i in range(len(dd))])
        else:
            ccm, ddm = cc, dd
        cuts = [0]
        for i in range(1, len(xs)):
            if xs[i] - xs[i - 1] > 3 * step or abs(ccm[i] - ccm[i - 1]) > 1.6 or abs(ddm[i] - ddm[i - 1]) > 2.6:
                cuts.append(i)
        cuts.append(len(xs))
        for a_, b_ in zip(cuts[:-1], cuts[1:]):
            if b_ - a_ < 3:
                continue
            sx = xs[a_:b_]
            bands.append(dict(x0=int(sx[0]), x1=int(min(sx[-1] + step - 1, x0 + w - 1)), y0=int(y0), y1=int(y0 + h - 1),
                              xs=sx, tops=tops[a_:b_], bots=bots[a_:b_], nruns=nruns[a_:b_],
                              thick=thick[a_:b_], otops=otops[a_:b_], obots=obots[a_:b_],
                              horizontal=horizontal))
    return bands, HM


def band_summary(b, wline=None):
    """Centre and centre-to-centre thickness of a band.  The centre comes from
    the outer edges (unbiased by the poche); the thickness is the outer width
    minus the outline stroke width when that is known, else the outline-centre
    estimate."""
    c = float(np.median((b['tops'] + b['bots']) / 2.0))
    if wline is None:
        d = float(np.median(b['bots'] - b['tops']))
    else:
        d = float(np.median(b['obots'] - b['otops'])) - wline
    return c, d


def band_outer(b):
    return float(np.median(b['obots'] - b['otops']))


def strip_features(kn, orient, c, d, a0, a1, inset=None):
    """Fill features of a wall strip in band-relative coordinates (u along, v across).

    orient 'h': strip spans x in [a0,a1], y in [c-d/2+inset, c+d/2-inset];
    'v' strips are transposed so that u runs along the wall.  Diagonal hatches
    keep their orientation under transposition."""
    if inset is None:
        inset = max(2.5, 0.22 * d)
    lo = int(np.ceil(c - d / 2.0 + inset))
    hi = int(np.floor(c + d / 2.0 - inset))
    if hi < lo:
        lo = hi = int(round(c))
    a0 = int(round(a0))
    a1 = int(round(a1))
    if a1 - a0 < 4:
        return None
    if orient == 'h':
        strip = kn[lo:hi + 1, a0:a1 + 1]
    else:
        strip = kn[a0:a1 + 1, lo:hi + 1].T
    if strip.size == 0:
        return None
    s = np.ascontiguousarray(strip, dtype=np.float32)
    gu = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gv = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    if s.shape[0] > 2:
        gu = gu[1:-1]
        gv = gv[1:-1]
    p = gu * gv
    colm = s.mean(axis=0)
    f = dict(mean=float(s.mean()),
             d1=float(np.mean(np.clip(p, 0, None))), d2=float(np.mean(np.clip(-p, 0, None))),
             ex=float(np.mean(gu * gu)), ey=float(np.mean(gv * gv)),
             std=float(colm.std()))
    return f


FEAT_KEYS = ('mean', 'd1', 'd2', 'ex', 'ey', 'std')


def feat_vec(f):
    return np.array([f[k] for k in FEAT_KEYS], float)


def feat_dist(f, g):
    """Distance between two wall-fill feature dicts.

    Uses the mean ink level, hatch energy along the wall (the across-wall
    gradient is dominated by the outlines and depends on wall thickness) and the
    diagonal orientation balance, weighted by how hatched both strips are."""
    ea = max(f['d1'] + f['d2'] + f['ex'], 0.0)
    eb = max(g['d1'] + g['d2'] + g['ex'], 0.0)
    sa, sb = np.sqrt(ea), np.sqrt(eb)
    oa = (f['d1'] - f['d2']) / (f['d1'] + f['d2'] + 1e-4)
    ob = (g['d1'] - g['d2']) / (g['d1'] + g['d2'] + 1e-4)
    w_o = 3.0 * float(np.clip(min(sa, sb) / 0.03, 0.0, 1.0))
    dm = abs(f['mean'] - g['mean']) / 0.08
    de = abs(sa - sb) / 0.02
    do = abs(oa - ob) * w_o
    return float(np.sqrt(dm * dm + de * de + do * do))


def legend_bar_features(kn, bars):
    out = []
    for (x, y, w, h) in bars:
        prof = kn[max(0, y - 3):y + h + 3, x + int(0.1 * w):x + int(0.9 * w)].mean(axis=1)
        m = len(prof) // 2

        def pk(p):
            i = int(np.argmax(p))
            if 0 < i < len(p) - 1:
                a, b_, c = p[i - 1], p[i], p[i + 1]
                return i + (a - c) / (2 * (a - 2 * b_ + c) + 1e-9)
            return float(i)
        t = max(0, y - 3) + pk(prof[:m])
        bo = max(0, y - 3) + m + pk(prof[m:])
        f = strip_features(kn, 'h', (t + bo) / 2.0, bo - t, x + 0.1 * w, x + 0.9 * w)
        f['d'] = float(bo - t)
        out.append(f)
    return out


def faces_continuous(kn, orient, c, d, u0, u1, thr=0.35, frac=0.75):
    """Do both wall faces carry a dark line all along [u0, u1]?  True for a
    wall whose band detection broke, or a door opening (threshold lines);
    false for open floor."""
    u0, u1 = int(round(min(u0, u1))), int(round(max(u0, u1)))
    if u1 - u0 < 2:
        return True
    img = kn if orient == 'h' else kn.T
    H = img.shape[0]
    ok = []
    for v in (c - d / 2.0, c + d / 2.0):
        r0, r1 = max(0, int(round(v)) - 2), min(H, int(round(v)) + 3)
        prof = img[r0:r1, u0:u1].max(axis=0)
        ok.append(float(np.mean(prof > thr)) >= frac)
    return all(ok)


def classify_windows(kn, ppf, orient, c, d, a0, a1, barf, win=1.0, skip=0.3, allowed=(1, 2, 3)):
    """Label 1-ft windows along a band with the nearest legend bar (1=P3/EXT, 2=P2, 3=P1).
    Returns labels among `allowed`, their distances, and the best distance over all bars."""
    labs, dists, anyd = [], [], []
    a = a0 + skip * ppf
    while a + win * ppf <= a1 - skip * ppf + 1e-6:
        f = strip_features(kn, orient, c, d, a, a + win * ppf)
        if f is not None:
            ds = [feat_dist(f, g) for g in barf]
            j = min(allowed, key=lambda q: ds[q])
            labs.append(j)
            dists.append(float(ds[j]))
            anyd.append(float(min(ds[1:])))
        a += win * ppf
    return labs, dists, anyd


def assemble(kn, ppf, legend, hb, vb, doors=None, debug=False):
    """Build the wall model.  Returns dict with EXT polygon, interior segments, lengths."""
    barf = legend_bar_features(kn, legend['bars'])
    legx = min(q[0] for q in legend['swatches'])
    cands = []
    for orient, bl in (('h', hb), ('v', vb)):
        for b in bl:
            c, d = band_summary(b)
            do = band_outer(b)
            L = (b['x1'] - b['x0']) / ppf
            if not (0.3 * ppf <= d <= 0.85 * ppf) or L < 0.75:
                continue
            if orient == 'h' and b['x1'] > legx - 2 * ppf:
                continue
            if orient == 'v' and c > legx - 2 * ppf:
                continue
            cands.append(dict(o=orient, c=c, d=d, do=do, a0=float(b['x0']), a1=float(b['x1'])))
    # --- building extents from long bands (rectangular building assumed here)
    longh = [w for w in cands if w['o'] == 'h' and (w['a1'] - w['a0']) > 10 * ppf]
    longv = [w for w in cands if w['o'] == 'v' and (w['a1'] - w['a0']) > 10 * ppf]
    top = min(longh, key=lambda w: w['c'])['c']
    bot = max(longh, key=lambda w: w['c'])['c']
    lef = min(longv, key=lambda w: w['c'])['c']
    rig = max(longv, key=lambda w: w['c'])['c']
    tol = 0.35 * ppf
    ext, inner = [], []
    for w in cands:
        if w['o'] == 'h':
            inside = (top - tol <= w['c'] <= bot + tol) and w['a1'] >= lef - tol and w['a0'] <= rig + tol
            on_ext = abs(w['c'] - top) < tol or abs(w['c'] - bot) < tol
        else:
            inside = (lef - tol <= w['c'] <= rig + tol) and w['a1'] >= top - tol and w['a0'] <= bot + tol
            on_ext = abs(w['c'] - lef) < tol or abs(w['c'] - rig) < tol
        if not inside:
            continue
        (ext if on_ext else inner).append(w)
    # outline stroke width from the exterior walls (8" CMU): outer width - 8"
    t_ext = 8.0 / 12.0 * ppf
    if ext:
        wl = float(np.median([w['do'] for w in ext])) - t_ext
        if not (0.5 <= wl <= 0.35 * ppf):
            wl = float(np.median([w['do'] - w['d'] for w in ext]))
    else:
        wl = max(1.0, 0.12 * ppf)
    dext = float(np.median([w['d'] for w in ext])) if ext else t_ext
    # --- classify interior candidates
    segs = []
    for w in inner:
        thin = w['d'] < 0.75 * dext or w['d'] < 0.48 * ppf
        allowed = (3,) if thin else (1, 2)
        labs, dists, anyd = classify_windows(kn, ppf, w['o'], w['c'], w['d'], w['a0'], w['a1'], barf,
                                             allowed=allowed)
        L = (w['a1'] - w['a0']) / ppf
        if not labs:
            # very short piece; keep only if thin/thick consistent and adjacent later
            typ = 3 if thin else None
            conf = 0.0
        else:
            good = [l for l, dd in zip(labs, anyd) if dd < 2.5]
            if thin:
                typ = 3
            else:
                # weighted vote: windows that match better count more
                sc = np.zeros(4)
                for l, dd in zip(labs, dists):
                    sc[l] += 1.0 / (0.5 + dd)
                typ = int(max(allowed, key=lambda q: sc[q]))
            conf = len(good) / float(len(labs))
        w.update(typ=typ, labs=labs, dists=dists, conf=conf)
        if typ is None:
            continue
        # reject fixture-like bands: few well-matching windows
        if labs and conf < 0.3 and L < 3.0:
            w['reject'] = True
            if debug:
                print('reject', w['o'], round(w['c'], 1), round(w['a0']), round(w['a1']), labs, np.round(dists, 2))
            continue
        segs.append(w)
    # --- merge collinear pieces of the same type across small gaps
    perp = segs + ext

    def crosses_wall(o, c, u0, u1):
        # is there a perpendicular wall meeting this line inside (u0, u1)?
        for v in perp:
            if v['o'] == o:
                continue
            if u0 + 0.3 * ppf < v['c'] < u1 - 0.3 * ppf and v['a0'] - 0.5 * ppf <= c <= v['a1'] + 0.5 * ppf:
                return True
        return False
    merged = []
    for o in ('h', 'v'):
        for typ in (1, 2, 3):
            ss = sorted([w for w in segs if w['o'] == o and w['typ'] == typ], key=lambda w: (w['c'], w['a0']))
            used = [False] * len(ss)
            for i in range(len(ss)):
                if used[i]:
                    continue
                grp = [ss[i]]
                used[i] = True
                for j in range(i + 1, len(ss)):
                    if used[j]:
                        continue
                    if abs(ss[j]['c'] - ss[i]['c']) < 0.2 * ppf:
                        grp.append(ss[j])
                        used[j] = True
                grp.sort(key=lambda w: w['a0'])
                cur = dict(grp[0])
                for w in grp[1:]:
                    gap = w['a0'] - cur['a1']
                    cm, dm = (cur['c'] + w['c']) / 2.0, (cur['d'] + w['d']) / 2.0
                    poche_on = False
                    if 1.0 * ppf < gap <= 4.0 * ppf:
                        fsg = strip_features(kn, o, cm, dm, cur['a1'] + 2, w['a0'] - 2)
                        poche_on = fsg is not None and feat_dist(fsg, barf[typ]) < 1.5
                    if gap <= 1.0 * ppf or poche_on or (gap <= 8.0 * ppf and not crosses_wall(o, cm, cur['a1'], w['a0'])
                                                        and faces_continuous(kn, o, cm, dm, cur['a1'], w['a0'])):
                        cur['a1'] = max(cur['a1'], w['a1'])
                        cur['d'] = (cur['d'] + w['d']) / 2.0
                    else:
                        merged.append(cur)
                        cur = dict(w)
                merged.append(cur)
    # --- extend ends to the centreline of the wall they meet
    allw = merged + [dict(w, typ=0) for w in ext]

    def meets(end, w):
        best = None
        for v in allw:
            if v['o'] == w['o']:
                continue
            half = v['d'] / 2.0
            jt = 0.35 * ppf
            if not (v['c'] - half - jt <= end <= v['c'] + half + jt):
                continue
            if not (v['a0'] - w['d'] / 2.0 - jt <= w['c'] <= v['a1'] + w['d'] / 2.0 + jt):
                continue
            dd = abs(end - v['c'])
            if best is None or dd < best[0]:
                best = (dd, v['c'])
        return best

    for w in merged:
        m0 = meets(w['a0'], w)
        m1 = meets(w['a1'], w)
        w['e0'] = m0[1] if m0 else w['a0']
        w['e1'] = m1[1] if m1 else w['a1']
        w['j0'] = m0 is not None
        w['j1'] = m1 is not None
    # free ends: extend up to 6 ft to a perpendicular wall when the faces run on
    def reach(w, end, sign):
        best = None
        for v in allw:
            if v['o'] == w['o']:
                continue
            if not (v['a0'] - w['d'] <= w['c'] <= v['a1'] + w['d']):
                continue
            dist = (v['c'] - end) * sign
            if 0 < dist <= 6.0 * ppf and (best is None or dist < best[0]):
                best = (dist, v)
        if best is None:
            return None
        v = best[1]
        face = v['c'] - sign * v['d'] / 2.0
        if faces_continuous(kn, w['o'], w['c'], w['d'], end, face):
            return v['c']
        return None
    for w in merged:
        if not w['j0']:
            e = reach(w, w['a0'], -1)
            if e is not None:
                w['e0'], w['j0'] = e, True
        if not w['j1']:
            e = reach(w, w['a1'], +1)
            if e is not None:
                w['e1'], w['j1'] = e, True
    # isolated pieces (no junction at either end) are usually fixture edges
    # (cooler-door rows, casework): keep only long, clearly wall-like ones
    merged = [w for w in merged if w['j0'] or w['j1'] or
              ((w['e1'] - w['e0']) >= 15.0 * ppf and w.get('conf', 1.0) >= 0.6)]
    # very short pieces hanging off one wall (door leaves, jamb returns) are not walls
    merged = [w for w in merged if (w['j0'] and w['j1']) or (w['e1'] - w['e0']) >= 3.0 * ppf]
    lengths = {'P3': 0.0, 'P2': 0.0, 'P1': 0.0}
    names = {1: 'P3', 2: 'P2', 3: 'P1'}
    for w in merged:
        lengths[names[w['typ']]] += (w['e1'] - w['e0']) / ppf
    lengths['EXT'] = 2 * ((bot - top) + (rig - lef)) / ppf
    return dict(ext_box=(top, bot, lef, rig), dext=dext, ext=ext, segs=merged, lengths=lengths, barf=barf,
                wline=wl)


def detect_bands_multi(kn, ppf, horizontal=True, taus=(0.45, 0.6)):
    """Union of band detections at several outline thresholds.  A low threshold
    keeps faint outlines; a higher one separates outlines from dark floor fills
    that would otherwise merge with them.  Later bands are added only where no
    collinear band was found before."""
    out = []
    hm0 = None
    for t in taus:
        bands, HM = detect_bands(kn, ppf, tau=t, horizontal=horizontal)
        if hm0 is None:
            hm0 = HM
        for b in bands:
            c, d = band_summary(b)
            dup = False
            for o_ in out:
                c2, d2 = band_summary(o_)
                if not (0.3 * ppf <= d2 <= 0.85 * ppf):
                    continue
                if abs(c - c2) < max(4.0, 0.5 * max(d, d2)) and min(b['x1'], o_['x1']) - max(b['x0'], o_['x0']) > 0:
                    dup = True
                    break
            if not dup:
                out.append(b)
    return out, hm0
