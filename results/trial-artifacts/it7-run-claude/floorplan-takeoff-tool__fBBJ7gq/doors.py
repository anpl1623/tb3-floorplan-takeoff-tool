"""Door detection: openings between wall jambs, classified by their symbol."""
import numpy as np
import cv2

import walls as _walls


def _sample(img, xs, ys, r=1):
    """Max of img in a (2r+1)^2 neighbourhood at float points."""
    H, W = img.shape
    xi = np.clip(np.round(xs).astype(int), r, W - 1 - r)
    yi = np.clip(np.round(ys).astype(int), r, H - 1 - r)
    best = np.zeros(len(xi), np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            best = np.maximum(best, img[yi + dy, xi + dx])
    return best


def _mean_sample(img, xs, ys, r=1):
    H, W = img.shape
    xi = np.clip(np.round(xs).astype(int), r, W - 1 - r)
    yi = np.clip(np.round(ys).astype(int), r, H - 1 - r)
    acc = np.zeros(len(xi), np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            acc += img[yi + dy, xi + dx]
    return acc / (2 * r + 1) ** 2


def wall_list(res):
    top, bot, lef, rig = res['ext_box']
    de = res['dext']
    allw = [dict(w) for w in res['segs']]
    allw += [dict(o='h', c=top, d=de, e0=lef, e1=rig, typ=0, ext=True),
             dict(o='h', c=bot, d=de, e0=lef, e1=rig, typ=0, ext=True),
             dict(o='v', c=lef, d=de, e0=top, e1=bot, typ=0, ext=True),
             dict(o='v', c=rig, d=de, e0=top, e1=bot, typ=0, ext=True)]
    return allw


def find_jambs(kn, w, thr=0.5, prom=0.25):
    """Jamb lines crossing the full wall thickness: the minimum ink across the
    strip peaks, and is clearly above the level on at least one side."""
    o, c, d = w['o'], w['c'], w['d']
    a0, a1 = int(round(w['e0'])), int(round(w['e1']))
    img = kn if o == 'h' else kn.T
    lo, hi = int(round(c - d / 2 + 2.5)), int(round(c + d / 2 - 2.5))
    if hi < lo:
        lo = hi = int(round(c))
    strip = img[lo:hi + 1, a0:a1]
    J = strip.min(axis=0)
    J = np.maximum(J, np.r_[J[1:], 0])
    n = len(J)
    pk = np.nonzero((J > thr) & (J >= np.r_[0, J[:-1]]) & (J >= np.r_[J[1:], 0]))[0]
    peaks = []
    for p in pk:
        left = J[max(0, p - 10):max(0, p - 3)]
        right = J[min(n, p + 4):min(n, p + 11)]
        lv = float(np.median(left)) if len(left) else 0.0
        rv = float(np.median(right)) if len(right) else 0.0
        if J[p] - min(lv, rv) < prom:
            continue
        if peaks and p - peaks[-1] < 4:
            continue
        peaks.append(int(p))
    return [p + a0 for p in peaks], J


def to_xy(o, u, v):
    """wall coords (u along, v across) -> image (x, y)."""
    return (u, v) if o == 'h' else (v, u)


def arc_score(kn, o, cu, cv, R, sgn_u, sgn_v, n=36):
    """Fraction of arc angles where the ink peaks at radius R (radial profile test)."""
    th = np.deg2rad(np.linspace(12, 78, n))
    dr = np.arange(-6, 7)
    rr = R + dr[None, :]
    u = cu + sgn_u * rr * np.cos(th)[:, None]
    v = cv + sgn_v * rr * np.sin(th)[:, None]
    x, y = to_xy(o, u, v)
    H, W = kn.shape
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    prof = kn[yi, xi]                      # (n, 13)
    on = prof[:, 5:8].max(axis=1)
    base = np.median(prof, axis=1)
    return float(np.mean((on - base > 0.08) & (on > 0.15)))


def best_arc(kn, o, cu, cv, W, sgn_u, sgn_v):
    best = 0.0
    for f in (0.86, 0.9, 0.94, 0.98, 1.02):
        for dc in (-2, 0, 2):
            s = arc_score(kn, o, cu + sgn_u * dc, cv, W * f, sgn_u, sgn_v)
            best = max(best, s)
    return best


def classify_opening(kn, ppf, w, u1, u2):
    o, c, d = w['o'], w['c'], w['d']
    W = u2 - u1
    res = {}
    # swing tests
    singles = []
    pairs = []
    for s in (-1, 1):
        vf = c + s * d / 2.0
        a1 = best_arc(kn, o, u1, vf, W, +1, s)
        a2 = best_arc(kn, o, u2, vf, W, -1, s)
        singles.append(max(a1, a2))
        p1 = best_arc(kn, o, u1, vf, W / 2.0, +1, s)
        p2 = best_arc(kn, o, u2, vf, W / 2.0, -1, s)
        pairs.append(min(p1, p2))
    res['single'] = max(singles)
    res['pair'] = max(pairs)
    # inside the opening strip
    img = kn if o == 'h' else kn.T
    ua, ub = int(round(u1 + 3)), int(round(u2 - 3))
    lo, hi = int(round(c - d / 2 + 2.5)), int(round(c + d / 2 - 2.5))
    strip = img[lo:hi + 1, ua:ub] if ub > ua and hi >= lo else np.zeros((1, 1), np.float32)
    res['fill'] = float(strip.mean())
    # dashed centre line: profile along u at the centre rows
    cen = img[int(round(c)) - 1:int(round(c)) + 2, ua:ub].max(axis=0) if ub > ua else np.zeros(1)
    on = cen > 0.3
    res['center_cov'] = float(on.mean()) if len(on) else 0.0
    res['center_tr'] = int(np.sum(on[1:] != on[:-1])) if len(on) > 1 else 0
    # panels: rows near each face inside the strip
    res['W'] = W / ppf
    return res


def strip_halves(kn, w, ppf=None):
    """Per-position darkness of the two halves of a wall strip: the strip is
    smoothed along the wall, then the darkest row of each half is taken."""
    o, c, d = w['o'], w['c'], w['d']
    a0, a1 = int(round(w['e0'])), int(round(w['e1']))
    img = kn if o == 'h' else kn.T
    ins = max(2.0, 0.2 * d)
    lo, hi = int(round(c - d / 2 + ins)), int(round(c + d / 2 - ins))
    mid = int(round(c))
    strip = img[lo:hi + 1, a0:a1].astype(np.float32)
    if strip.size == 0 or strip.shape[1] < 3:
        z = np.zeros(max(0, a1 - a0), np.float32)
        return z, z, a0
    k = max(3, int(round(0.5 * ppf))) if ppf else 5
    strip = cv2.blur(strip, (k, 1))
    m = mid - lo
    U = strip[:max(1, m)].max(axis=0)
    L = strip[min(strip.shape[0] - 1, m + 1):].max(axis=0)
    return U, L, a0


def _runs(b):
    d = np.diff(np.r_[0, b.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def find_sliding(kn, ppf, w, tau=0.25):
    """Sliding doors: one half of the wall strip dark (a panel) followed by the
    other half dark (second panel), possibly overlapping."""
    U, L, a0 = strip_halves(kn, w, ppf)
    if len(U) < 10:
        return []
    Um, Lm = float(np.median(U)), float(np.median(L))
    delta = 0.2
    du = U - Um
    dl = L - Lm
    bu = (np.maximum(U - L, du - dl) > delta) & ((du > 0.15) | (dl < -0.15))
    bl = (np.maximum(L - U, dl - du) > delta) & ((dl > 0.15) | (du < -0.15))
    ru = [(a, b) for a, b in _runs(bu) if b - a >= 1.2 * ppf]
    rl = [(a, b) for a, b in _runs(bl) if b - a >= 1.2 * ppf]
    out = []
    for (a, b) in ru:
        for (c_, d_) in rl:
            # adjacent (gap = overlap region where both dark) <= 2 ft
            if 0 <= c_ - b <= 2.0 * ppf or 0 <= a - d_ <= 2.0 * ppf:
                s0 = min(a, c_)
                s1 = max(b, d_)
                if 3.0 * ppf <= s1 - s0 <= 16 * ppf:
                    out.append((s0 + a0, s1 + a0))
    # dedupe
    out.sort()
    res = []
    for s in out:
        if res and s[0] < res[-1][1]:
            continue
        res.append(s)
    return res


def overhead_test(kn, ppf, w, u1, u2):
    o, c, d = w['o'], w['c'], w['d']
    img = kn if o == 'h' else kn.T
    ua, ub = int(round(u1 + 4)), int(round(u2 - 4))
    if ub - ua < 10:
        return 0.0, 0.0, 0
    ci = int(round(c))
    C = img[ci - 1:ci + 2, ua:ub].max(axis=0)
    lo, hi = int(round(c - d / 2 + 2.5)), int(round(c + d / 2 - 2.5))
    off_rows = [r for r in range(lo, hi + 1) if abs(r - c) > 2.0]
    if not off_rows:
        return 0.0, 1.0, 0
    O = img[off_rows][:, ua:ub].mean()
    on = (C > 0.2) & (C - O > 0.12)
    cov = float(on.mean())
    tr = int(np.sum(on[1:] != on[:-1]))
    return cov, float(O), tr


def detect_doors(kn, ppf, res, debug=False):
    counts = dict(single=0, pair=0, overhead_coiling=0, auto_sliding=0)
    found = []
    # is the exterior poche hollow (white)?  then a clear wide opening is ambiguous
    bf = res.get('barf')
    ext_hollow = bool(bf) and bf[0]['mean'] < 0.1 and (bf[0]['d1'] + bf[0]['d2'] + bf[0]['ex']) < 0.002
    for w in wall_list(res):
        slides = find_sliding(kn, ppf, w)
        for (s0, s1) in slides:
            counts['auto_sliding'] += 1
            found.append(('auto_sliding', w['o'], w['c'], s0, s1))
        jb, J = find_jambs(kn, w)
        for a, b in zip(jb[:-1], jb[1:]):
            Wf = (b - a) / ppf
            if Wf < 1.5 or Wf > 25:
                continue
            if any(s0 - 3 <= a and b <= s1 + 3 for (s0, s1) in slides):
                continue
            r = classify_opening(kn, ppf, w, a, b)
            kind = None
            if r['pair'] >= 0.7 and r['pair'] >= r['single'] - 0.05:
                kind = 'pair'
            elif r['single'] >= 0.7:
                kind = 'single'
            else:
                cov, O, tr = overhead_test(kn, ppf, w, a, b)
                dpo = 0.0
                if bf:
                    pidx = 0 if w.get('ext') else int(w.get('typ', 1) or 1)
                    fs = _walls.strip_features(kn, w['o'], w['c'], w['d'], a + 4, b - 4)
                    dpo = _walls.feat_dist(fs, bf[pidx]) if fs is not None else 0.0
                r['poche_dist'] = dpo
                dashed = Wf >= 5.0 and 0.3 <= cov <= 0.97 and tr >= 0.6 * Wf and (O < 0.08 or (O < 0.15 and dpo > 1.2))
                if dashed and r['single'] < 0.5 and r['pair'] < 0.5:
                    kind = 'overhead_coiling'
                elif (w.get('ext') and not ext_hollow and 7.0 <= Wf <= 20.0 and O < 0.2 and r['fill'] < 0.22
                      and r['single'] < 0.5 and r['pair'] < 0.5 and bf and dpo > 2.0):
                    # wide clear opening in the exterior wall (receiving door): its
                    # strip must not look like the wall's own poche
                    kind = 'overhead_coiling'
                r.update(oh_cov=cov, oh_off=O, oh_tr=tr)
            if debug:
                print(w['o'], round(w['c'], 1), a, b, round(Wf, 2), kind, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
            if kind:
                counts[kind] += 1
                found.append((kind, w['o'], w['c'], a, b))
    return counts, found


def bridge_gaps(kn, ppf, wres):
    """Merge collinear same-type wall pieces whose gap holds a door (openings
    whose threshold lines were too faint to be part of the wall band)."""
    segs = wres['segs']
    changed = True
    nmerge = 0
    while changed:
        changed = False
        segs.sort(key=lambda w: (w['o'], w['typ'], round(w['c']), w['e0']))
        for i in range(len(segs)):
            a = segs[i]
            best = None
            for j in range(len(segs)):
                b = segs[j]
                if i == j or b['o'] != a['o'] or b['typ'] != a['typ']:
                    continue
                if abs(b['c'] - a['c']) > 0.2 * ppf:
                    continue
                gap = b['a0'] - a['a1'] if 'a0' in b else b['e0'] - a['e1']
                if 1.2 * ppf <= gap <= 14 * ppf and (best is None or gap < best[0]):
                    best = (gap, j)
            if best is None:
                continue
            b = segs[best[1]]
            u1, u2 = a['a1'], b['a0']
            w = dict(o=a['o'], c=(a['c'] + b['c']) / 2.0, d=(a['d'] + b['d']) / 2.0,
                     e0=u1 - 0.5 * ppf, e1=u2 + 0.5 * ppf, typ=a['typ'])
            r = classify_opening(kn, ppf, w, u1, u2)
            ok = r['single'] >= 0.7 or r['pair'] >= 0.7
            if not ok:
                ok = bool(find_sliding(kn, ppf, dict(w, e0=u1 - 2 * ppf, e1=u2 + 2 * ppf)))
            if not ok:
                cov, O, tr = overhead_test(kn, ppf, w, u1, u2)
                ok = 0.3 <= cov <= 0.95 and O < 0.08 and tr >= 0.6 * (u2 - u1) / ppf
            if ok:
                a['e1'] = b['e1']
                a['a1'] = b['a1']
                a['j1'] = b.get('j1', False)
                segs.pop(best[1])
                nmerge += 1
                changed = True
                break
    # drop collinear pieces that are (mostly) contained in another wall segment
    segs.sort(key=lambda w: -(w['e1'] - w['e0']))
    keep = []
    for w in segs:
        dup = False
        for k_ in keep:
            if k_['o'] != w['o'] or abs(k_['c'] - w['c']) > 0.3 * ppf:
                continue
            ov = min(k_['e1'], w['e1']) - max(k_['e0'], w['e0'])
            if ov > 0.8 * (w['e1'] - w['e0']):
                dup = True
                break
        if not dup:
            keep.append(w)
    segs[:] = keep
    lengths = dict(wres['lengths'])
    for k in ('P3', 'P2', 'P1'):
        lengths[k] = 0.0
    names = {1: 'P3', 2: 'P2', 3: 'P1'}
    for w in segs:
        lengths[names[w['typ']]] += (w['e1'] - w['e0']) / ppf
    wres['lengths'] = lengths
    return nmerge
