"""Door detection: swing (single / pair), overhead coiling and automatic sliding doors."""
import numpy as np
import cv2


def _gather(M, xs, ys):
    H, W = M.shape
    xi = np.clip(np.round(xs).astype(np.int64), 0, W - 1)
    yi = np.clip(np.round(ys).astype(np.int64), 0, H - 1)
    return M[yi, xi]


def swing_candidates(ink, fr, lines, radii_ft=(2.5, 3.0, 3.5, 4.0), thr=0.18):
    """Scan for quarter-circle door swings hinged on wall faces."""
    M = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    out = []
    thetas = np.deg2rad(np.linspace(12, 78, 23))
    for L in lines.values():
        o = L.o
        pa = fr.pp(o)                       # px per ft along
        pc = fr.pp('v' if o == 'h' else 'h')  # px per ft across
        for (s, e) in L.runs:
            a0 = fr.along_px(o, s)
            a1 = fr.along_px(o, e)
            hs = np.arange(int(a0), int(a1) + 1, dtype=np.float64)
            if len(hs) < 5:
                continue
            for side in (-1, 1):
                face = fr.cross_px(o, L.c + side * L.t / 2.0)
                for R_ft in radii_ft:
                    R = R_ft * pa
                    for d in (-1, 1):
                        acc = np.zeros(len(hs))
                        acc_in = np.zeros(len(hs))
                        acc_out = np.zeros(len(hs))
                        for th in thetas:
                            for rr, accx in ((R, acc), (R - 5, acc_in), (R + 5, acc_out)):
                                al = hs + d * rr * np.sin(th)
                                cr = face + side * rr * np.cos(th) * (pc / pa)
                                if o == 'h':
                                    v = _gather(M, al, np.full_like(al, 0) + cr)
                                else:
                                    v = _gather(M, np.full_like(al, 0) + cr, al)
                                accx += v
                        n = len(thetas)
                        arc = acc / n
                        bg = np.maximum(acc_in, acc_out) / n
                        score = arc - bg
                        # local maxima above threshold
                        idx = np.where(score > thr)[0]
                        for i in idx:
                            lo = max(0, i - 3)
                            hi = min(len(score), i + 4)
                            if score[i] < score[lo:hi].max():
                                continue
                            out.append({'line': L, 'h_px': hs[i], 'h_ft': fr.along(o, hs[i]),
                                        'side': side, 'dir': d, 'R': R_ft, 'score': float(score[i]),
                                        'arc': float(arc[i]), 'bg': float(bg[i])})
    return out


def nms_swings(cands, fr):
    """Keep the best hypothesis per door leaf (same line/side, overlapping arcs)."""
    cands = sorted(cands, key=lambda c: -c['score'])
    keep = []
    for c in cands:
        dup = False
        for k in keep:
            if k['line'] is c['line'] and k['side'] == c['side']:
                # arc footprint along the line: [h, h + d*R]
                a1 = sorted([c['h_ft'], c['h_ft'] + c['dir'] * c['R']])
                a2 = sorted([k['h_ft'], k['h_ft'] + k['dir'] * k['R']])
                ov = min(a1[1], a2[1]) - max(a1[0], a2[0])
                if ov > 0.4 * min(a1[1] - a1[0], a2[1] - a2[0]):
                    dup = True
                    break
        if not dup:
            keep.append(c)
    return keep


def leaf_score(ink, fr, c):
    """Contrast of the straight door leaf drawn from the hinge perpendicular to the wall."""
    L = c['line']
    o = L.o
    pa = fr.pp(o)
    pc = fr.pp('v' if o == 'h' else 'h')
    R = c['R'] * pa
    face = fr.cross_px(o, L.c + c['side'] * L.t / 2.0)
    rs = np.linspace(0.15, 0.85, 15) * R * (pc / pa)
    best = -1.0
    M = c.get('_M')
    for off in (0.0, 1.5, 3.0):
        al = c['h_px'] + c['dir'] * off
        vals = []
        bgs = []
        for r in rs:
            cr = face + c['side'] * r
            if o == 'h':
                xs, ys = np.array([al]), np.array([cr])
                bx1, by1 = np.array([al + c['dir'] * 9]), ys
                bx2, by2 = np.array([al - c['dir'] * 7]), ys
            else:
                xs, ys = np.array([cr]), np.array([al])
                bx1, by1 = xs, np.array([al + c['dir'] * 9])
                bx2, by2 = xs, np.array([al - c['dir'] * 7])
            vals.append(_gather(ink, xs, ys)[0])
            bgs.append(max(_gather(ink, bx1, by1)[0], _gather(ink, bx2, by2)[0]))
        sc = float(np.mean(vals) - np.mean(bgs))
        best = max(best, sc)
    return best


def _band(ink, fr, L, a_px0, a_px1, inset=1.5):
    o = L.o
    f1 = fr.cross_px(o, L.c - L.t / 2)
    f2 = fr.cross_px(o, L.c + L.t / 2)
    r0 = int(np.ceil(f1 + inset))
    r1 = int(np.floor(f2 - inset)) + 1
    a0 = int(round(a_px0))
    a1 = int(round(a_px1))
    if r1 <= r0 or a1 <= a0:
        return None
    if o == 'h':
        return ink[r0:r1, a0:a1]
    return ink[a0:a1, r0:r1].T


def _piece_kind(L, f):
    for p0, p1, kind, tpx, votes in L.pieces:
        if p0 - 1e-6 <= f <= p1 + 1e-6:
            return kind
    return None


def _empty_dist(pm, k):
    T = pm.strips[k]
    if T is None:
        return 0.02
    return pm.dist(np.zeros_like(T), k)


def find_openings(ink, fr, lines, pm, min_w=1.6, max_w=18.0):
    """Openings (door gaps) in walls, found as pairs of jamb lines whose interior
    does not look like the wall's poche."""
    from walls import interior_strip
    out = []
    kind_idx = {'EXT': [0, 1], 'CMU': [0, 1], 'P3': [0, 1], 'P2': [2], 'P1': [3]}
    for L in lines.values():
        o = L.o
        for (s, e) in L.runs:
            a0 = fr.along_px(o, s)
            a1 = fr.along_px(o, e)
            B = _band(ink, fr, L, a0, a1)
            if B is None or B.shape[1] < 10:
                continue
            J = cv2.blur(B, (3, 1)).mean(0)
            base = float(np.median(J))
            jt = max(0.42, base + 0.28)
            pk = []
            for i in range(1, len(J) - 1):
                if J[i] >= jt and J[i] >= J[i - 1] and J[i] >= J[i + 1]:
                    if pk and i - pk[-1] < 4:
                        if J[i] > J[pk[-1]]:
                            pk[-1] = i
                        continue
                    pk.append(i)
            jambs = [fr.along(o, a0 + i) for i in pk]
            for (cp, tp, P) in L.crossings:
                for f in (cp - tp / 2, cp + tp / 2):
                    if s + 0.2 < f < e - 0.2:
                        jambs.append(f)
            jambs.sort()
            for j1, j2 in zip(jambs[:-1], jambs[1:]):
                w = j2 - j1
                if w < min_w or w > max_w:
                    continue
                if any(abs((j1 + j2) / 2 - cp) < tp / 2 + 0.1 for (cp, tp, P) in L.crossings):
                    continue
                kind = 'EXT' if L.ext else _piece_kind(L, 0.5 * (j1 + j2))
                ks = kind_idx.get(kind, [0, 1, 2, 3])
                q0 = fr.along_px(o, j1) + 4
                q1 = fr.along_px(o, j2) - 4
                if q1 - q0 < 8:
                    continue
                S = interior_strip(ink, o, fr.cross_px(o, L.c - L.t / 2), fr.cross_px(o, L.c + L.t / 2), q0, q1)
                if S is None:
                    continue
                d_wall = min(pm.dist(S, k) for k in ks)
                d_empty = max(min(_empty_dist(pm, k) for k in ks), 1e-4)
                r = d_wall / d_empty
                tmean = float(np.mean([pm.mean[k] for k in ks]))
                h = min([S.shape[0]] + [pm.strips[k].shape[0] for k in ks if pm.strips[k] is not None])
                o0 = (S.shape[0] - h) // 2
                mean_in = float(S[o0:o0 + h].mean())
                is_open = r > 0.4 or (mean_in < 0.6 * tmean and r > 0.15)
                out.append({'line': L, 'j1': j1, 'j2': j2, 'w': w, 'kind': kind, 'd_wall': d_wall,
                            'r': r, 'mean_in': mean_in, 'tmean': tmean, 'open': is_open})
    return out


def merge_openings(ops, gap=1.2):
    """Merge open intervals on the same line separated by tiny gaps (sliding door panels)."""
    ops = [o for o in ops if o['open']]
    ops.sort(key=lambda q: (id(q['line']), q['j1']))
    out = []
    for op in ops:
        if out and out[-1]['line'] is op['line'] and op['j1'] - out[-1]['j2'] <= gap:
            out[-1] = dict(out[-1])
            out[-1]['j2'] = op['j2']
            out[-1]['w'] = out[-1]['j2'] - out[-1]['j1']
            out[-1]['merged'] = True
        else:
            out.append(dict(op))
    return out


def detect_doors(ink, fr, lines, pm):
    cands = swing_candidates(ink, fr, lines, thr=0.12)
    sw = nms_swings(cands, fr)
    good = []
    weak = []
    for c in sw:
        c['leaf'] = leaf_score(ink, fr, c)
        if c['score'] >= 0.2 and c['leaf'] >= 0.25:
            good.append(c)
        elif c['score'] >= 0.12 and c['leaf'] >= 0.15:
            weak.append(c)
    sliding, overhead = band_door_scan(ink, fr, lines)

    def blocked(Lx, a, b):
        """Is [a,b] on line Lx already a sliding / overhead door?"""
        for (L2, s2, e2) in sliding + overhead:
            if L2 is Lx and min(b, e2) - max(a, s2) > 0.3 * (b - a):
                return True
        return False
    good = [c for c in good if (c['score'] >= 0.3 and c['leaf'] >= 0.4) or
            not blocked(c['line'], min(c['h_ft'], c['h_ft'] + c['dir'] * c['R']),
                        max(c['h_ft'], c['h_ft'] + c['dir'] * c['R']))]
    # pairs: two leaves on the same wall side, opposite directions, meeting in the middle
    used = set()
    pairs = []
    for i, a in enumerate(good):
        for j, b in enumerate(good):
            if i >= j or i in used or j in used:
                continue
            if a['line'] is not b['line'] or a['side'] != b['side']:
                continue
            lo, hi = (a, b) if a['h_ft'] < b['h_ft'] else (b, a)
            if lo['dir'] != 1 or hi['dir'] != -1:
                continue
            span = hi['h_ft'] - lo['h_ft']
            if abs(span - (lo['R'] + hi['R'])) <= 0.6:
                pairs.append((lo, hi))
                used.add(i)
                used.add(j)
    singles = [c for k, c in enumerate(good) if k not in used]
    # openings not explained by any door symbol: a faint swing door?
    spans = []
    for c in singles:
        spans.append((c['line'], min(c['h_ft'], c['h_ft'] + c['dir'] * c['R']),
                      max(c['h_ft'], c['h_ft'] + c['dir'] * c['R'])))
    for lo, hi in pairs:
        spans.append((lo['line'], lo['h_ft'], hi['h_ft']))
    for (L2, s2, e2) in sliding + overhead:
        spans.append((L2, s2, e2))
    ops = merge_openings(find_openings(ink, fr, lines, pm))
    extra = []
    for op in ops:
        expl = False
        for (Lx, s0, s1) in spans:
            if Lx is op['line'] and min(s1, op['j2']) - max(s0, op['j1']) > 0.3 * (op['j2'] - op['j1']):
                expl = True
        if expl or not (2.4 <= op['w'] <= 4.6):
            continue
        # weak swing evidence hinged at one of the jambs
        for c in weak:
            if c['line'] is op['line'] and abs(c['R'] - op['w']) < 0.7 and \
                    (abs(c['h_ft'] - op['j1']) < 0.6 or abs(c['h_ft'] - op['j2']) < 0.6):
                extra.append(c)
                break
    res = {'single': len(singles) + len(extra), 'pair': len(pairs),
           'overhead_coiling': len(overhead), 'auto_sliding': len(sliding)}
    dbg = {'singles': singles, 'pairs': pairs, 'sliding': sliding, 'overhead': overhead,
           'openings': ops, 'extra': extra, 'others': []}
    return res, dbg


def _runs_of(mask):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def band_door_scan(ink, fr, lines):
    """Sliding and overhead coiling doors from the wall band profile.
    Sliding: two panels, one in each half of the wall thickness, side by side (top-bottom
    asymmetry changes sign).  Overhead coiling: a (dashed) line along the centre of the band
    over a wide opening, with the rest of the band empty."""
    sliding = []
    overhead = []
    for L in lines.values():
        o = L.o
        pa = fr.pp(o)
        for (s, e) in L.runs:
            a0 = fr.along_px(o, s)
            a1 = fr.along_px(o, e)
            B = _band(ink, fr, L, a0, a1, inset=1.5)
            if B is None or B.shape[1] < 3 * pa or B.shape[0] < 4:
                continue
            h = B.shape[0]
            m = max(1, int(round(0.15 * h)))
            k = max(3, int(0.25 * pa))
            Bs = cv2.blur(B, (k, 1))
            top = Bs[m:h // 2].mean(0)
            bot = Bs[(h + 1) // 2:h - m].mean(0)
            A = top - bot
            # ---- sliding: positive and negative asymmetry runs adjoining
            pos = _runs_of(A > 0.15)
            neg = _runs_of(A < -0.15)
            pos = [(i, j) for i, j in pos if 1.5 * pa <= j - i <= 5.0 * pa]
            neg = [(i, j) for i, j in neg if 1.5 * pa <= j - i <= 5.0 * pa]
            used = set()
            for (i, j) in pos:
                for (u, v) in neg:
                    if (u, v) in used:
                        continue
                    gap = max(u - j, i - v)
                    if gap <= 1.0 * pa:
                        a = min(i, u)
                        b = max(j, v)
                        sliding.append((L, s + a / pa, s + b / pa))
                        used.add((u, v))
                        break
            # ---- overhead coiling: centre line, everything else in the band empty
            k2 = max(3, int(1.0 * pa))
            Bl = cv2.blur(B, (k2, 1))
            oh = np.zeros(Bl.shape[1], bool)
            if L.t * 12.0 < 6.2:
                continue          # overhead doors are in thick (CMU / exterior) walls
            m2 = 2 if h >= 11 else 1
            ex = 2 if h >= 13 else 1
            for rc in range(max(1, h // 2 - 2), min(h - 1, h // 2 + 3)):
                C = Bl[rc]
                off = [r for r in range(m2, h - m2) if abs(r - rc) > ex]
                if len(off) < 2:
                    continue
                Qmed = np.median(Bl[off], axis=0)
                Qmax = Bl[off].max(0)
                oh |= ((C - Qmed) > 0.1) & (Qmed < np.maximum(0.06, 0.25 * C)) & \
                      (Qmax < np.maximum(0.15, 0.5 * C))
            oh = cv2.morphologyEx(oh.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE,
                                  np.ones((1, int(0.6 * pa)), np.uint8)).ravel() > 0
            for (i, j) in _runs_of(oh):
                if j - i >= 6.0 * pa:
                    overhead.append((L, s + i / pa, s + j / pa))
    return sliding, overhead
