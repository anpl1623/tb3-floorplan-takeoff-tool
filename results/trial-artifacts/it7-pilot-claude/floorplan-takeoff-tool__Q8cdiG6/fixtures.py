"""Fixture detection (chain CAD blocks) and counting."""
import numpy as np
import cv2

from blocks import BLOCKS, response_map, peaks

DEFAULT_THR = 0.78


def variant_mapper(name, ppf, k, flip, vshape):
    """Return function mapping canonical block coords (ft) to offsets from the variant's centre (px)."""
    b = BLOCKS[name]
    w, h = b['size']
    ml, mt, mr, mb = b.get('margin', (0.35,) * 4)
    W = int(np.ceil((w + ml + mr) * ppf))
    H = int(np.ceil((h + mt + mb) * ppf))
    vh, vw = vshape

    def f(u, v):
        x = (u + ml) * ppf
        y = (v + mt) * ppf
        if flip:
            x = W - x
        cw, ch = W, H
        for _ in range(k % 4):
            # np.rot90 (counter-clockwise): (x, y) -> (y, cw - x); new size (ch, cw)
            x, y = y, cw - x
            cw, ch = ch, cw
        return x - vw / 2.0, y - vh / 2.0
    return f


def footprint(det, ppf, rect_ft):
    """Image-space axis-aligned rectangle (x0,y0,x1,y1) of canonical rect (u0,v0,u1,v1) in ft."""
    f = variant_mapper(det['name'], ppf, det['k'], det['flip'], (det['h'], det['w']))
    u0, v0, u1, v1 = rect_ft
    pts = [f(u0, v0), f(u1, v1)]
    xs = [det['x'] + p[0] for p in pts]
    ys = [det['y'] + p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def gap_threshold(scores, default=DEFAULT_THR, lo=0.66, hi=0.84, min_gap=0.07):
    s = sorted([x for x in scores if x > 0.5], reverse=True)
    best = None
    for i in range(len(s) - 1):
        if s[i] >= lo and s[i + 1] <= hi:
            g = s[i] - s[i + 1]
            if g >= min_gap and (best is None or g > best[0]):
                best = (g, 0.5 * (s[i] + s[i + 1]))
    if best is not None:
        return best[1]
    return default


def detect_all(sub, interior, ppf):
    dets = {}
    for name in ('checkout', 'sco', 'produce', 'endcap', 'cooler', 'gond4', 'gond3'):
        resp, arg, vs = response_map(sub, ppf, name)
        resp[~interior] = -1
        b = BLOCKS[name]
        lst = []
        for sc, x, y in peaks(resp, 0.5, 0.4 * max(b['size']) * ppf):
            k, flip, v = vs[arg[y, x]]
            lst.append(dict(score=sc, x=x, y=y, k=k, flip=flip, w=v.shape[1], h=v.shape[0], name=name))
        dets[name] = lst
    return dets


def _overlap(r1, r2):
    ix = max(0.0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
    iy = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    return ix * iy


def count_fixtures(dets, ppf):
    out = {}
    accepted = {}
    for name in ('checkout', 'sco', 'produce', 'endcap', 'cooler'):
        lst = dets[name]
        thr = gap_threshold([d['score'] for d in lst])
        acc = [d for d in lst if d['score'] >= thr]
        accepted[name] = acc
    # gondola sections: joint 4ft / 3ft decision
    cand = []
    for name, L in (('gond4', 4.0), ('gond3', 3.0)):
        for d in dets[name]:
            if d['score'] < 0.6:
                continue
            r = footprint(d, ppf, (0, 0, 4.0, L))
            cand.append((d['score'], name, d, r))
    cand.sort(key=lambda c: -c[0])
    chosen = []
    for sc, name, d, r in cand:
        area = (r[2] - r[0]) * (r[3] - r[1])
        if any(_overlap(r, c[3]) > 0.25 * area for c in chosen):
            continue
        chosen.append((sc, name, d, r))
    thr = gap_threshold([c[0] for c in chosen])
    g4 = [c[2] for c in chosen if c[1] == 'gond4' and c[0] >= thr]
    g3 = [c[2] for c in chosen if c[1] == 'gond3' and c[0] >= thr]
    accepted['gond4'] = g4
    accepted['gond3'] = g3
    out['gondola_sections_4ft'] = len(g4)
    out['gondola_sections_3ft'] = len(g3)
    out['endcaps'] = len(accepted['endcap'])
    out['checkouts'] = len(accepted['checkout'])
    out['self_checkouts'] = len(accepted['sco'])
    out['cooler_doors'] = len(accepted['cooler'])
    out['produce_tables'] = len(accepted['produce'])
    return out, accepted


FOOT = {
    'gond4': (0, 0, 4.0, 4.0),
    'gond3': (0, 0, 4.0, 3.0),
    'endcap': (0, 0, 4.0, 1.5),
    'checkout': (0, 0, 2.5, 12.0),
    'sco': (0, 0, 3.0, 3.0),
    'cooler': (-3.0, 0, 0.6, 2.3),   # include the (white) case behind the doors
    'produce': (0, 0, 5.0, 4.0),
}


def fixture_mask(shape, accepted, ppf, pad=3):
    M = np.zeros(shape, bool)
    for name, lst in accepted.items():
        for d in lst:
            x0, y0, x1, y1 = footprint(d, ppf, FOOT[name])
            M[max(0, int(y0) - pad):int(y1) + pad + 1, max(0, int(x0) - pad):int(x1) + pad + 1] = True
    return M
