"""Synthetic templates of the chain's CAD blocks and template matching."""
import numpy as np
import cv2

# Block geometry in feet. Each block: size (w, h) and list of primitives.
# primitives: ('line', x0, y0, x1, y1), ('rect', x0, y0, x1, y1), ('circle', cx, cy, r)
BLOCKS = {
    'gond4': dict(size=(4.0, 4.0), prims=[('rect', 0, 0, 4, 4), ('line', 2, 0, 2, 4)]),
    'gond3': dict(size=(4.0, 3.0), prims=[('rect', 0, 0, 4, 3), ('line', 2, 0, 2, 3)]),
    'endcap': dict(size=(4.0, 1.5), prims=[('rect', 0, 0, 4, 1.5), ('line', 0, 1.5, 4, 0)]),
    'checkout': dict(size=(2.5, 12.0), prims=[('rect', 0, 0, 2.5, 9.0), ('rect', 0, 9.0, 2.5, 12.0),
                                               ('line', 0, 9.0, 2.5, 12.0), ('line', 2.5, 9.0, 0, 12.0),
                                               ('line', 1.25, 0.5, 1.25, 8.43)]),
    'sco': dict(size=(3.0, 3.0), prims=[('rect', 0, 0, 3, 3), ('circle', 1.5, 1.5, 0.78)]),
    'cooler': dict(size=(0.6, 2.3), prims=[('rect', 0, 0, 0.6, 2.3), ('line', 0, 2.3, 0.6, 0)],
                   margin=(0.6, 0.08, 0.06, 0.08)),
    'produce': dict(size=(5.0, 4.0), prims=[('rect', 0, 0, 5, 4), ('line', 0, 0, 5, 4), ('line', 0, 4, 5, 0)]),
    # single swing door, 3'-0": hinge at (3, 3) on wall face line y=3; leaf up, arc to (0, 3)
    'swing': dict(size=(3.0, 3.0), prims=[('rect', 2.83, 0, 3.0, 3.0), ('arc', 3.0, 3.0, 3.0, 180, 270)],
                  margin=(0.15, 0.15, 0.15, 0.1)),
    'pair': dict(size=(6.0, 3.0), prims=[('rect', 0, 0, 0.17, 3.0), ('arc', 0, 3.0, 3.0, 270, 360),
                                         ('rect', 5.83, 0, 6.0, 3.0), ('arc', 6.0, 3.0, 3.0, 180, 270)],
                 margin=(0.15, 0.15, 0.15, 0.1)),
}


def render(name, ppf, lw_px=2.2, margin_ft=0.35, ss=4):
    """Render block template as darkness image (float32), plus offset of block origin."""
    b = BLOCKS[name]
    w, h = b['size']
    ml, mt, mr, mb = b.get('margin', (margin_ft,) * 4)
    W = int(np.ceil((w + ml + mr) * ppf))
    H = int(np.ceil((h + mt + mb) * ppf))
    canvas = np.zeros((H * ss, W * ss), np.uint8)
    t = max(1, int(round(lw_px * ss)))

    def P(x, y):
        return (int(round((x + ml) * ppf * ss)), int(round((y + mt) * ppf * ss)))

    for p in b['prims']:
        if p[0] == 'line':
            cv2.line(canvas, P(p[1], p[2]), P(p[3], p[4]), 255, t, cv2.LINE_AA)
        elif p[0] == 'rect':
            cv2.rectangle(canvas, P(p[1], p[2]), P(p[3], p[4]), 255, t, cv2.LINE_AA)
        elif p[0] == 'circle':
            c = P(p[1], p[2])
            cv2.circle(canvas, c, int(round(p[3] * ppf * ss)), 255, t, cv2.LINE_AA)
        elif p[0] == 'arc':
            # ('arc', cx, cy, r, start_deg, end_deg)  angles in image coords (y down)
            c = P(p[1], p[2])
            rr = int(round(p[3] * ppf * ss))
            tt = max(1, int(round(0.7 * lw_px * ss)))
            cv2.ellipse(canvas, c, (rr, rr), 0, p[4], p[5], 255, tt, cv2.LINE_AA)
    img = cv2.resize(canvas.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    return img


def variants(img):
    """All 8 dihedral variants, deduplicated. Returns list of (k, flip, img)."""
    out = []
    seen = []
    for flip in (False, True):
        base = img[:, ::-1] if flip else img
        for k in range(4):
            v = np.ascontiguousarray(np.rot90(base, k))
            dup = False
            for s in seen:
                if s.shape == v.shape and np.abs(s - v).mean() < 0.01:
                    dup = True
                    break
            if dup:
                continue
            seen.append(v)
            out.append((k, flip, v))
    return out


def match(D, tmpl, thr, nms_r):
    """NCC template matching; returns list of (score, cx, cy) local maxima above thr."""
    if D.shape[0] < tmpl.shape[0] or D.shape[1] < tmpl.shape[1]:
        return []
    res = cv2.matchTemplate(D, tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
    k = 2 * int(nms_r) + 1
    mx = cv2.dilate(res, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((res >= mx) & (res > thr))
    th, tw = tmpl.shape
    return [(float(res[y, x]), x + tw / 2.0, y + th / 2.0) for y, x in zip(ys, xs)]


def response_map(D, ppf, name, tmpl_img=None):
    """Max NCC response over all dihedral variants, indexed by block centre.
    Returns (resp, argvariant, variants)."""
    img = render(name, ppf) if tmpl_img is None else tmpl_img
    H, W = D.shape
    resp = np.full((H, W), -1.0, np.float32)
    arg = np.zeros((H, W), np.int8)
    vs = variants(img)
    for vi, (k, flip, v) in enumerate(vs):
        th, tw = v.shape
        if H < th or W < tw:
            continue
        r = cv2.matchTemplate(D, v.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        oy, ox = th // 2, tw // 2
        tgt = resp[oy:oy + r.shape[0], ox:ox + r.shape[1]]
        better = r > tgt
        tgt[better] = r[better]
        arg[oy:oy + r.shape[0], ox:ox + r.shape[1]][better] = vi
    return resp, arg, vs


def peaks(resp, thr, r):
    k = 2 * int(max(1, r)) + 1
    mx = cv2.dilate(resp, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((resp >= mx) & (resp > thr))
    order = np.argsort(-resp[ys, xs])
    return [(float(resp[ys[i], xs[i]]), int(xs[i]), int(ys[i])) for i in order]


def detect(D, ppf, name, thr=0.45, nms_frac=0.4):
    resp, arg, vs = response_map(D, ppf, name)
    b = BLOCKS[name]
    r = nms_frac * min(b['size']) * ppf
    out = []
    for sc, x, y in peaks(resp, thr, r):
        k, flip, v = vs[arg[y, x]]
        out.append(dict(score=sc, x=x, y=y, k=k, flip=flip, w=v.shape[1], h=v.shape[0], name=name))
    return out
