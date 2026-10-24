"""Fixture counting by template matching on the line map (fixtures are standard CAD blocks)."""
import numpy as np
import cv2

# Fixture block geometry in feet: list of primitives
#   ('L', x0, y0, x1, y1) line, ('R', x0, y0, x1, y1) rectangle, ('C', cx, cy, r) circle
BLOCKS = {
    'produce': [('R', 0, 0, 5, 4), ('L', 0, 0, 5, 4), ('L', 5, 0, 0, 4)],
    'kiosk': [('R', 0, 0, 3, 3), ('C', 1.5, 1.5, 0.8)],
    'checkout': [('R', 0, 0, 2.5, 9), ('L', 1.25, 0.45, 1.25, 8.6), ('R', 0, 9, 2.5, 12),
                 ('L', 0, 9, 2.5, 12), ('L', 2.5, 9, 0, 12), ('R', 2.5, 4.8, 3.7, 6.0)],
    'endcap': [('R', 0, 0, 4, 1.5), ('L', 0, 1.5, 4, 0)],
    'cooler': [('R', 0, 0, 2.35, 0.6), ('L', 0, 0.6, 2.35, 0)],
    'cooler2': [('B', 0, 0, 2.35, 2.1), ('R', 0, 1.5, 2.35, 2.1), ('L', 0, 2.1, 2.35, 1.5)],
    'sec4': [('R', 0, 0, 4, 4), ('L', 2, 0, 2, 4)],
    'sec3': [('R', 0, 0, 4, 3), ('L', 2, 0, 2, 3)],
}


def _transform(prims, rot, mirror):
    """Rotate by rot*90 deg and optionally mirror (x -> -x) a primitive list; renormalise to origin."""
    def tp(x, y):
        if mirror:
            x = -x
        for _ in range(rot):
            x, y = -y, x
        return x, y
    out = []
    for p in prims:
        if p[0] in ('L', 'R', 'B'):
            a = tp(p[1], p[2]); b = tp(p[3], p[4])
            out.append((p[0], a[0], a[1], b[0], b[1]))
        else:
            c = tp(p[1], p[2])
            out.append(('C', c[0], c[1], p[3]))
    xs, ys = [], []
    for p in out:
        if p[0] == 'C':
            xs += [p[1] - p[3], p[1] + p[3]]; ys += [p[2] - p[3], p[2] + p[3]]
        else:
            xs += [p[1], p[3]]; ys += [p[2], p[4]]
    mx, my = min(xs), min(ys)
    res = []
    for p in out:
        if p[0] == 'C':
            res.append(('C', p[1] - mx, p[2] - my, p[3]))
        else:
            res.append((p[0], p[1] - mx, p[2] - my, p[3] - mx, p[4] - my))
    return res


def render(prims, ppf, pad_ft=0.25, lw=2):
    xs, ys = [], []
    for p in prims:
        if p[0] == 'C':
            xs += [p[1] + p[3]]; ys += [p[2] + p[3]]
        else:
            xs += [p[1], p[3]]; ys += [p[2], p[4]]
    W = int(np.ceil((max(xs) + 2 * pad_ft) * ppf)) + 1
    H = int(np.ceil((max(ys) + 2 * pad_ft) * ppf)) + 1
    S = 4
    img = np.zeros((H * S, W * S), np.uint8)
    def P(x, y):
        return (int(round((x + pad_ft) * ppf * S)), int(round((y + pad_ft) * ppf * S)))
    for p in prims:
        if p[0] == 'L':
            cv2.line(img, P(p[1], p[2]), P(p[3], p[4]), 255, lw * S, cv2.LINE_AA)
        elif p[0] == 'R':
            cv2.rectangle(img, P(p[1], p[2]), P(p[3], p[4]), 255, lw * S, cv2.LINE_AA)
        elif p[0] == 'B':
            pass
        else:
            cv2.circle(img, P(p[1], p[2]), int(round(p[3] * ppf * S)), 255, lw * S, cv2.LINE_AA)
    t = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return t


def variants(name):
    prims = BLOCKS[name]
    seen = set()
    out = []
    for mirror in (False, True):
        for rot in range(4):
            v = _transform(prims, rot, mirror)
            key = tuple(sorted(tuple(round(c, 3) for c in p[1:]) + (p[0],) for p in v))
            if key in seen:
                continue
            seen.add(key)
            out.append(((rot, mirror), v))
    return out


def match_image(lm):
    m = np.clip(lm, 0, 0.8) / 0.8
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)


def detect(img, name, ppf, thr, nms_ft=None, pad_ft=0.25, overlap=0.3):
    """Return list of (score, x, y, w, h, variant) detections (top-left in img coords)."""
    dets = []
    for var, prims in variants(name):
        t = render(prims, ppf, pad_ft=pad_ft)
        px, py = [], []
        for p in prims:
            if p[0] in ('L', 'R'):
                px += [p[1], p[3]]; py += [p[2], p[4]]
            elif p[0] == 'C':
                px += [p[1] - p[3], p[1] + p[3]]; py += [p[2] - p[3], p[2] + p[3]]
        drawn = ((min(px) + pad_ft) * ppf, (min(py) + pad_ft) * ppf, (max(px) + pad_ft) * ppf, (max(py) + pad_ft) * ppf)
        t = cv2.GaussianBlur(t, (0, 0), 0.8)
        if t.shape[0] >= img.shape[0] or t.shape[1] >= img.shape[1]:
            continue
        r = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        # local maxima above threshold
        k = max(3, int(0.5 * ppf) | 1)
        mx = cv2.dilate(r, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        ys, xs = np.nonzero((r >= thr) & (r >= mx))
        if len(ys) > 5000:
            o = np.argsort(-r[ys, xs])[:5000]
            ys, xs = ys[o], xs[o]
        for y, x in zip(ys, xs):
            dets.append((float(r[y, x]), int(x), int(y), t.shape[1], t.shape[0], var,
                         (x + drawn[0], y + drawn[1], x + drawn[2], y + drawn[3])))
    dets.sort(key=lambda d: -d[0])
    # NMS by centre distance / overlap
    keep = []
    if not dets:
        return keep
    arr = np.array([[d[1], d[2], d[1] + d[3], d[2] + d[4]] for d in dets], float)
    area = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    alive = np.ones(len(dets), bool)
    for i in range(len(dets)):
        if not alive[i]:
            continue
        keep.append(dets[i])
        ox = np.minimum(arr[i, 2], arr[:, 2]) - np.maximum(arr[i, 0], arr[:, 0])
        oy = np.minimum(arr[i, 3], arr[:, 3]) - np.maximum(arr[i, 1], arr[:, 1])
        inter = np.clip(ox, 0, None) * np.clip(oy, 0, None)
        sup = inter > overlap * np.minimum(area[i], area)
        alive &= ~sup
    return keep
