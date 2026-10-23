"""Reading the sheet legend: partition-type tags per wall appearance, finish codes.

The examples all use the same legend; this only guards against a sheet whose
legend assigns tags or codes differently.  Every step falls back to the defaults
unless the reading is unambiguous.
"""
import os
import numpy as np
import cv2

from . import fixtures as X

_TPL = None
DEFAULT_WALL_MAP = {'ext': 'EXT', 'hatch': 'P3', 'gray': 'P2', 'light': 'P1'}


def _templates():
    global _TPL
    if _TPL is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'legend.npz')
        d = np.load(path)
        _TPL = {k: d[k].astype(np.float32) for k in d.files}
    return _TPL


def _best(Gc, tpl, x0, y0, x1, y1):
    H, W = Gc.shape
    x0 = max(0, int(x0)); y0 = max(0, int(y0)); x1 = min(W, int(x1)); y1 = min(H, int(y1))
    win = Gc[y0:y1, x0:x1]
    if win.shape[0] <= tpl.shape[0] or win.shape[1] <= tpl.shape[1]:
        return -1.0, None
    R = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    i = int(np.argmax(R))
    yy, xx = divmod(i, R.shape[1])
    return float(R.flat[i]), (x0 + xx + tpl.shape[1] / 2.0, y0 + yy + tpl.shape[0] / 2.0)


def partition_swatches(pieces, s):
    """Legend partition swatches: short horizontal wall samples, left aligned, stacked."""
    cands = [p for p in pieces if p['o'] == 'H' and 6.0 * s <= (p['a1'] - p['a0']) <= 9.0 * s and p.get('cls')]
    best = []
    for p in cands:
        grp = [q for q in cands if abs(q['a0'] - p['a0']) < 0.5 * s and abs(q['a1'] - p['a1']) < 0.5 * s]
        if len(grp) > len(best):
            best = grp
    best = sorted(best, key=lambda q: q['c'])
    # de-duplicate (same row detected twice)
    out = []
    for q in best:
        if out and abs(q['c'] - out[-1]['c']) < 1.0 * s:
            continue
        out.append(q)
    return out if len(out) >= 3 else []


def read_wall_map(G, s, pieces):
    sw = partition_swatches(pieces, s)
    if not sw:
        return DEFAULT_WALL_MAP, None
    T = _templates()
    f = X.C / s
    tags = ['EXT', 'P1', 'P2', 'P3']
    x0 = int(min(q['a0'] for q in sw)) - int(s)
    x1 = int(max(q['a1'] for q in sw)) + int(6 * s)
    y0 = int(sw[0]['c'] - 2 * s)
    y1 = int(sw[-1]['c'] + 2 * s)
    sub = G[max(0, y0):y1, max(0, x0):x1]
    Gc = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    rows = []
    for q in sw:
        cx0 = (q['a1'] - max(0, x0) + 0.8 * s) * f
        cx1 = (q['a1'] - max(0, x0) + 4.8 * s) * f
        cy0 = (q['c'] - max(0, y0) - 1.0 * s) * f
        cy1 = (q['c'] - max(0, y0) + 1.0 * s) * f
        sc = {t: _best(Gc, T['tag_' + t], cx0, cy0, cx1, cy1)[0] for t in tags}
        order = sorted(sc, key=lambda t: -sc[t])
        rows.append(dict(cls=q['cls'], tag=order[0], score=sc[order[0]], margin=sc[order[0]] - sc[order[1]]))
    # build appearance -> tag map
    m = {}
    ok = True
    for r in rows:
        if r['score'] < 0.6 or r['margin'] < 0.03:
            ok = False
            break
        key = r['cls']
        if r['tag'] == 'EXT':
            key = 'ext'
        if key in m and m[key] != r['tag']:
            ok = False
            break
        m[key] = r['tag']
    if not ok or set(m.keys()) != {'ext', 'hatch', 'gray', 'light'} or len(set(m.values())) != 4:
        return DEFAULT_WALL_MAP, rows
    return m, rows


def read_finish_codes(G, s, pieces, default_codes):
    """Finish codes listed above the partition legend, in legend order."""
    sw = partition_swatches(pieces, s)
    if not sw:
        return list(default_codes), None
    T = _templates()
    f = X.C / s
    left = min(q['a0'] for q in sw)
    x0 = int(left - 1.0 * s); x1 = int(left + 16.0 * s)
    y1 = int(sw[0]['c'] - 1.0 * s); y0 = max(0, int(y1 - 50 * s))
    sub = G[y0:y1, max(0, x0):x1]
    if sub.size == 0:
        return list(default_codes), None
    Gc = cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    found = []
    for k in default_codes:
        key = 'code_' + k
        if key not in T:
            continue
        R = cv2.matchTemplate(Gc, T[key], cv2.TM_CCOEFF_NORMED)
        i = int(np.argmax(R))
        yy, xx = divmod(i, R.shape[1])
        found.append(dict(code=k, score=float(R.flat[i]), y=yy, x=xx))
    good = [d for d in found if d['score'] >= 0.75]
    if len(good) != len(default_codes):
        return list(default_codes), found
    # distinct rows
    ys = sorted(d['y'] for d in good)
    if min(np.diff(ys)) < 2.0 * X.C:
        return list(default_codes), found
    return [d['code'] for d in sorted(good, key=lambda d: d['y'])], found
