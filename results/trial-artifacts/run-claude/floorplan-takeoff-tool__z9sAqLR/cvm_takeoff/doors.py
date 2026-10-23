"""Door detection: openings in wall runs, classified by what is drawn in and around them."""
import numpy as np
import cv2
from scipy import ndimage as ndi

from . import walls as W


def find_openings(runs, idxs, ppf, min_pier_ft=0.75):
    """Non-wall intervals of wall runs.

    Neighbouring openings are merged unless a substantial piece of wall (a pier of at least
    `min_pier_ft`) separates them: jamb lines, overlapping sliding panels and the like
    produce short wall-looking slivers inside a single opening.
    """
    ops = []
    for i in idxs:
        r = runs[i]
        ivs = sorted(r['intervals'], key=lambda iv: iv['a'])
        cur = None
        wall_run = 0.0
        for iv in ivs:
            if iv['cls'] in W.WALL_CLASSES:
                wall_run += (iv['b'] - iv['a'] + 1)
                if cur is not None and wall_run >= min_pier_ft * ppf:
                    ops.append(cur)
                    cur = None
                continue
            if cur is not None:
                cur['b'] = iv['b']
                cur['parts'].append(iv)
            else:
                cur = dict(run=i, axis=r['axis'], c=r['c'], p1=r['p1'], p2=r['p2'], t=r['t'],
                           wall=r.get('type', r['cls']), a=iv['a'], b=iv['b'], parts=[iv],
                           ext_side=r.get('ext_side', 0))
            wall_run = 0.0
        if cur is not None:
            ops.append(cur)
    # discard openings that are too narrow to be doors
    return [o for o in ops if (o['b'] - o['a'] + 1) >= 1.5 * ppf]


def _sample(img, xs, ys):
    return ndi.map_coordinates(img, [ys, xs], order=1, mode='constant', cval=0.0)


def arc_score(Ds, cx, cy, R, th0, th1, n_pts=48):
    """Center-surround score of a circular arc (angles in radians, image coords y down)."""
    th = np.linspace(th0, th1, n_pts)
    # skip the ends (near the wall and the leaf)
    th = th[int(0.12 * n_pts):int(0.88 * n_pts)]
    best = -1e9
    for dr in (-3, -1.5, 0, 1.5, 3):
        rr = R + dr
        xs = cx + rr * np.cos(th)
        ys = cy + rr * np.sin(th)
        nx, ny = np.cos(th), np.sin(th)
        c = _sample(Ds, xs, ys)
        o1 = _sample(Ds, xs + 3.5 * nx, ys + 3.5 * ny)
        o2 = _sample(Ds, xs - 3.5 * nx, ys - 3.5 * ny)
        s = c - 0.5 * (o1 + o2)
        v = float(np.median(s))
        best = max(best, v)
    return best


def swing_scores(Ds, op, ppf):
    """Best single-leaf and pair arc scores for an opening."""
    a, b = op['a'], op['b']
    e1, e2 = a - 2.0, b + 2.0
    w = e2 - e1
    out = dict(single=-1e9, pair=-1e9, single_hyp=None, pair_hyp=None)
    for face, sgn in ((op['p1'], -1.0), (op['p2'], 1.0)):
        for doff in (-2.0, 0.0, 2.0):
            f = face + sgn * doff
            for eoff in (-3.0, 0.0, 3.0):
                for scale in (0.94, 1.0, 1.06):
                    R = w * scale
                    # single hinged at e1 / e2
                    for hinge, other in ((e1 + eoff, 1.0), (e2 - eoff, -1.0)):
                        if op['axis'] == 0:
                            cx, cy = hinge, f
                            # closed position along +/-x, open position along sgn*y
                            t_closed = 0.0 if other > 0 else np.pi
                            t_open = np.pi / 2 if sgn > 0 else -np.pi / 2
                        else:
                            cx, cy = f, hinge
                            t_closed = np.pi / 2 if other > 0 else -np.pi / 2
                            t_open = 0.0 if sgn > 0 else np.pi
                        t0, t1 = _arc_range(t_closed, t_open)
                        s = arc_score(Ds, cx, cy, R, t0, t1)
                        if s > out['single']:
                            out['single'] = s
                            out['single_hyp'] = (hinge, f, R, sgn)
                    # pair: two arcs of half width
                    Rh = 0.5 * R
                    sc = []
                    for hinge, other in ((e1 + eoff, 1.0), (e2 - eoff, -1.0)):
                        if op['axis'] == 0:
                            cx, cy = hinge, f
                            t_closed = 0.0 if other > 0 else np.pi
                            t_open = np.pi / 2 if sgn > 0 else -np.pi / 2
                        else:
                            cx, cy = f, hinge
                            t_closed = np.pi / 2 if other > 0 else -np.pi / 2
                            t_open = 0.0 if sgn > 0 else np.pi
                        t0, t1 = _arc_range(t_closed, t_open)
                        sc.append(arc_score(Ds, cx, cy, Rh, t0, t1))
                    s = min(sc)
                    if s > out['pair']:
                        out['pair'] = s
                        out['pair_hyp'] = (e1, e2, f, Rh, sgn)
    return out


def _arc_range(t_closed, t_open):
    """Shortest angular range between closed and open directions (quarter turn)."""
    d = (t_open - t_closed + np.pi) % (2 * np.pi) - np.pi
    return (t_closed, t_closed + d)


def opening_content(D, op):
    """Darkness statistics inside the wall opening (between the faces)."""
    a, b = int(op['a']), int(op['b'])
    band = W.band_array(D, op['axis'], op['p1'], op['p2'], a, b)
    if band.size == 0:
        return dict(mean=0.0, rowmax=0.0)
    rowm = band.mean(1)
    return dict(mean=float(band.mean()), rowmax=float(rowm.max()))


def classify_doors(D, runs, idxs, ppf, k=1.0):
    ops = find_openings(runs, idxs, ppf)
    Ds = cv2.GaussianBlur(D, (0, 0), 1.0)
    counts = dict(single=0, pair=0, overhead_coiling=0, auto_sliding=0)
    details = []
    for op in ops:
        w_ft = (op['b'] - op['a'] + 5) / ppf
        cont = opening_content(D, op)
        sw = swing_scores(Ds, op, ppf)
        kind = None
        if cont['mean'] >= 40.0:
            kind = 'auto_sliding'
        else:
            best_swing = max(sw['single'], sw['pair'])
            if best_swing >= 18.0 * k:
                if sw['pair'] >= 0.75 * sw['single'] and w_ft >= 4.0:
                    kind = 'pair'
                elif sw['single'] >= sw['pair']:
                    kind = 'single'
                else:
                    kind = 'pair' if w_ft >= 4.0 else 'single'
            elif w_ft >= 6.0 and cont['rowmax'] >= 45.0 * k:
                kind = 'overhead_coiling'
        if kind is not None:
            counts[kind] += 1
        details.append(dict(op=op, width_ft=w_ft, content=cont, swing=sw, kind=kind))
    return counts, details
