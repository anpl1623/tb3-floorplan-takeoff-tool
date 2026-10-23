"""Door openings (gaps in wall runs) and their classification."""
import numpy as np
import cv2

from canon import PPF
import params
from walls import OPEN, NOMINAL_T_PX


def find_openings(runs):
    ops = []
    for r in runs:
        perp = [p for p in runs if p.orient != r.orient and p.s0 - p.t <= r.c <= p.s1 + p.t]
        walls = sorted([p for p in r.pieces if p[2] != OPEN])
        if not walls:
            continue
        gaps = list(zip([w[1] for w in walls[:-1]], [w[0] for w in walls[1:]]))
        # openings at the run ends (run extended through a door to a wall)
        if walls[0][0] - r.s0 >= 1.5 * PPF:
            gaps.insert(0, (r.s0, walls[0][0]))
        if r.s1 - walls[-1][1] >= 1.5 * PPF:
            gaps.append((walls[-1][1], r.s1))
        for g0, g1 in gaps:
            if g1 - g0 < 1.0 * PPF:
                continue
            # (u, v, u_at_wall, v_at_wall): jamb formed by a perpendicular wall
            segs = [(g0, g1, False, False)]
            for p in perp:
                lo, hi = p.c - p.t / 2 - 2, p.c + p.t / 2 + 2
                new = []
                for (u, v, uw, vw) in segs:
                    if hi <= u or lo >= v:
                        new.append((u, v, uw, vw))
                        continue
                    if lo > u:
                        new.append((u, lo, uw, True))
                    if hi < v:
                        new.append((hi, v, True, vw))
                segs = new
            for (u, v, uw, vw) in segs:
                if v - u >= 1.5 * PPF:
                    ops.append(dict(run=r, u=float(u), v=float(v), u_wall=uw, v_wall=vw))
    return ops


def _local(g, r):
    """Image in the run's frame: rows across the wall, columns along it."""
    return g if r.orient == 'H' else g.T


LEAF_DARK = 165.0


def classify_opening(g, op, perp_lines):
    """perp_lines: dict orient -> mask (in that run's local frame) of dark
    lines running perpendicular to the wall."""
    r = op['run']
    L = _local(g, r)
    Hh, Ww = L.shape
    u, v = op['u'], op['v']
    T = NOMINAL_T_PX(r)
    c = r.c
    f0, f1 = c - T / 2, c + T / 2           # wall faces
    leaves = []
    reach = int(2.2 * PPF)
    for side in (-1, 1):
        face = f0 if side < 0 else f1
        if side < 0:
            ya, yb = int(face - reach), int(face - 3)
        else:
            ya, yb = int(face + 3), int(face + reach)
        ya, yb = max(ya, 0), min(yb, Hh - 1)
        if yb - ya < 10:
            continue
        for jamb, sgn, at_wall in ((u, 1, op.get('u_wall', False)), (v, -1, op.get('v_wall', False))):
            # next to a perpendicular wall, keep its outline out of the window
            lead = 2 if at_wall else -1
            xa = int(round(jamb + lead)) if sgn > 0 else int(round(jamb - 10))
            xb = int(round(jamb + 10)) if sgn > 0 else int(round(jamb - lead))
            xa, xb = max(xa, 0), min(xb, Ww - 1)
            blk = L[ya:yb, xa:xb].astype(np.float32)
            if blk.size == 0:
                continue
            # a leaf is a straight dark bar perpendicular to the wall: the
            # mean over its length of the darkest 2-px column pair is low
            cm = blk.mean(axis=0)
            dark = float(cm.min())
            if dark < params.thr(LEAF_DARK):
                leaves.append((side, jamb, dark))
    # interior of the band within the opening (sliding panels)
    ia, ib = int(np.ceil(f0 + 2.5)), int(np.floor(f1 - 2.5))
    xa, xb = int(u + 4), int(v - 4)
    panel = 0.0
    dash = 0.0
    fill = 255.0
    if ib > ia and xb > xa:
        blk = L[ia:ib + 1, xa:xb]
        panel = float((blk < params.thr(110)).any(axis=0).mean())
        dash = float((blk < params.thr(205)).any(axis=0).mean())
        fill = float(np.median(blk.astype(np.float32).mean(axis=0)))
        # a coil-door opening shows one dashed line; hatch/fill shows in every row
        rowfrac = (blk < params.thr(205)).mean(axis=1)
        if (rowfrac > 0.2).sum() > 4 or rowfrac.max() < 0.35:
            dash = 0.0
    sides = {}
    for s, j, fr in leaves:
        sides.setdefault(s, set()).add(round(j))
    nleaf = max((len(v_) for v_ in sides.values()), default=0)
    width = (v - u) / PPF
    blank = fill >= params.thr(226)      # opening shows paper between the faces
    if panel > 0.5:
        kind = 'auto_sliding'
    elif not blank and dash <= 0.7:
        kind = None          # wall fill continues: not an opening
    elif nleaf >= 2:
        kind = 'pair'
    elif nleaf == 1:
        kind = 'single'
    elif dash > 0.7 or width >= 8.0:
        kind = 'overhead_coiling'
    else:
        kind = None          # gap without door evidence
    op.update(kind=kind, nleaf=nleaf, panel=panel, dash=dash, fill=fill, width=width, leaves=leaves)
    return kind


def perp_line_masks(g, thr=145, minlen=19):
    dark = (g < thr).astype(np.uint8)
    vl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((minlen, 1), np.uint8)).astype(bool)
    hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, minlen), np.uint8)).astype(bool)
    # for an H wall, leaves are vertical lines (local frame = g);
    # for a V wall the local frame is g.T and leaves are horizontal lines in g
    return {'H': vl, 'V': np.ascontiguousarray(hl.T)}


def count_doors(g, runs):
    ops = find_openings(runs)
    pl = perp_line_masks(g)
    counts = {'single': 0, 'pair': 0, 'overhead_coiling': 0, 'auto_sliding': 0}
    for op in ops:
        k = classify_opening(g, op, pl)
        if k is not None:
            counts[k] += 1
    return counts, ops


def extend_free_ends(g, runs, max_gap_ft=14.0):
    """A run that stops short of a perpendicular wall with a door in the gap
    (door at the end of a wall): extend it to that wall."""
    changed = False
    for r in runs:
        for end in (0, 1):
            if (r.j0 if end == 0 else r.j1) is not None:
                continue
            s = r.s0 if end == 0 else r.s1
            best = None
            for p in runs:
                if p.orient == r.orient:
                    continue
                tol = NOMINAL_T_PX(r) / 2 + 4
                if not (p.s0 - tol <= r.c <= p.s1 + tol):
                    continue
                hp = NOMINAL_T_PX(p) / 2
                face = p.c + hp if end == 0 else p.c - hp
                d = (s - face) if end == 0 else (face - s)
                if 1.5 * PPF <= d <= max_gap_ft * PPF and (best is None or d < best[0]):
                    best = (d, p, face)
            if best is None:
                continue
            d, p, face = best
            lo_, hi_ = (face, s) if end == 0 else (s, face)
            # the stretch must not already belong to a collinear wall run
            if any(q is not r and q.orient == r.orient and abs(q.c - r.c) < 4 and
                   min(q.s1, hi_) - max(q.s0, lo_) > 4 for q in runs):
                continue
            u, v = (face + 3, s) if end == 0 else (s, face - 3)
            op = dict(run=r, u=float(u), v=float(v))
            classify_opening(g, op, None)
            free_jamb = s
            leaf_at_free = any(abs(j - free_jamb) < 2 for _, j, _ in op['leaves'])
            if leaf_at_free or op['nleaf'] >= 1 or op['panel'] > 0.5:
                if end == 0:
                    r.s0 = int(round(face))
                else:
                    r.s1 = int(round(face))
                r.pieces.append((min(u, v), max(u, v), OPEN))
                r.pieces.sort()
                changed = True
    return changed
