"""Door leaves, swing arcs, coiling panels and sliding panels."""
import cv2
import numpy as np
from scipy import ndimage


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def panel_doors(a, walls):
    sliding, coiling = [], []
    for wi, wall in enumerate(walls):
        c, lo, hi = round(wall.c), round(wall.lo), round(wall.hi)
        image = a if wall.axis == 0 else a.T
        # Sliding leaves are two overlapping solid lines in the wall's body.
        # Opening before closing suppresses CMU hatch marks and then joins the
        # two leaf strokes, without bridging separate doors.
        strip = image[c-2:c+3, lo:hi+1]
        profile = strip.min(0) < 125
        panel = ndimage.binary_opening(profile, structure=np.ones(25))
        panel = ndimage.binary_closing(panel, structure=np.ones(13))
        for l, r in intervals(panel):
            if 105 < r-l < 205:
                sliding.append({"wall": wi, "lo": int(l+lo), "hi": int(r+lo)})
        if wall.kind not in ["EXT", "P3"]:
            continue
        # The coiling-door dashed center stroke has sustained contrast against
        # the cleared wall body. The diagonal CMU hatch averages to zero.
        strip = image[max(0, c-6):c+7, lo:hi+1].astype(np.float32)
        contrast = ((strip[1:3].mean(0)+strip[10:12].mean(0))/2-strip[5:8].min(0))
        response = ndimage.uniform_filter1d(contrast, 80)
        for l, r in intervals(response > 45):
            if r-l < 100:
                continue
            l, r = int(l+lo), int(r+lo)
            if any(d["wall"] == wi and min(r, d["hi"]) > max(l, d["lo"]) for d in sliding):
                continue
            coiling.append({"wall": wi, "lo": l, "hi": r})
    return sliding, coiling


def leaf_candidates(a, walls):
    found = []
    for axis in [0, 1]:
        for threshold in [165, 180, 200]:
            ink = (a < threshold).astype(np.uint8)
            kernel = np.ones((1, 45) if axis == 0 else (45, 1), np.uint8)
            lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
            n, _, stats, centers = cv2.connectedComponentsWithStats(lines)
            for (x, y, w, h, area), (cx, cy) in zip(stats[1:], centers[1:]):
                c, lo, hi, thick = (cy, x, x+w-1, h) if axis == 0 else (cx, y, y+h-1, w)
                if not (60 < hi-lo < 110 and thick < 16):
                    continue
                near = [(min(abs(v.c-lo), abs(v.c-hi)), i, v) for i, v in enumerate(walls)
                        if v.axis != axis and v.lo-10 < c < v.hi+10
                        and min(abs(v.c-lo), abs(v.c-hi)) < 24]
                parallel = [v for v in walls if v.axis == axis and abs(v.c-c) < 24
                            and min(hi, v.hi) > max(lo, v.lo)]
                if not near or parallel:
                    continue
                _, wi, wall = min(near, key=lambda t: t[0])
                side = 1 if (lo+hi)/2 > wall.c else -1
                found.append({"axis": axis, "along": float(c), "wall": wi, "side": side})
    # A leaf is drawn with two faces. Combine them before finding its arc.
    unique = []
    for leaf in found:
        close = next((v for v in unique if leaf["axis"] == v["axis"] and leaf["wall"] == v["wall"]
                      and leaf["side"] == v["side"] and abs(leaf["along"]-v["along"]) < 7), None)
        if close:
            close["along"] = (close["along"]+leaf["along"])/2
        else:
            unique.append(leaf)
    return unique


def fit_arc(a, leaf, wall):
    theta = np.linspace(np.deg2rad(12), np.deg2rad(78), 90)
    si, co = np.sin(theta), np.cos(theta)
    best = None
    # Curve samples exclude the wall and the straight leaf. Hatches cannot
    # supply a continuous quarter-circle at the graphic-scale radius.
    for swing in [-1, 1]:
        for shift in [-4., -2., 0., 2., 4.]:
            for offset in [-2., 0., 2.]:
                for radius in [71., 72., 73.]:
                    along = leaf["along"]+shift
                    base = wall.c+leaf["side"]*wall.width/2+offset
                    vals = []
                    for radial in [-1., 0., 1., -5., 5.]:
                        r = radius+radial
                        u = along+swing*r*si
                        v = base+leaf["side"]*r*co
                        x, y = (u, v) if leaf["axis"] == 1 else (v, u)
                        vals.append(ndimage.map_coordinates(a, [y, x], order=1, mode="constant", cval=255))
                    value = np.min(vals[:3], axis=0)
                    background = np.mean(vals[3:], axis=0)
                    contrast = background-value
                    score = float(np.clip(contrast, -30, 100).mean()/100.)
                    coverage = float((contrast > 15).mean())
                    record = dict(leaf, score=score, coverage=coverage, swing=swing,
                                  hinge=along, base=base, radius=radius)
                    if best is None or score > best["score"]:
                        best = record
    return best


def recognize_doors(a, walls, debug=None):
    candidates = leaf_candidates(a, walls)
    arcs = [fit_arc(a, v, walls[v["wall"]]) for v in candidates]
    valid = [v for v in arcs if v["score"] > .22 and v["coverage"] > .65]
    pairs = []
    used = set()
    for i, v in enumerate(valid):
        for j, u in enumerate(valid[:i]):
            if i in used or j in used:
                continue
            if (v["wall"] == u["wall"] and v["side"] == u["side"]
                    and abs(abs(v["hinge"]-u["hinge"])-144) < 9
                    and v["swing"] == np.sign(u["hinge"]-v["hinge"])
                    and u["swing"] == np.sign(v["hinge"]-u["hinge"])):
                used.update([i, j])
                pairs.append([j, i])
    sliding, coiling = panel_doors(a, walls)
    counts = {"single": len(valid)-len(used), "pair": len(pairs),
              "overhead_coiling": len(coiling), "auto_sliding": len(sliding)}
    if debug is not None:
        vis = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        for i, v in enumerate(arcs):
            col = (0, 0, 255) if v in valid else (255, 100, 0)
            c, b = v["hinge"], v["base"]
            x, y = (c, b) if v["axis"] == 1 else (b, c)
            cv2.circle(vis, (round(x), round(y)), 8, col, 3)
            cv2.putText(vis, f"{i}:{v['score']:.2f}/{v['coverage']:.2f}", (round(x+5), round(y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, col, 2)
        cv2.imwrite(str(debug / "door_arcs.png"), vis)
    return counts, {"arcs": arcs, "pairs": pairs, "sliding": sliding, "coiling": coiling}
