"""Line and region geometry in rectified drawing pixels."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class Wall:
    axis: int                    # 0 horizontal, 1 vertical
    c: float
    lo: float
    hi: float
    width: float
    kind: str

    def record(self):
        return asdict(self)


def line_segments(a, axis, threshold=135, minimum=38):
    ink = (a < threshold).astype(np.uint8)
    kernel = np.ones((1, minimum) if axis == 0 else (minimum, 1), np.uint8)
    mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    lines = []
    for st, center in zip(stats[1:], centers[1:]):
        x, y, w, h, area = st
        length, thick = (w, h) if axis == 0 else (h, w)
        if length < minimum or thick > 7:
            continue
        lo = x if axis == 0 else y
        c = center[1] if axis == 0 else center[0]
        lines.append([float(c), float(lo), float(lo + length - 1)])
    # Rejoin small scan-induced interruptions in a single face.
    lines.sort()
    merged = []
    for c, lo, hi in lines:
        old = next((v for v in reversed(merged) if abs(v[0]-c) < 1.8
                    and min(v[2], hi) >= max(v[1], lo)-5), None)
        if old is None:
            merged.append([c, lo, hi])
        else:
            old[0] = (old[0]+c)/2
            old[1] = min(old[1], lo)
            old[2] = max(old[2], hi)
    return sorted(merged)


def detect_walls(a, box, ppf=24.):
    walls = []
    for axis in [0, 1]:
        faces = line_segments(a, axis)
        for i, (c, lo, hi) in enumerate(faces):
            for d, start, end in faces[i+1:]:
                width = d - c
                if width > ppf * .79:
                    break
                if not (ppf * .31 < width < ppf * .79):
                    continue
                l, r = max(lo, start), min(hi, end)
                if r - l < ppf * 1.5:
                    continue
                # Interior wall faces are both dark; use the wall body to
                # distinguish the solid fire partition from CMU hatching.
                q0, q1 = round((c+d)/2-width*.2), round((c+d)/2+width*.2)+1
                if axis == 0:
                    patch = a[q0:q1, round(l+5):round(r-4)]
                else:
                    patch = a[round(l+5):round(r-4), q0:q1]
                if patch.size == 0:
                    continue
                kind = "P1" if width < ppf * .53 else ("P2" if np.median(patch) < 205 else "P3")
                walls.append(Wall(axis, (c+d)/2, l, r, width, kind))
    # Exterior is a continuous centerline, including every door opening.
    left, top, right, bottom = box
    ext_width = ppf * 2/3
    xc = [left + ext_width/2, right - ext_width/2]
    yc = [top + ext_width/2, bottom - ext_width/2]
    walls = [w for w in walls if not (w.axis == 0 and min(abs(w.c-yc[0]), abs(w.c-yc[1])) < 7)
             and not (w.axis == 1 and min(abs(w.c-xc[0]), abs(w.c-xc[1])) < 7)]
    # Merge faces across openings and at wall intersections.
    walls.sort(key=lambda w: (w.axis, w.c, w.lo))
    merged = []
    for w in walls:
        merged.append(w)
    changed = True
    def supported_gap(w, other):
        lo, hi = min(w.hi, other.hi), max(w.lo, other.lo)
        if hi <= lo:
            return True
        lo, hi = round(lo), round(hi)
        for offset in [-1, 1]:
            c = round((w.c+other.c)/2+offset*(w.width+other.width)/4)
            patch = a[c-2:c+3, lo:hi] if w.axis == 0 else a[lo:hi, c-2:c+3].T
            if patch.size and (patch.min(0) < 155).mean() > .78:
                return True
        return False
    while changed:
        changed = False
        for i, w in enumerate(merged):
            match = None
            for j, old in enumerate(merged[:i]):
                if (w.axis == old.axis and w.kind == old.kind and abs(w.c-old.c) < 2.2
                        and (min(w.hi, old.hi) >= max(w.lo, old.lo) - ppf*(7 if w.axis == 0 else 4.2)
                             or supported_gap(w, old))):
                    match = old
                    break
            if match:
                match.c = (match.c + w.c)/2
                match.lo = min(match.lo, w.lo)
                match.hi = max(match.hi, w.hi)
                match.width = (match.width + w.width)/2
                merged.pop(i)
                changed = True
                break
    merged.extend([Wall(0, y, xc[0], xc[1], ext_width, "EXT") for y in yc])
    merged.extend([Wall(1, x, yc[0], yc[1], ext_width, "EXT") for x in xc])
    # Extend each centerline to the centerline of the wall it meets.
    for w in merged:
        for name in ["lo", "hi"]:
            end = getattr(w, name)
            near = [v.c for v in merged if v.axis != w.axis
                    and v.lo - 14 <= w.c <= v.hi + 14 and abs(v.c-end) < 16]
            if near:
                setattr(w, name, min(near, key=lambda t: abs(t-end)))
    return merged


def draw_walls(a, walls):
    out = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    colors = {"EXT": (200, 0, 0), "P1": (0, 150, 0), "P2": (0, 0, 240), "P3": (160, 0, 160)}
    for i, w in enumerate(walls):
        p = (round(w.lo), round(w.c)) if w.axis == 0 else (round(w.c), round(w.lo))
        q = (round(w.hi), round(w.c)) if w.axis == 0 else (round(w.c), round(w.hi))
        cv2.line(out, p, q, colors[w.kind], 4)
        cv2.putText(out, f"{i}:{w.kind}", p, cv2.FONT_HERSHEY_SIMPLEX, .6, colors[w.kind], 2)
    return out
