"""Render a store's floor plan as a 36" x 24" CAD-style plot, the way the chain's architect plots A101."""

import math
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Polygon as MplPolygon, Rectangle, RegularPolygon, Arc  # noqa: E402
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from plan import THICKNESS, E  # noqa: E402

DPI = 200
SHEET_W, SHEET_H = 36.0, 24.0
SCALE = 1 / 8                 # sheet inches per foot (1/8" = 1'-0")
OX, OY = 2.6, 4.4             # sheet position of the building's SW outer corner

HATCH = dict(color="0.45", linewidth=0.3)
FIX_LW = 0.35
WALL_LW = 1.0
EDGE_LW = 0.8      # finish transitions and thresholds


def S(x, y):
    return OX + x * SCALE, OY + y * SCALE


def ft_in(feet):
    total = round(feet * 12)
    f, i = divmod(total, 12)
    return f"{f}'-{i}\""


# ----------------------------------------------------------------- hatches --
def _lines_clip(poly, segs):
    if poly.is_empty:
        return []
    clipped = poly.intersection(MultiLineString(segs))
    out = []
    for g in getattr(clipped, "geoms", [clipped]):
        if isinstance(g, LineString) and not g.is_empty:
            out.append(list(g.coords))
        elif hasattr(g, "geoms"):
            out.extend(list(x.coords) for x in g.geoms if isinstance(x, LineString))
    return out


def _parallel(poly, angle, spacing, phase=0.0):
    minx, miny, maxx, maxy = poly.bounds
    r = math.hypot(maxx - minx, maxy - miny) + 2
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    a = math.radians(angle)
    d = (math.cos(a), math.sin(a))
    n = (-d[1], d[0])
    # offsets measured from the origin so patterns stay aligned across zones
    c0 = cx * n[0] + cy * n[1]
    k0 = math.floor((c0 - r) / spacing)
    segs = []
    for k in range(k0, k0 + int(2 * r / spacing) + 3):
        off = k * spacing + phase
        px, py = n[0] * off, n[1] * off
        t = (cx - px) * d[0] + (cy - py) * d[1]
        segs.append([(px + d[0] * (t - r), py + d[1] * (t - r)), (px + d[0] * (t + r), py + d[1] * (t + r))])
    return _lines_clip(poly, segs)


def _rows(poly, course, length, stagger, angle):
    """Plank or brick joints: rows `course` apart, joints every `length` along a row,
    each row shifted by `stagger` feet; rows run along `angle` (0 or 90)."""
    minx, miny, maxx, maxy = poly.bounds
    segs = []
    if angle == 0:
        lo, hi, a0, a1 = miny, maxy, minx, maxx
    else:
        lo, hi, a0, a1 = minx, maxx, miny, maxy
    row = math.floor(lo / course)
    while row * course < hi:
        c0 = row * course
        off = (row * stagger) % length
        t = math.floor((a0 - off) / length) * length + off
        while t < a1:
            segs.append([(t, c0), (t, c0 + course)] if angle == 0 else [(c0, t), (c0 + course, t)])
            t += length
        row += 1
    return _parallel(poly, 0 if angle == 0 else 90, course) + _lines_clip(poly, segs)


def _dashes(segs, dash, gap):
    out = []
    for (x0, y0), (x1, y1) in segs:
        L = math.hypot(x1 - x0, y1 - y0)
        if L == 0:
            continue
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        start = (x0 * ux + y0 * uy) % (dash + gap)       # keep dashes aligned across zones
        t = -start
        while t < L:
            a, b = max(t, 0), min(t + dash, L)
            if b > a:
                out.append([(x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b)])
            t += dash + gap
    return out


def pattern(spec, poly, rng):
    """(line segments, dots as (x, y, r), triangles, tint grey or None) for a pattern spec, in feet."""
    kind, *a = spec
    lines, dots, tris, tint = [], [], [], None
    if kind == "planks":
        width, length, angle = a
        lines = _rows(poly, width, length, 1.375, angle)
    elif kind == "brick":
        course, length = a
        lines = _rows(poly, course, length, length / 2, 0)
    elif kind == "lines":
        lines = _parallel(poly, a[0], a[1])
    elif kind == "dashed":
        angle, spacing, dash, gap = a
        lines = _dashes(_parallel(poly, angle, spacing), dash, gap)
    elif kind == "double":
        angle, spacing, gap = a
        lines = _parallel(poly, angle, spacing) + _parallel(poly, angle, spacing, phase=gap)
    elif kind == "cross":
        lines = _parallel(poly, a[0], a[2]) + _parallel(poly, a[1], a[2])
    elif kind == "dots":
        spacing, r = a
        minx, miny, maxx, maxy = poly.bounds
        x = math.floor(minx / spacing) * spacing + spacing / 2
        while x < maxx:
            y = math.floor(miny / spacing) * spacing + spacing / 2
            while y < maxy:
                if poly.contains(Point(x, y)):
                    dots.append((x, y, r))
                y += spacing
            x += spacing
    elif kind == "stipple":
        density = a[0]
        minx, miny, maxx, maxy = poly.bounds
        area = (maxx - minx) * (maxy - miny)
        for _ in range(int(area * density)):
            pt = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if poly.contains(Point(pt)):
                dots.append((pt[0], pt[1], 0.05))
        for _ in range(int(area * 0.12)):
            cx, cy = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            ang = rng.uniform(0, 2 * math.pi)
            tri = [(cx + 0.28 * math.cos(ang + k * 2.094), cy + 0.28 * math.sin(ang + k * 2.094)) for k in range(3)]
            if poly.contains(Polygon(tri)):
                tris.append(tri)
    elif kind == "joints":
        lines = _parallel(poly, 0, a[0]) + _parallel(poly, 90, a[0])
    elif kind == "tint":
        tint = a[0]
    return lines, dots, tris, tint


# ---------------------------------------------------------------- drawing --
class Sheet:
    def __init__(self):
        self.fig = plt.figure(figsize=(SHEET_W, SHEET_H), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, SHEET_W)
        self.ax.set_ylim(0, SHEET_H)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.z = 1

    def nz(self):
        self.z += 1
        return self.z

    def lines_ft(self, segs, **kw):
        if segs:
            pts = [[S(*p) for p in s] for s in segs]
            self.ax.add_collection(LineCollection(pts, zorder=kw.pop("zorder", self.nz()), **kw))

    def poly_ft(self, coords, **kw):
        self.ax.add_patch(MplPolygon([S(*p) for p in coords], closed=True, zorder=kw.pop("zorder", self.nz()), **kw))

    def rect_ft(self, x, y, w, h, **kw):
        self.poly_ft([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], **kw)

    def text_ft(self, x, y, s, size=7, **kw):
        sx, sy = S(x, y)
        self.text(sx, sy, s, size, **kw)

    def text(self, x, y, s, size=7, weight="normal", ha="center", va="center", box=False, rotation=0, z=None):
        bbox = dict(boxstyle="square,pad=0.15", fc="white", ec="none") if box else None
        self.ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, family="DejaVu Sans",
                     bbox=bbox, rotation=rotation, zorder=z or self.nz())

    def save_array(self):
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        img = (0.299 * buf[..., 0] + 0.587 * buf[..., 1] + 0.114 * buf[..., 2]).astype(np.uint8)
        plt.close(self.fig)
        return img


def poly_parts(g):
    return [x for x in getattr(g, "geoms", [g]) if isinstance(x, Polygon) and not x.is_empty]


def draw_hatched(sh, poly, spec, rng, z, office):
    lines, dots, tris, tint = pattern(spec, poly, rng)
    if tint is not None:
        for part in poly_parts(poly):
            sh.poly_ft(list(part.exterior.coords), fc=str(tint), ec="none", zorder=z - 0.5)
    sh.lines_ft(lines, zorder=z, color=str(office["hatch_grey"]), linewidth=office["hatch_lw"])
    if dots:
        xs, ys = zip(*[S(x, y) for x, y, r in dots])
        size = [(2 * r * SCALE * 72) ** 2 for x, y, r in dots]
        sh.ax.scatter(xs, ys, s=size, c=str(office["hatch_grey"] - 0.15), marker="o", linewidths=0, zorder=z)
    for t in tris:
        sh.poly_ft(t, fill=False, ec="0.35", lw=0.3, zorder=z)


def draw_poche(sh, part, spec, z, office):
    """Fill one wall band with an office's poche spec."""
    lines, dots, tris, tint = pattern(spec, part, random.Random(0))
    sh.poly_ft(list(part.exterior.coords), fc=str(tint) if tint is not None else "white", ec="none", zorder=z)
    if lines:
        sh.lines_ft(lines, color="0.15", linewidth=0.35, zorder=z + 0.1)


def draw_walls(sh, plan, z, office):
    openings = unary_union([d.opening() for d in plan.doors])
    bands = {k: unary_union([w.band() for w in plan.walls if w.kind == k]).difference(openings)
             for k in ("EXT", "P3", "P2", "P1")}
    for kind in ("P1", "P2", "P3", "EXT"):
        spec = office["walls"]["CMU" if kind in ("EXT", "P3") else kind]
        for part in poly_parts(bands[kind]):
            draw_poche(sh, part, spec, z, office)
    allw = unary_union(list(bands.values()))
    rings = []
    for part in poly_parts(allw):
        rings.append(list(part.exterior.coords))
        rings += [list(r.coords) for r in part.interiors]
    sh.lines_ft([r for r in rings], color="black", linewidth=WALL_LW, zorder=z + 0.2)


def draw_doors(sh, plan, z):
    for d in plan.doors:
        w = d.wall
        h = THICKNESS[w.kind] / 2
        if w.horizontal:
            edges = [[(d.start, w.y1 + s * h), (d.start + d.width, w.y1 + s * h)] for s in (-1, 1)]
        else:
            edges = [[(w.x1 + s * h, d.start), (w.x1 + s * h, d.start + d.width)] for s in (-1, 1)]
        sh.lines_ft(edges, color="black", linewidth=EDGE_LW, zorder=z - 1)
        if d.kind in ("single", "pair"):
            leaves = []
            if d.kind == "single":
                leaves.append((d.start if d.hinge == 0 else d.start + d.width, d.width, 1 if d.hinge == 0 else -1))
            else:
                half = d.width / 2
                leaves += [(d.start, half, 1), (d.start + d.width, half, -1)]
            for hinge, length, toward in leaves:
                if w.horizontal:
                    fy = w.y1 + d.swing * h
                    sh.rect_ft(hinge - (0.15 if toward < 0 else 0), fy if d.swing > 0 else fy - length, 0.15, length,
                               fc="white", ec="black", lw=0.55, zorder=z)
                    a0 = 90 if d.swing > 0 else 270
                    t1 = 0 if toward > 0 else 180
                    cx, cy = S(hinge, fy)
                    theta = sorted([a0, t1 if not (a0 == 270 and t1 == 0) else 360])
                    sh.ax.add_patch(Arc((cx, cy), 2 * length * SCALE, 2 * length * SCALE, theta1=theta[0],
                                        theta2=theta[1], lw=0.35, color="black", zorder=z))
                else:
                    fx = w.x1 + d.swing * h
                    sh.rect_ft(fx if d.swing > 0 else fx - length, hinge - (0.15 if toward < 0 else 0), length, 0.15,
                               fc="white", ec="black", lw=0.55, zorder=z)
                    a0 = 0 if d.swing > 0 else 180
                    t1 = 90 if toward > 0 else 270
                    cx, cy = S(fx, hinge)
                    theta = sorted([a0, t1]) if not (a0 == 0 and t1 == 270) else [270, 360]
                    sh.ax.add_patch(Arc((cx, cy), 2 * length * SCALE, 2 * length * SCALE, theta1=theta[0],
                                        theta2=theta[1], lw=0.35, color="black", zorder=z))
        elif d.kind == "overhead_coiling":
            y = w.y1
            sh.lines_ft([[(d.start, y), (d.start + d.width, y)]], color="black", linewidth=0.5,
                        linestyles=(0, (4, 2)), zorder=z)
            sh.rect_ft(d.start - 0.5, y - h - 1.5, d.width + 1.0, 1.5, fill=False, ec="black", lw=0.4,
                       ls=(0, (4, 2)), zorder=z)
        elif d.kind == "auto_sliding":
            y = w.y1
            half = d.width / 2
            sh.lines_ft([[(d.start, y + 0.08), (d.start + half + 0.4, y + 0.08)],
                         [(d.start + half - 0.4, y - 0.08), (d.start + d.width, y - 0.08)]],
                        color="black", linewidth=0.9, zorder=z)
            for x0, dx in ((d.start + 0.5, -1), (d.start + d.width - 0.5, 1)):
                sh.ax.annotate("", xy=S(x0 + dx * 1.0, y + 0.7), xytext=S(x0, y + 0.7),
                               arrowprops=dict(arrowstyle="->", lw=0.4), zorder=z)


def draw_fixtures(sh, plan, z):
    W = dict(fc="white", ec="black", lw=FIX_LW, zorder=z)
    for g in plan.gondolas:
        L = g.section_ft * g.sections
        sh.rect_ft(g.x, g.y, 4, L, **W)
        sh.lines_ft([[(g.x, g.y + g.section_ft * k), (g.x + 4, g.y + g.section_ft * k)] for k in range(1, g.sections)],
                    color="black", linewidth=FIX_LW, zorder=z + 0.1)
        sh.lines_ft([[(g.x + 2, g.y), (g.x + 2, g.y + L)]], color="black", linewidth=0.3, zorder=z + 0.1)
        for flag, y0 in ((g.endcap_south, g.y - 1.5), (g.endcap_north, g.y + L)):
            if flag:
                sh.rect_ft(g.x, y0, 4, 1.5, **W)
                sh.lines_ft([[(g.x, y0), (g.x + 4, y0 + 1.5)]], color="black", linewidth=FIX_LW, zorder=z + 0.1)
    for c in plan.coolers:
        n = c.doors
        if c.facing == "S":
            sh.rect_ft(c.x, c.y, 2.5 * n, 3, **W)
            for i in range(n):
                x0 = c.x + 2.5 * i
                sh.rect_ft(x0 + 0.1, c.y, 2.3, 0.6, fc="white", ec="black", lw=0.4, zorder=z + 0.1)
                sh.lines_ft([[(x0 + 0.1, c.y + 0.6), (x0 + 2.4, c.y)]], color="black", linewidth=FIX_LW, zorder=z + 0.2)
        else:
            sh.rect_ft(c.x, c.y, 3, 2.5 * n, **W)
            for i in range(n):
                y0 = c.y + 2.5 * i
                sh.rect_ft(c.x + 3 - 0.6, y0 + 0.1, 0.6, 2.3, fc="white", ec="black", lw=0.4, zorder=z + 0.1)
                sh.lines_ft([[(c.x + 3 - 0.6, y0 + 0.1), (c.x + 3, y0 + 2.4)]], color="black", linewidth=FIX_LW,
                            zorder=z + 0.2)
    for x, y in plan.checkouts:
        sh.rect_ft(x, y + 3, 2.5, 9, **W)
        sh.lines_ft([[(x + 1.25, y + 3.4), (x + 1.25, y + 11.6)]], color="black", linewidth=0.3, zorder=z + 0.1)
        sh.rect_ft(x, y, 2.5, 3, **W)
        sh.lines_ft([[(x, y), (x + 2.5, y + 3)], [(x, y + 3), (x + 2.5, y)]], color="black", linewidth=0.3,
                    zorder=z + 0.1)
        sh.rect_ft(x + 2.5, y + 6, 1.2, 1.2, **W)
    for x, y in plan.self_checkouts:
        sh.rect_ft(x, y, 3, 3, **W)
        cx, cy = S(x + 1.5, y + 1.5)
        sh.ax.add_patch(Circle((cx, cy), 0.8 * SCALE, fill=False, ec="black", lw=0.4, zorder=z + 0.1))
    for x, y in plan.tables:
        sh.rect_ft(x, y, 5, 4, **W)
        sh.lines_ft([[(x, y), (x + 5, y + 4)], [(x, y + 4), (x + 5, y)]], color="black", linewidth=FIX_LW,
                    zorder=z + 0.1)
    for x, y in plan.columns:
        sh.rect_ft(x - 0.5, y - 0.5, 1, 1, fc="black", ec="black", lw=0.3, zorder=z + 0.5)


def draw_annotations(sh, plan, z):
    for code, x, y in plan.finish_tags:
        sx, sy = S(x, y)
        sh.ax.text(sx, sy, code, fontsize=6, ha="center", va="center", family="DejaVu Sans", zorder=z,
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.4))
    for code, x, y in plan.wall_tags:
        sx, sy = S(x, y)
        sh.ax.add_patch(RegularPolygon((sx, sy), 6, radius=0.13, orientation=math.pi / 6, fc="white", ec="black",
                                       lw=0.45, zorder=z))
        sh.ax.text(sx, sy, code, fontsize=5, ha="center", va="center", family="DejaVu Sans", zorder=z + 0.1)
    for name, num, x, y in plan.rooms:
        sx, sy = S(x, y)
        sh.ax.text(sx, sy + 0.07, name, fontsize=7.5, fontweight="bold", ha="center", va="center",
                   family="DejaVu Sans", zorder=z, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        sh.ax.text(sx, sy - 0.1, num, fontsize=6, ha="center", va="center", family="DejaVu Sans", zorder=z,
                   bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="black", lw=0.4))
    # door tags
    for i, d in enumerate(plan.doors):
        w = d.wall
        mid = d.start + d.width / 2
        off = -(THICKNESS[w.kind] / 2 + 2.2) * (1 if d.kind != "single" else d.swing)
        x, y = (mid, w.y1 + off) if w.horizontal else (w.x1 + off, mid)
        sx, sy = S(x, y)
        sh.ax.add_patch(Circle((sx, sy), 0.075, fc="white", ec="black", lw=0.35, zorder=z))
        sh.ax.text(sx, sy, f"{100 + i}", fontsize=3.6, ha="center", va="center", family="DejaVu Sans",
                   zorder=z + 0.1)


def draw_grid_and_dims(sh, plan, z):
    W, D = plan.width, plan.depth
    xs = dict(zip("ABCDEFG", plan.grid_x))
    ys = dict(zip("123456", plan.grid_y))
    dash = (0, (12, 3, 2, 3))
    for k, x in xs.items():
        sh.lines_ft([[(x, -9), (x, D + 9)]], color="0.25", linewidth=0.25, linestyles=dash, zorder=z)
        for y in (-11, D + 11):
            sx, sy = S(x, y)
            sh.ax.add_patch(Circle((sx, sy), 0.17, fc="white", ec="black", lw=0.5, zorder=z + 1))
            sh.text(sx, sy, k, 8, z=z + 2)
    for k, y in ys.items():
        sh.lines_ft([[(-9, y), (W + 9, y)]], color="0.25", linewidth=0.25, linestyles=dash, zorder=z)
        for x in (-11, W + 11):
            sx, sy = S(x, y)
            sh.ax.add_patch(Circle((sx, sy), 0.17, fc="white", ec="black", lw=0.5, zorder=z + 1))
            sh.text(sx, sy, k, 8, z=z + 2)

    def dim(p1, p2, offset, label, horizontal):
        if horizontal:
            (x1, y0), (x2, _) = p1, p2
            y = y0 + offset
            sh.lines_ft([[(x1, y), (x2, y)], [(x1, y0 + (2 if offset > 0 else -2)), (x1, y + (1 if offset > 0 else -1))],
                         [(x2, y0 + (2 if offset > 0 else -2)), (x2, y + (1 if offset > 0 else -1))]],
                        color="black", linewidth=0.3, zorder=z)
            for x in (x1, x2):
                sh.lines_ft([[(x - 0.7, y - 0.7), (x + 0.7, y + 0.7)]], color="black", linewidth=0.7, zorder=z)
            sh.text_ft((x1 + x2) / 2, y + 1.1, label, 7, z=z)
        else:
            (x0, y1), (_, y2) = p1, p2
            x = x0 + offset
            sh.lines_ft([[(x, y1), (x, y2)], [(x0 + (2 if offset > 0 else -2), y1), (x + (1 if offset > 0 else -1), y1)],
                         [(x0 + (2 if offset > 0 else -2), y2), (x + (1 if offset > 0 else -1), y2)]],
                        color="black", linewidth=0.3, zorder=z)
            for yy in (y1, y2):
                sh.lines_ft([[(x - 0.7, yy - 0.7), (x + 0.7, yy + 0.7)]], color="black", linewidth=0.7, zorder=z)
            sx, sy = S(x - 1.1, (y1 + y2) / 2)
            sh.text(sx, sy, label, 7, rotation=90, z=z)

    dim((0, 0), (W, 0), -6.5, ft_in(W), True)
    dim((0, D), (W, D), 6.5, ft_in(W), True)
    dim((0, 0), (0, D), -6.5, ft_in(D), False)
    dim((W, 0), (W, D), 6.5, ft_in(D), False)
    gx = [0] + list(plan.grid_x[1:-1]) + [W]
    for a, b in zip(gx, gx[1:]):
        dim((a, 0), (b, 0), -3.2, ft_in(b - a), True)
    gy = [0] + list(plan.grid_y[1:-1]) + [D]
    for a, b in zip(gy, gy[1:]):
        dim((0, a), (0, b), -3.2, ft_in(b - a), False)


def draw_scale_bar(sh, x, y, z):
    """Graphic scale, 0-32 feet, alternating filled segments; (x, y) in plan feet."""
    marks = [0, 4, 8, 16, 32]
    for k, (a, b) in enumerate(zip(marks, marks[1:])):
        sh.rect_ft(x + a, y, b - a, 1.0, fc="black" if k % 2 == 0 else "white", ec="black", lw=0.6, zorder=z)
    for m in marks:
        sh.text_ft(x + m, y - 1.6, str(m), 6.5, z=z)
    sh.text_ft(x + 38, y + 0.5, "FEET", 6.5, z=z)


def legend_swatch(sh, spec, x, y, rng, z, office):
    w, h = 0.9 / SCALE, 0.45 / SCALE   # swatch size in plan feet
    fx, fy = (x - OX) / SCALE, (y - OY) / SCALE
    poly = box(fx, fy, fx + w, fy + h)
    draw_hatched(sh, poly, spec, rng, z, office)
    sh.ax.add_patch(Rectangle((x, y), 0.9, 0.45, fill=False, ec="black", lw=0.5, zorder=z + 0.5))


def draw_legend(sh, plan_rng, z, office):
    x0, top = 24.4, 21.6
    sh.ax.add_patch(Rectangle((x0 - 0.2, 2.2), 7.4, top - 2.0, fill=False, ec="black", lw=0.6, zorder=z))
    sh.text(x0, top - 0.25, "LEGEND", 11, weight="bold", ha="left", z=z)
    y = top - 0.75
    sh.text(x0, y, "FLOOR FINISHES", 8, weight="bold", ha="left", z=z)
    finishes = [("LVT-1", "LUXURY VINYL PLANK"), ("PC-1", "POLISHED CONCRETE"),
                ("SC-1", "SEALED CONCRETE"), ("EP-1", "RESINOUS EPOXY FLOORING"),
                ("VCT-1", "VINYL COMPOSITION TILE"), ("CT-1", "PORCELAIN TILE"),
                ("CPT-1", "CARPET TILE"), ("WOM-1", "RECESSED WALK-OFF MAT")]
    y -= 0.35
    for code, desc in finishes:
        y -= 0.62
        legend_swatch(sh, office["finishes"][code], x0, y, plan_rng, z, office)
        sh.text(x0 + 1.1, y + 0.3, code, 7.5, weight="bold", ha="left", z=z)
        sh.text(x0 + 1.1, y + 0.1, desc, 6, ha="left", z=z)
    y -= 0.55
    sh.text(x0, y, "PARTITION TYPES", 8, weight="bold", ha="left", z=z)
    walls = [("EXT", "8\" CMU EXTERIOR WALL (BUILDING PERIMETER)"), ("P3", "8\" CMU INTERIOR PARTITION"),
             ("P2", "6\" MTL STUD, (1) LAYER 5/8\" TYPE X GWB EA SIDE, 1 HR"),
             ("P1", "3-5/8\" MTL STUD, (1) LAYER 5/8\" GWB EA SIDE")]
    y -= 0.15
    for code, desc in walls:
        y -= 0.5
        t = THICKNESS[code] * SCALE
        fpoly = box((x0 - OX) / SCALE, (y + 0.2 - t / 2 - OY) / SCALE, (x0 + 0.9 - OX) / SCALE,
                    (y + 0.2 + t / 2 - OY) / SCALE)
        draw_poche(sh, fpoly, office["walls"]["CMU" if code in ("EXT", "P3") else code], z, office)
        sh.ax.add_patch(Rectangle((x0, y + 0.2 - t / 2), 0.9, t, fill=False, ec="black", lw=WALL_LW, zorder=z + 0.3))
        sx, sy = x0 + 1.25, y + 0.2
        sh.ax.add_patch(RegularPolygon((sx, sy), 6, radius=0.13, orientation=math.pi / 6, fc="white", ec="black",
                                       lw=0.45, zorder=z))
        sh.text(sx, sy, code, 5, z=z + 0.1)
        sh.text(x0 + 1.5, y + 0.2, desc, 6, ha="left", z=z)
    y -= 0.55
    sh.text(x0, y, "FIXTURES (BY OWNER, INSTALLED BY GC)", 8, weight="bold", ha="left", z=z)
    y -= 0.15
    items = ["GONDOLA RUN, 4'-0\" DEEP, 3'-0\" OR 4'-0\" SECTIONS AS DRAWN", "GONDOLA ENDCAP", "CHECKOUT STAND",
             "SELF-CHECKOUT KIOSK", "REACH-IN COOLER / FREEZER DOOR", "PRODUCE DISPLAY TABLE, 5'-0\" x 4'-0\"",
             "STRUCTURAL COLUMN"]
    for i, desc in enumerate(items):
        y -= 0.55
        fx, fy = (x0 + 0.1 - OX) / SCALE, (y + 0.05 - OY) / SCALE
        W = dict(fc="white", ec="black", lw=FIX_LW, zorder=z)
        if i == 0:
            sh.rect_ft(fx, fy + 0.5, 6, 2.5, **W)
            sh.lines_ft([[(fx + 4, fy + 0.5), (fx + 4, fy + 3)], [(fx, fy + 1.75), (fx + 6, fy + 1.75)]],
                        color="black", linewidth=FIX_LW, zorder=z + 0.1)
        elif i == 1:
            sh.rect_ft(fx + 1, fy + 0.5, 1.5, 3, **W)
            sh.lines_ft([[(fx + 1, fy + 0.5), (fx + 2.5, fy + 3.5)]], color="black", linewidth=0.3, zorder=z + 0.1)
        elif i == 2:
            sh.rect_ft(fx + 0.5, fy + 1, 6, 2.5, **W)
            sh.rect_ft(fx - 1.5, fy + 1, 2, 2.5, **W)
            sh.lines_ft([[(fx - 1.5, fy + 1), (fx + 0.5, fy + 3.5)], [(fx - 1.5, fy + 3.5), (fx + 0.5, fy + 1)]],
                        color="black", linewidth=0.3, zorder=z + 0.1)
            sh.rect_ft(fx + 2.5, fy - 0.2, 1.2, 1.2, **W)
        elif i == 3:
            sh.rect_ft(fx + 1, fy + 0.5, 3, 3, **W)
            cx, cy = S(fx + 2.5, fy + 2)
            sh.ax.add_patch(Circle((cx, cy), 0.8 * SCALE, fill=False, ec="black", lw=0.4, zorder=z + 0.1))
        elif i == 4:
            sh.rect_ft(fx + 0.5, fy + 1, 2.3, 0.6, fc="white", ec="black", lw=0.4, zorder=z + 0.1)
            sh.lines_ft([[(fx + 0.5, fy + 1.6), (fx + 2.8, fy + 1)]], color="black", linewidth=FIX_LW, zorder=z + 0.2)
            sh.rect_ft(fx + 0.4, fy + 1, 2.5, 3, fill=False, ec="black", lw=FIX_LW, zorder=z)
        elif i == 5:
            sh.rect_ft(fx + 0.5, fy + 0.2, 5, 4, **W)
            sh.lines_ft([[(fx + 0.5, fy + 0.2), (fx + 5.5, fy + 4.2)], [(fx + 0.5, fy + 4.2), (fx + 5.5, fy + 0.2)]],
                        color="black", linewidth=0.3, zorder=z + 0.1)
        else:
            sh.rect_ft(fx + 2, fy + 1.5, 1, 1, fc="black", ec="black", lw=0.3, zorder=z)
        sh.text(x0 + 1.1, y + 0.25, desc, 6, ha="left", z=z)
    return y


def draw_notes(sh, y, z):
    x0 = 24.4
    y -= 0.6
    sh.text(x0, y, "GENERAL NOTES", 8, weight="bold", ha="left", z=z)
    notes = ["1. DIMENSIONS ARE TO OUTSIDE FACE OF EXTERIOR WALL OR TO",
             "   COLUMN GRID LINE UNLESS NOTED OTHERWISE.",
             "2. FLOOR FINISHES EXTEND UNDER ALL FIXTURES AND EQUIPMENT.",
             "3. FINISH TRANSITIONS OCCUR AT LINES SHOWN AND AT DOOR",
             "   THRESHOLDS. RACETRACK AISLE CORNERS ARE RADIUSED.",
             "4. FIXTURE LAYOUT PER OWNER'S PLANOGRAM, ISSUE 3.",
             "5. VERIFY ALL CONDITIONS IN FIELD."]
    for line in notes:
        y -= 0.24
        sh.text(x0, y, line, 6, ha="left", z=z)


def draw_title_block(sh, z, plan, number="A101", title="FLOOR PLAN", subtitle="FIXTURE & FINISH", page=2,
                     scale="1/8\" = 1'-0\"", bar=True):
    x0, x1 = 32.0, 35.55
    ax = sh.ax
    ax.add_patch(Rectangle((x0, 0.45), x1 - x0, 23.1, fill=False, ec="black", lw=1.0, zorder=z))
    rows = [21.4, 18.6, 15.6, 12.4, 6.4, 3.4]
    for r in rows:
        ax.plot([x0, x1], [r, r], color="black", lw=0.6, zorder=z)
    cx = (x0 + x1) / 2
    words = plan.office_name.split()
    sh.text(cx, 22.8, words[0], 15, weight="bold", z=z)
    sh.text(cx, 22.3, " ".join(words[1:]), 9, z=z)
    sh.text(cx, 21.85, "1840 W. COMMERCE ST, SUITE 300", 6, z=z)
    sh.text(cx, 20.8, "CEDAR VALLEY MARKET", 10, weight="bold", z=z)
    sh.text(cx, 20.35, f"STORE #{plan.store}", 9, weight="bold", z=z)
    sh.text(cx, 19.5, plan.city, 6.5, z=z)
    sh.text(cx, 18.95, "AS-BUILT CONDITIONS", 6.5, z=z)
    sh.text(x0 + 0.15, 18.3, "REVISIONS", 6.5, weight="bold", ha="left", z=z)
    revs = ["1  03/14/2019  BID SET", "2  05/02/2019  PERMIT SET", "3  09/30/2019  AS-BUILT"]
    for i, r in enumerate(revs):
        sh.text(x0 + 0.15, 17.9 - 0.3 * i, r, 6, ha="left", z=z)
    sh.text(x0 + 0.15, 15.3, "DRAWN: KRL    CHECKED: JMP", 6, ha="left", z=z)
    sh.text(x0 + 0.15, 15.0, f"PROJECT NO: 19-{plan.store}", 6, ha="left", z=z)
    sh.text(x0 + 0.15, 14.7, "DATE: 09/30/2019", 6, ha="left", z=z)
    sh.text(x0 + 0.15, 14.1, f"SCALE: {scale}", 7, weight="bold", ha="left", z=z)
    if bar:
        # one-inch verification bar
        ax.add_patch(Rectangle((x0 + 0.6, 13.15), 1.0, 0.09, fc="black", ec="black", lw=0, zorder=z))
        for t in (0.0, 1.0):
            ax.plot([x0 + 0.6 + t, x0 + 0.6 + t], [13.05, 13.34], color="black", lw=0.6, zorder=z)
        sh.text(x0 + 1.1, 12.85, "IF THIS BAR DOES NOT MEASURE 1\"", 5.5, z=z)
        sh.text(x0 + 1.1, 12.65, "THE DRAWING IS NOT TO SCALE", 5.5, z=z)
    sh.text(cx, 9.6, title, 13, weight="bold", z=z)
    sh.text(cx, 9.0, subtitle, 8, z=z)
    sh.text(cx, 4.9, number, 30, weight="bold", z=z)
    sh.text(cx, 1.6, f"SHEET {page} OF 14", 6, z=z)


def render(plan, seed=7, office=None):
    from offices import OFFICES
    office = office or OFFICES["NORTHLINE"]
    plan.office_name = office["name"]
    rng = random.Random(seed)
    sh = Sheet()
    ax = sh.ax
    ax.add_patch(Rectangle((0.45, 0.45), SHEET_W - 0.9, SHEET_H - 0.9, fill=False, ec="black", lw=1.2, zorder=1))
    zfloor, zgrid, zwall, zfix, zdoor, zann = 10, 20, 30, 40, 50, 60
    walls = plan.wall_footprint()
    for zone in plan.zones:
        for part in poly_parts(zone.shape.difference(walls)):
            draw_hatched(sh, part, office["finishes"][zone.finish], rng, zfloor, office)
    segs = []
    for t in plan.transitions:
        for g in getattr(t, "geoms", [t]):
            segs.append(list(g.coords))
    sh.lines_ft(segs, color="black", linewidth=EDGE_LW, zorder=zfloor + 1)
    draw_grid_and_dims(sh, plan, zgrid)
    draw_walls(sh, plan, zwall, office)
    draw_fixtures(sh, plan, zfix)
    draw_doors(sh, plan, zdoor)
    draw_annotations(sh, plan, zann)
    # plan title and north arrow
    sx, sy = S(0, -15.5)
    sh.ax.add_patch(Circle((sx + 0.3, sy), 0.25, fc="white", ec="black", lw=0.8, zorder=zann))
    sh.text(sx + 0.3, sy + 0.08, "1", 9, weight="bold", z=zann + 1)
    sh.text(sx + 0.3, sy - 0.12, "A101", 5, z=zann + 1)
    sh.text(sx + 0.75, sy + 0.05, "FLOOR PLAN - FIXTURE & FINISH", 12, weight="bold", ha="left", z=zann)
    sh.text(sx + 0.75, sy - 0.25, "SCALE: 1/8\" = 1'-0\"", 8, ha="left", z=zann)
    draw_scale_bar(sh, plan.width - 62, -16.5, zann)
    nx, ny = S(plan.width - 10, -14)
    sh.ax.add_patch(Circle((nx, ny), 0.3, fill=False, ec="black", lw=0.6, zorder=zann))
    sh.ax.add_patch(MplPolygon([(nx, ny + 0.38), (nx - 0.12, ny - 0.2), (nx, ny - 0.08), (nx + 0.12, ny - 0.2)],
                               closed=True, fc="black", zorder=zann))
    sh.text(nx, ny + 0.52, "N", 9, weight="bold", z=zann)
    y = draw_legend(sh, random.Random(seed + 1), zann, office)
    draw_notes(sh, y, zann)
    draw_title_block(sh, zann, plan)
    return sh.save_array()
