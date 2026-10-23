"""Render the store #0417 floor plan as a 36" x 24" CAD-style plot."""

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

from plan import THICKNESS, E, Light  # noqa: E402

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


def pattern(code, poly, rng):
    """Return (line segments, dots, triangles) for a finish hatch, in feet."""
    lines, dots, tris = [], [], []
    if code == "LVT-1":
        w = 7 / 12
        lines += _parallel(poly, 0, w)
        minx, miny, maxx, maxy = poly.bounds
        row = math.floor(miny / w)
        while row * w < maxy:
            y0 = row * w
            off = (row * 1.375) % 4
            x = math.floor((minx - off) / 4) * 4 + off
            segs = []
            while x < maxx:
                segs.append([(x, y0), (x, y0 + w)])
                x += 4
            lines += _lines_clip(poly, segs) if segs else []
            row += 1
    elif code == "PC-1":
        lines += _parallel(poly, 0, 10) + _parallel(poly, 90, 10)
    elif code == "SC-1":
        minx, miny, maxx, maxy = poly.bounds
        area = (maxx - minx) * (maxy - miny)
        for _ in range(int(area * 0.9)):
            pt = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if poly.contains(Point(pt)):
                dots.append(pt)
        for _ in range(int(area * 0.12)):
            cx, cy = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            a = rng.uniform(0, 2 * math.pi)
            tri = [(cx + 0.28 * math.cos(a + k * 2.094), cy + 0.28 * math.sin(a + k * 2.094)) for k in range(3)]
            if poly.contains(Polygon(tri)):
                tris.append(tri)
    elif code == "EP-1":
        lines += _parallel(poly, 45, 1.0)
    elif code == "VCT-1":
        lines += _parallel(poly, 0, 1.0) + _parallel(poly, 90, 1.0)
    elif code == "CT-1":
        lines += _parallel(poly, 45, 0.75) + _parallel(poly, 135, 0.75)
    elif code == "CPT-1":
        lines += _parallel(poly, 135, 0.5)
    elif code == "WOM-1":
        lines += _parallel(poly, 0, 0.25)
    return lines, dots, tris


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


def draw_hatched(sh, poly, code, rng, z):
    lines, dots, tris = pattern(code, poly, rng)
    sh.lines_ft(lines, zorder=z, **HATCH)
    if dots:
        xs, ys = zip(*[S(*p) for p in dots])
        sh.ax.scatter(xs, ys, s=0.35, c="0.3", marker="o", linewidths=0, zorder=z)
    for t in tris:
        sh.poly_ft(t, fill=False, ec="0.35", lw=0.3, zorder=z)


def draw_walls(sh, plan, z):
    openings = unary_union([d.opening() for d in plan.doors])
    bands = {k: unary_union([w.band() for w in plan.walls if w.kind == k]).difference(openings)
             for k in ("EXT", "P3", "P2", "P1")}
    for kind in ("P1", "P2", "P3", "EXT"):
        for part in poly_parts(bands[kind]):
            fc = {"P1": "0.88", "P2": "0.62", "P3": "white", "EXT": "white"}[kind]
            sh.poly_ft(list(part.exterior.coords), fc=fc, ec="none", zorder=z)
            if kind in ("EXT", "P3"):
                sh.lines_ft(_parallel(part, 45, 0.32), color="0.15", linewidth=0.35, zorder=z + 0.1)
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


def draw_grid_and_dims(sh, z):
    xs = {"A": E, "B": 44, "C": 88, "D": 132, "E": 160 - E}
    ys = {"1": E, "2": 27.5, "3": 53.5, "4": 80, "5": 110 - E}
    dash = (0, (12, 3, 2, 3))
    for k, x in xs.items():
        sh.lines_ft([[(x, -9), (x, 119)]], color="0.25", linewidth=0.25, linestyles=dash, zorder=z)
        for y in (-11, 121):
            sx, sy = S(x, y)
            sh.ax.add_patch(Circle((sx, sy), 0.17, fc="white", ec="black", lw=0.5, zorder=z + 1))
            sh.text(sx, sy, k, 8, z=z + 2)
    for k, y in ys.items():
        sh.lines_ft([[(-9, y), (169, y)]], color="0.25", linewidth=0.25, linestyles=dash, zorder=z)
        for x in (-11, 171):
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

    dim((0, 0), (160, 0), -6.5, "160'-0\"", True)
    dim((0, 110), (160, 110), 6.5, "160'-0\"", True)
    dim((0, 0), (0, 110), -6.5, "110'-0\"", False)
    dim((160, 0), (160, 110), 6.5, "110'-0\"", False)
    gx = [0] + [44, 88, 132] + [160]
    for a, b in zip(gx, gx[1:]):
        dim((a, 0), (b, 0), -3.2, ft_in(b - a), True)
    gy = [0, 27.5, 53.5, 80, 110]
    for a, b in zip(gy, gy[1:]):
        dim((0, a), (0, b), -3.2, ft_in(b - a), False)


def legend_swatch(sh, code, x, y, rng, z):
    w, h = 0.9 / SCALE, 0.45 / SCALE   # swatch size in plan feet
    fx, fy = (x - OX) / SCALE, (y - OY) / SCALE
    poly = box(fx, fy, fx + w, fy + h)
    draw_hatched(sh, poly, code, rng, z)
    sh.ax.add_patch(Rectangle((x, y), 0.9, 0.45, fill=False, ec="black", lw=0.5, zorder=z + 0.5))


def draw_legend(sh, plan_rng, z):
    x0, top = 24.4, 21.6
    sh.ax.add_patch(Rectangle((x0 - 0.2, 2.2), 7.4, top - 2.0, fill=False, ec="black", lw=0.6, zorder=z))
    sh.text(x0, top - 0.25, "LEGEND", 11, weight="bold", ha="left", z=z)
    y = top - 0.75
    sh.text(x0, y, "FLOOR FINISHES", 8, weight="bold", ha="left", z=z)
    finishes = [("LVT-1", "LUXURY VINYL PLANK, 7\" x 48\""), ("PC-1", "POLISHED CONCRETE, SAWCUT JOINTS 10'-0\" O.C."),
                ("SC-1", "SEALED CONCRETE"), ("EP-1", "RESINOUS EPOXY FLOORING"),
                ("VCT-1", "VINYL COMPOSITION TILE, 12\" x 12\""), ("CT-1", "PORCELAIN TILE, 9\" x 9\" DIAGONAL"),
                ("CPT-1", "CARPET TILE, ASHLAR INSTALL"), ("WOM-1", "RECESSED WALK-OFF MAT")]
    y -= 0.35
    for code, desc in finishes:
        y -= 0.62
        legend_swatch(sh, code, x0, y, plan_rng, z)
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
        rect = box(x0, y + 0.2 - t / 2, x0 + 0.9, y + 0.2 + t / 2)
        fc = {"P2": "0.62", "P1": "0.88"}.get(code, "white")
        sh.ax.add_patch(Rectangle((x0, y + 0.2 - t / 2), 0.9, t, fc=fc, ec="black", lw=WALL_LW, zorder=z))
        if code in ("EXT", "P3"):
            fpoly = box((x0 - OX) / SCALE, (y + 0.2 - t / 2 - OY) / SCALE, (x0 + 0.9 - OX) / SCALE,
                        (y + 0.2 + t / 2 - OY) / SCALE)
            sh.lines_ft(_parallel(fpoly, 45, 0.32), color="0.15", linewidth=0.35, zorder=z + 0.1)
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


def draw_title_block(sh, z, number="A101", title="FLOOR PLAN", subtitle="FIXTURE & FINISH", page=2,
                     scale="1/8\" = 1'-0\"", bar=True):
    x0, x1 = 32.0, 35.55
    ax = sh.ax
    ax.add_patch(Rectangle((x0, 0.45), x1 - x0, 23.1, fill=False, ec="black", lw=1.0, zorder=z))
    rows = [21.4, 18.6, 15.6, 12.4, 6.4, 3.4]
    for r in rows:
        ax.plot([x0, x1], [r, r], color="black", lw=0.6, zorder=z)
    cx = (x0 + x1) / 2
    sh.text(cx, 22.8, "NORTHLINE", 15, weight="bold", z=z)
    sh.text(cx, 22.3, "RETAIL ARCHITECTS", 9, z=z)
    sh.text(cx, 21.85, "1840 W. COMMERCE ST, SUITE 300", 6, z=z)
    sh.text(cx, 20.8, "CEDAR VALLEY MARKET", 10, weight="bold", z=z)
    sh.text(cx, 20.35, "STORE #0417 REMODEL", 9, weight="bold", z=z)
    sh.text(cx, 19.8, "2215 LAKESHORE PKWY", 6.5, z=z)
    sh.text(cx, 19.5, "LAKEPORT", 6.5, z=z)
    sh.text(cx, 18.95, "AS-BUILT CONDITIONS", 6.5, z=z)
    sh.text(x0 + 0.15, 18.3, "REVISIONS", 6.5, weight="bold", ha="left", z=z)
    revs = ["1  03/14/2019  BID SET", "2  05/02/2019  PERMIT SET", "3  09/30/2019  AS-BUILT"]
    for i, r in enumerate(revs):
        sh.text(x0 + 0.15, 17.9 - 0.3 * i, r, 6, ha="left", z=z)
    sh.text(x0 + 0.15, 15.3, "DRAWN: KRL    CHECKED: JMP", 6, ha="left", z=z)
    sh.text(x0 + 0.15, 15.0, "PROJECT NO: 19-0417", 6, ha="left", z=z)
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


def render(plan, seed=7):
    rng = random.Random(seed)
    sh = Sheet()
    ax = sh.ax
    ax.add_patch(Rectangle((0.45, 0.45), SHEET_W - 0.9, SHEET_H - 0.9, fill=False, ec="black", lw=1.2, zorder=1))
    zfloor, zgrid, zwall, zfix, zdoor, zann = 10, 20, 30, 40, 50, 60
    walls = plan.wall_footprint()
    for zone in plan.zones:
        for part in poly_parts(zone.shape.difference(walls)):
            draw_hatched(sh, part, zone.finish, rng, zfloor)
    segs = []
    for t in plan.transitions:
        for g in getattr(t, "geoms", [t]):
            segs.append(list(g.coords))
    sh.lines_ft(segs, color="black", linewidth=EDGE_LW, zorder=zfloor + 1)
    draw_grid_and_dims(sh, zgrid)
    draw_walls(sh, plan, zwall)
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
    nx, ny = S(150, -14)
    sh.ax.add_patch(Circle((nx, ny), 0.3, fill=False, ec="black", lw=0.6, zorder=zann))
    sh.ax.add_patch(MplPolygon([(nx, ny + 0.38), (nx - 0.12, ny - 0.2), (nx, ny - 0.08), (nx + 0.12, ny - 0.2)],
                               closed=True, fc="black", zorder=zann))
    sh.text(nx, ny + 0.52, "N", 9, weight="bold", z=zann)
    y = draw_legend(sh, random.Random(seed + 1), zann)
    draw_notes(sh, y, zann)
    draw_title_block(sh, zann)
    return sh.save_array()


def draw_plan_title(sh, number, title, z):
    sx, sy = S(0, -15.5)
    sh.ax.add_patch(Circle((sx + 0.3, sy), 0.25, fc="white", ec="black", lw=0.8, zorder=z))
    sh.text(sx + 0.3, sy + 0.08, "1", 9, weight="bold", z=z + 1)
    sh.text(sx + 0.3, sy - 0.12, number, 5, z=z + 1)
    sh.text(sx + 0.75, sy + 0.05, title, 12, weight="bold", ha="left", z=z)
    sh.text(sx + 0.75, sy - 0.25, "SCALE: 1/8\" = 1'-0\"", 8, ha="left", z=z)
    nx, ny = S(150, -14)
    sh.ax.add_patch(Circle((nx, ny), 0.3, fill=False, ec="black", lw=0.6, zorder=z))
    sh.ax.add_patch(MplPolygon([(nx, ny + 0.38), (nx - 0.12, ny - 0.2), (nx, ny - 0.08), (nx + 0.12, ny - 0.2)],
                               closed=True, fc="black", zorder=z))
    sh.text(nx, ny + 0.52, "N", 9, weight="bold", z=z)


CEILING_DESC = {"ACT-1": "2'x2' ACOUSTICAL CEILING TILE, 15/16\" GRID", "ACT-2": "2'x4' ACOUSTICAL CEILING TILE, 15/16\" GRID",
                "GYP": "5/8\" GYPSUM BOARD CEILING, PAINTED", "OPEN": "OPEN TO STRUCTURE, PAINT EXPOSED DECK"}
LIGHT_DESC = [("L1", "2'x4' LED TROFFER, RECESSED"), ("L1E", "L1 WITH EMERGENCY BATTERY PACK"),
              ("L2", "2'x2' LED TROFFER, RECESSED"), ("L2E", "L2 WITH EMERGENCY BATTERY PACK"),
              ("L3", "LINEAR LED PENDANT, CONTINUOUS ROW, 4'-0\" MODULES"), ("L4", "6\" LED DOWNLIGHT"),
              ("L5", "LED HIGH-BAY, 24,000 LM"), ("X1", "LED EXIT SIGN, UNIVERSAL MOUNT")]


def ceiling_pattern(code, poly, rng):
    if code == "ACT-1":
        return _parallel(poly, 0, 2.0) + _parallel(poly, 90, 2.0), []
    if code == "ACT-2":
        return _parallel(poly, 0, 4.0) + _parallel(poly, 90, 2.0), []
    if code == "GYP":
        minx, miny, maxx, maxy = poly.bounds
        dots = []
        for _ in range(int((maxx - minx) * (maxy - miny) * 0.35)):
            pt = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if poly.contains(Point(pt)):
                dots.append(pt)
        return [], dots
    return [], []


def draw_light(sh, l, z):
    W = dict(fc="white", ec="black", lw=0.45, zorder=z)
    if l.kind in ("L1", "L1E", "L2", "L2E"):
        w, h = (2, 4) if l.kind.startswith("L1") else (2, 2)
        if l.rotated:
            w, h = h, w
        sh.rect_ft(l.x, l.y, w, h, **W)
        lens = dict(fc="black" if l.kind.endswith("E") else "white", ec="black", lw=0.4, zorder=z + 0.1)
        sh.rect_ft(l.x + 0.35, l.y + 0.35, w - 0.7, h - 0.7, **lens)
    elif l.kind == "L4":
        cx, cy = S(l.x, l.y)
        sh.ax.add_patch(Circle((cx, cy), 0.45 * SCALE, fc="white", ec="black", lw=0.45, zorder=z))
        sh.ax.add_patch(Circle((cx, cy), 0.12 * SCALE, fc="black", ec="black", lw=0.2, zorder=z + 0.1))
    elif l.kind == "L5":
        cx, cy = S(l.x, l.y)
        sh.ax.add_patch(Circle((cx, cy), 0.9 * SCALE, fc="white", ec="black", lw=0.45, zorder=z))
        sh.ax.add_patch(Circle((cx, cy), 0.45 * SCALE, fc="white", ec="black", lw=0.45, zorder=z + 0.1))
    elif l.kind == "X1":
        sh.rect_ft(l.x - 0.7, l.y - 0.3, 1.4, 0.6, **W)
        sh.rect_ft(l.x - 0.7, l.y - 0.3, 0.7, 0.6, fc="black", ec="black", lw=0.3, zorder=z + 0.1)


def render_rcp(plan, seed=8):
    rng = random.Random(seed)
    sh = Sheet()
    ax = sh.ax
    ax.add_patch(Rectangle((0.45, 0.45), SHEET_W - 0.9, SHEET_H - 0.9, fill=False, ec="black", lw=1.2, zorder=1))
    zclg, zgrid, zwall, zlight, zann = 10, 20, 30, 40, 60
    walls = plan.wall_footprint()
    inside = box(2 * E, 2 * E, 160 - 2 * E, 110 - 2 * E).difference(walls)
    for poly in poly_parts(inside):
        numbers = [num for _, num, x, y in plan.rooms if poly.contains(Point(x, y))]
        if not numbers:
            continue
        code = plan.schedule[numbers[0]]["ceiling"]
        lines, dots = ceiling_pattern(code, poly, rng)
        sh.lines_ft(lines, zorder=zclg, **HATCH)
        if dots:
            xs, ys = zip(*[S(*pt) for pt in dots])
            ax.scatter(xs, ys, s=0.35, c="0.3", marker="o", linewidths=0, zorder=zclg)
    draw_grid_and_dims(sh, zgrid)
    draw_walls(sh, plan, zwall)
    # door heads close each opening on the ceiling plan
    heads = []
    for d in plan.doors:
        w = d.wall
        if w.horizontal:
            heads.append([(d.start, w.y1), (d.start + d.width, w.y1)])
        else:
            heads.append([(w.x1, d.start), (w.x1, d.start + d.width)])
    sh.lines_ft(heads, color="black", linewidth=EDGE_LW, zorder=zwall + 1)
    for a, b in plan.l3_rows:
        sh.lines_ft([[a, b]], color="black", linewidth=2.2, zorder=zlight)
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        ticks = []
        for k in range(int(round(length / 4)) + 1):
            cx, cy = a[0] + ux * 4 * k, a[1] + uy * 4 * k
            ticks.append([(cx - uy * 0.45, cy + ux * 0.45), (cx + uy * 0.45, cy - ux * 0.45)])
        sh.lines_ft(ticks, color="black", linewidth=0.6, zorder=zlight)
    for l in plan.lights:
        draw_light(sh, l, zlight)
    # ceiling tags and room names, placed clear of the fixtures
    rcp_tag = {"125": (107, 104), "126": (127, 104), "124": (112, 95), "121": (122, 82.2), "122": (134, 82.2),
               "123": (149, 89.6), "120": (107, 83.5), "127": (142, 104), "128": (154, 104)}
    for name, num, x, y in plan.rooms:
        x, y = rcp_tag.get(num, (x, y))
        row = plan.schedule[num]
        sx, sy = S(x, y)
        tag = row["ceiling"] if row["ceiling"] == "OPEN" else f"{row['ceiling']}\n{row['height']} AFF"
        ax.text(sx, sy + 0.12, name, fontsize=7, fontweight="bold", ha="center", va="center", family="DejaVu Sans",
                zorder=zann, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.text(sx, sy - 0.12, tag, fontsize=5.5, ha="center", va="center", family="DejaVu Sans", zorder=zann,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.4))
    draw_plan_title(sh, "A102", "REFLECTED CEILING PLAN", zann)
    # legend
    x0, top = 24.4, 21.6
    ax.add_patch(Rectangle((x0 - 0.2, 6.0), 7.4, top - 5.8, fill=False, ec="black", lw=0.6, zorder=zann))
    sh.text(x0, top - 0.25, "LEGEND", 11, weight="bold", ha="left", z=zann)
    y = top - 0.75
    sh.text(x0, y, "CEILING TYPES", 8, weight="bold", ha="left", z=zann)
    y -= 0.35
    for code in ("ACT-1", "ACT-2", "GYP", "OPEN"):
        y -= 0.62
        fx, fy = (x0 - OX) / SCALE, (y - OY) / SCALE
        swatch = box(fx, fy, fx + 0.9 / SCALE, fy + 0.45 / SCALE)
        lines, dots = ceiling_pattern(code, swatch, rng)
        sh.lines_ft(lines, zorder=zann, **HATCH)
        if dots:
            xs, ys = zip(*[S(*pt) for pt in dots])
            ax.scatter(xs, ys, s=0.35, c="0.3", marker="o", linewidths=0, zorder=zann)
        ax.add_patch(Rectangle((x0, y), 0.9, 0.45, fill=False, ec="black", lw=0.5, zorder=zann))
        sh.text(x0 + 1.1, y + 0.3, code, 7.5, weight="bold", ha="left", z=zann)
        sh.text(x0 + 1.1, y + 0.1, CEILING_DESC[code], 6, ha="left", z=zann)
    y -= 0.55
    sh.text(x0, y, "LIGHTING (SEE FIXTURE SCHEDULE, A601)", 8, weight="bold", ha="left", z=zann)
    y -= 0.15
    for code, desc in LIGHT_DESC:
        y -= 0.62
        fx, fy = (x0 + 0.2 - OX) / SCALE, (y - OY) / SCALE
        if code == "L3":
            sh.lines_ft([[(fx, fy + 2), (fx + 5, fy + 2)]], color="black", linewidth=2.2, zorder=zann)
            sh.lines_ft([[(fx + 4, fy + 1.55), (fx + 4, fy + 2.45)]], color="black", linewidth=0.6, zorder=zann)
        else:
            rot = code.startswith("L1")
            draw_light(sh, Light(code, fx + (0.5 if code in ("L1", "L1E", "L2", "L2E") else 2),
                                 fy + (1 if code in ("L1", "L1E", "L2", "L2E") else 2), rotated=rot), zann)
        sh.text(x0 + 1.1, y + 0.3, code, 7.5, weight="bold", ha="left", z=zann)
        sh.text(x0 + 1.1, y + 0.1, desc, 6, ha="left", z=zann)
    y -= 0.6
    sh.text(x0, y, "GENERAL NOTES", 8, weight="bold", ha="left", z=zann)
    for line in ["1. CEILING HEIGHTS ARE ABOVE FINISH FLOOR (AFF).",
                 "2. L3 ROWS ARE CONTINUOUS; TICKS MARK 4'-0\" MODULES.",
                 "3. COORDINATE ALL DEVICES WITH ELECTRICAL DRAWINGS."]:
        y -= 0.24
        sh.text(x0, y, line, 6, ha="left", z=zann)
    draw_title_block(sh, zann, "A102", "REFLECTED CEILING", "PLAN", 3)
    return sh.save_array()


def render_schedules(plan):
    sh = Sheet()
    ax = sh.ax
    ax.add_patch(Rectangle((0.45, 0.45), SHEET_W - 0.9, SHEET_H - 0.9, fill=False, ec="black", lw=1.2, zorder=1))
    z = 60

    def table(x0, top, title, headers, widths, rows, row_h=0.42):
        sh.text(x0, top + 0.35, title, 12, weight="bold", ha="left", z=z)
        total = sum(widths)
        y = top
        ax.add_patch(Rectangle((x0, y - row_h), total, row_h, fc="0.85", ec="black", lw=0.8, zorder=z))
        cx = x0
        for h, w in zip(headers, widths):
            sh.text(cx + 0.1, y - row_h / 2, h, 7, weight="bold", ha="left", z=z + 1)
            cx += w
        y -= row_h
        for r in rows:
            ax.add_patch(Rectangle((x0, y - row_h), total, row_h, fill=False, ec="black", lw=0.5, zorder=z))
            cx = x0
            for val, w in zip(r, widths):
                sh.text(cx + 0.1, y - row_h / 2, val or "--", 7, ha="left", z=z + 1)
                cx += w
            y -= row_h
        cx = x0
        for w in widths[:-1]:
            cx += w
            ax.plot([cx, cx], [top, y], color="black", lw=0.5, zorder=z)
        return y

    rows = [(num, r["name"], r["floor"], r["base"], r["walls"], r["ceiling"], r["height"])
            for num, r in sorted(plan.schedule.items())]
    widths = [0.8, 2.3, 1.6, 1.0, 1.6, 1.0, 1.0]
    y = table(1.2, 21.0, "ROOM FINISH SCHEDULE", ["ROOM", "NAME", "FLOOR", "BASE", "WALLS", "CEILING", "CLG HT"],
              widths, rows)
    notes = ["BASE: RB-1 = 4\" RUBBER COVE BASE.  CTB-1 = 6\" PORCELAIN COVE BASE TO MATCH CT-1.",
             "BASE IS INSTALLED AT ALL WALLS OF ROOMS SCHEDULED TO RECEIVE BASE.",
             "WALLS: PT-1 EGGSHELL PAINT.  PT-2 EPOXY PAINT.  FRP-1 FIBERGLASS PANEL, FULL HEIGHT.",
             "FLOOR FINISH CODES PER LEGEND, SHEET A101."]
    for line in notes:
        y -= 0.3
        sh.text(1.2, y, line, 6.5, ha="left", z=z)
    lrows = [(code, desc, "RECESSED" if code in ("L1", "L1E", "L2", "L2E", "L4") else
              "PENDANT" if code in ("L3", "L5") else "WALL / CEILING", "LED") for code, desc in LIGHT_DESC]
    table(12.4, 21.0, "LIGHT FIXTURE SCHEDULE", ["TYPE", "DESCRIPTION", "MOUNTING", "SOURCE"],
          [0.8, 5.0, 1.6, 0.9], lrows)
    draw_title_block(sh, z, "A601", "SCHEDULES", "FINISH & LIGHTING", 9, scale="NONE", bar=False)
    return sh.save_array()


# ------------------------------------------------------------- redlines --
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

HAND = FontProperties(fname=str(_Path(__file__).with_name("fonts") / "ArchitectsDaughter-Regular.ttf"))
INK = (0.97, 0.15, 0.13)        # red pen: passes almost all red light, so it drops out of the red channel


class Pen:
    """Freehand red markup on its own layer, multiplied onto the print later."""

    def __init__(self, sheet, seed=11):
        self.sh = sheet
        self.rng = random.Random(seed)

    def line(self, pts, lw=1.1):
        xs, ys = zip(*[S(*p) for p in pts])
        ln = Line2D(xs, ys, color=INK, linewidth=lw, solid_capstyle="round", zorder=self.sh.nz())
        ln.set_sketch_params(scale=1.2, length=60, randomness=12)
        self.sh.ax.add_line(ln)

    def wall(self, x1, y1, x2, y2, t=0.42):
        # two freehand lines for a new wall, the way it is drawn on a print
        if y1 == y2:
            self.line([(x1, y1 - t / 2), (x2, y2 - t / 2)])
            self.line([(x1, y1 + t / 2), (x2, y2 + t / 2)])
        else:
            self.line([(x1 - t / 2, y1), (x2 - t / 2, y2)])
            self.line([(x1 + t / 2, y1), (x2 + t / 2, y2)])

    def rect(self, x, y, w, h, cross=False):
        self.line([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
        if cross:
            self.line([(x, y), (x + w, y + h)])
            self.line([(x, y + h), (x + w, y)])

    def x(self, x0, y0, x1, y1):
        self.line([(x0, y0), (x1, y1)], lw=1.4)
        self.line([(x0, y1), (x1, y0)], lw=1.4)

    def cloud(self, x0, y0, x1, y1, r=0.9):
        pts = []
        perim = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        for (ax_, ay), (bx, by) in zip(perim, perim[1:]):
            length = math.hypot(bx - ax_, by - ay)
            n = max(2, int(length / (2 * r)))
            dx, dy = (bx - ax_) / n, (by - ay) / n
            nx, ny = dy / math.hypot(dx, dy), -dx / math.hypot(dx, dy)   # outward for counter-clockwise
            for k in range(n):
                sx, sy = ax_ + dx * k, ay + dy * k
                for t in np.linspace(0, 1, 9):
                    bulge = math.sin(math.pi * t) * r * 0.7
                    pts.append((sx + dx * t + nx * bulge, sy + dy * t + ny * bulge))
        self.line(pts, lw=0.9)

    def arrow(self, x0, y0, x1, y1):
        self.sh.ax.annotate("", xy=S(x1, y1), xytext=S(x0, y0), zorder=self.sh.nz(),
                            arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=INK, lw=1.0,
                                            shrinkA=0, shrinkB=0))

    def note(self, x, y, text, size=9.5, rotation=None, ha="center"):
        rot = rotation if rotation is not None else self.rng.uniform(-2.5, 2.5)
        sx, sy = S(x, y)
        self.sh.ax.text(sx, sy, text, fontproperties=HAND, fontsize=size, color=INK, ha=ha, va="center",
                        rotation=rot, linespacing=1.05, zorder=self.sh.nz())

    def door(self, hinge_x, face_y, width, swing):
        self.line([(hinge_x, face_y), (hinge_x, face_y + swing * width)])
        pts = [(hinge_x + width * math.sin(a), face_y + swing * width * math.cos(a))
               for a in np.linspace(0, math.pi / 2, 20)]
        self.line(pts, lw=0.8)


def draw_redlines(sh):
    """The contractor's as-built markups (see plan.asbuilt for what they mean)."""
    pen = Pen(sh)
    # office / break wall built 2'-0" east
    pen.cloud(116.4, 98.7, 121.8, 109.0)
    pen.wall(120, 98.3, 120, 109.4, t=0.4)
    pen.arrow(118.3, 102.0, 119.8, 102.0)
    pen.note(108.5, 107.6, "WALL BUILT 2'-0\" EAST", size=8.5)
    pen.line([(113.5, 106.7), (116.6, 104.5)], lw=0.8)
    # janitor / IT wall not built
    pen.x(136.8, 80.8, 139.2, 91.2)
    pen.x(131.6, 88.6, 135.4, 91.6)
    pen.note(143.5, 75.0, "WALL NOT BUILT - RMS 122 & 123\nONE ROOM, VCT-1 THRU-OUT.\nDOOR 115 DELETED", size=8.5)
    pen.arrow(140.0, 77.3, 137.6, 82.5)
    # baler / recycling room
    pen.wall(80, 100, 100, 100)
    pen.wall(80, 100, 80, 109.5)
    pen.door(86, 99.8, 3, -1)
    pen.note(62.0, 104.0, "NEW P1 PARTITION -\nBALER / RECYCLING RM\nw/ 3'-0\" DOOR. FINISHES\nTO MATCH STOCKROOM", size=9)
    pen.arrow(72.0, 103.0, 79.2, 102.0)
    # pharmacy wall built 4'-0" south
    pen.cloud(127.4, 48.6, 159.0, 51.4)
    pen.wall(128, 46, 159.6, 46, t=0.6)
    pen.arrow(152.0, 49.6, 152.0, 46.6)
    pen.door(136, 46.3, 3, 1)
    pen.note(143.0, 42.4, "PHARMACY WALL BUILT 4'-0\"\nSOUTH OF LOCATION SHOWN\n(DOOR 117 MOVES W/ WALL)", size=8.5)
    # door 104
    pen.cloud(83.2, 78.6, 90.8, 83.8)
    pen.note(89.0, 88.0, "DOOR 104: (1) 3'-0\" LEAF\nIN LIEU OF PAIR, INFILL CMU", size=8.5)
    # cash office carpet
    pen.cloud(115.0, 81.0, 129.0, 83.4, r=0.6)
    pen.note(122.0, 82.2, "CPT-1 IN LIEU OF VCT-1", size=8)
    # receiving epoxy to grid B
    pen.line([(36.2, 94), (44, 94), (44, 109.2)], lw=1.2)
    pen.arrow(37.5, 101.5, 43.4, 101.5)
    pen.note(28.5, 105.5, "EPOXY EXTENDS\nTO GRID LINE B", size=9)
    # gondola run not installed
    pen.x(56.4, 34.4, 61.6, 65.0)
    pen.note(52.0, 71.2, "RUN NOT INSTALLED", size=9)
    pen.arrow(53.5, 69.8, 57.2, 65.6)
    # west cooler lineup extended
    pen.rect(34, 76.667, 5, 3)
    for i in range(2):
        pen.rect(34.1 + 2.5 * i, 76.667, 2.3, 0.6, cross=True)
    pen.note(26.5, 72.4, "LINEUP EXTENDED -\n14 DOORS TOTAL", size=8.5)
    pen.arrow(31.5, 73.6, 35.0, 76.2)
    # self-checkout not installed
    pen.x(122.6, 14.6, 126.4, 18.4)
    pen.note(137.0, 21.5, "SCO NOT INSTALLED", size=9)
    pen.arrow(130.5, 20.8, 126.8, 18.3)
    # two more produce tables
    pen.rect(8.5, 68, 5, 4, cross=True)
    pen.rect(15.5, 68, 5, 4, cross=True)
    pen.note(13.0, 66.4, "(2) ADD'L PRODUCE TABLES", size=8.5, rotation=0.5)
    # run 112 rebuilt with 4'-0" sections
    pen.cloud(111.3, 35.0, 116.7, 63.8)
    pen.note(121.4, 49.5, "BUILT AS (6) 4'-0\" + (1) 3'-0\" SECTIONS", size=8.5, rotation=90)
    # notes that change nothing in the takeoff
    pen.note(96.5, 8.5, "OK AS SHOWN", size=9)
    pen.line([(93.2, 8.1), (93.8, 7.3), (95.0, 9.3)], lw=1.2)
    pen.note(125.0, 95.0, "PAINT ACCENT - SEE A601", size=7.5, rotation=0)
    pen.note(104.0, 3.2, "FLOOR CRACK @ ENTRY - PATCH ONLY", size=8)
    stamp(sh)


def stamp(sh, x=108, y=-15.0, extra=None):
    sx, sy = S(x, y)
    sh.ax.add_patch(Rectangle((sx - 1.35, sy - 0.42), 2.7, 0.84, fill=False, ec=INK, lw=1.4, zorder=sh.nz()))
    sh.ax.text(sx, sy + 0.18, "AS-BUILT REDLINES", fontproperties=HAND, fontsize=13, color=INK, ha="center",
               va="center", zorder=sh.nz())
    sh.ax.text(sx, sy - 0.2, "ALPINE BUILDERS - J. MORALES 9/28/19", fontproperties=HAND, fontsize=8.5,
               color=INK, ha="center", va="center", zorder=sh.nz())


def draw_redlines_rcp(sh):
    """Markups on the reflected ceiling plan (see plan.asbuilt)."""
    pen = Pen(sh, seed=12)
    pen.cloud(127.4, 49.2, 159.0, 79.4)
    for x in (136, 144, 152):
        for y in (57, 69):
            pts = [(x + 0.5 * math.cos(a), y + 0.5 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 24)]
            pen.line(pts, lw=1.0)
    pen.note(143.5, 44.2, "PHARMACY: GYP CLG IN LIEU OF ACT-1.\n(6) L4 IN LIEU OF L2 / L2E", size=9)
    pen.cloud(130.4, 80.6, 137.8, 91.4)
    pen.note(112.0, 76.6, "RMS 122 + 123 ONE ROOM - ACT-1 THRU-OUT.\nJAN L1 -> L2", size=8.5)
    pen.arrow(124.5, 77.6, 131.0, 82.5)
    pen.line([(107.6, 33), (109.4, 39), (107.6, 45), (109.4, 51), (107.6, 57), (109.4, 63), (107.6, 67)], lw=1.3)
    pen.note(100.0, 29.0, "L3 ROW NOT INSTALLED", size=9)
    pen.arrow(104.0, 30.0, 107.8, 33.0)
    for x in (40, 60):
        pts = [(x + 0.9 * math.cos(a), 94 + 0.9 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 30)]
        pen.line(pts, lw=1.1)
        pen.line([(x - 0.6, 94), (x + 0.6, 94)], lw=0.8)
    pen.note(50.0, 90.6, "(2) ADD'L L5", size=9)
    pen.note(130.0, 112.6, "ALL CORRIDOR L1 TO BE L1E (EM)", size=9, rotation=0)
    pen.arrow(130.0, 111.6, 130.0, 96.6)
    pen.note(60.0, 50.0, "VERIFY DECK HT 22'-0\" +/-", size=9)
    pen.note(108.0, -10.6, "CEILINGS & LIGHTS FOLLOW AS-BUILT WALLS - SEE A101 REDLINES", size=9, rotation=0)
    stamp(sh)


def draw_redlines_schedule(sh):
    """Markup on the room finish schedule: the pharmacy got porcelain base."""
    pen = Pen(sh, seed=13)
    ft = lambda xi, yi: ((xi - OX) / SCALE, (yi - OY) / SCALE)
    row_y = 21.0 - 0.42 - 0.42 * 2 - 0.21          # room 102 is the third row
    pen.line([ft(5.97, row_y), ft(6.38, row_y)], lw=1.3)
    pen.cloud(*ft(5.88, row_y - 0.19), *ft(6.92, row_y + 0.19), r=0.55)
    pen.note(*ft(6.68, row_y), "CTB-1", size=9, rotation=0)
    sx, sy = 6.0, 6.0
    sh.ax.add_patch(Rectangle((sx - 1.35, sy - 0.42), 2.7, 0.84, fill=False, ec=INK, lw=1.4, zorder=sh.nz()))
    sh.ax.text(sx, sy + 0.18, "AS-BUILT REDLINES", fontproperties=HAND, fontsize=13, color=INK, ha="center",
               va="center", zorder=sh.nz())
    sh.ax.text(sx, sy - 0.2, "ALPINE BUILDERS - J. MORALES 9/28/19", fontproperties=HAND, fontsize=8.5,
               color=INK, ha="center", va="center", zorder=sh.nz())


def render_redlines(sheet="A101"):
    sh = Sheet()
    {"A101": draw_redlines, "A102": draw_redlines_rcp, "A601": draw_redlines_schedule}[sheet](sh)
    sh.fig.canvas.draw()
    buf = np.asarray(sh.fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(sh.fig)
    return buf
