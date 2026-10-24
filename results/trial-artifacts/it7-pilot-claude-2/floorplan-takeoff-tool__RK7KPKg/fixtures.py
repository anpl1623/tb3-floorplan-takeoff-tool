"""Fixture counts from the white-filled CAD blocks of the chain."""
import numpy as np
import cv2


class Comp:
    __slots__ = ('i', 'x', 'y', 'w', 'h', 'a', 'W', 'H', 'fill', 'corners', 'cx', 'cy', 'used', 'diag', 'iso')

    def __init__(self, i, st, cen, mask, ppf):
        x, y, w, h, a = st
        self.i = i
        self.x, self.y, self.w, self.h, self.a = int(x), int(y), int(w), int(h), int(a)
        self.W = w / ppf
        self.H = h / ppf
        self.fill = a / float(w * h)
        self.cx, self.cy = float(cen[0]), float(cen[1])
        q = 3
        hh = max(1, h // q)
        ww = max(1, w // q)
        self.corners = (float(mask[:hh, :ww].mean()), float(mask[:hh, -ww:].mean()),
                        float(mask[-hh:, :ww].mean()), float(mask[-hh:, -ww:].mean()))
        self.used = False
        yy, xx = np.mgrid[0:h, 0:w]
        u = (xx + 0.5) / w
        v = (yy + 0.5) / h
        m = mask.astype(np.float32)
        upper_right = v < u          # above the tl-br diagonal
        upper_left = v < 1 - u       # above the tr-bl diagonal
        def mm(sel):
            return float(m[sel].mean()) if sel.any() else 0.0
        self.diag = {'tr': mm(upper_right) - mm(~upper_right), 'bl': mm(~upper_right) - mm(upper_right),
                     'tl': mm(upper_left) - mm(~upper_left), 'br': mm(~upper_left) - mm(upper_left)}
        # isosceles triangles (quarters of an X-divided box): IoU with ideal shapes
        ideal = {'down': v < 1 - np.abs(2 * u - 1),          # base on top, apex at bottom centre
                 'up': v > np.abs(2 * u - 1),                 # base at bottom
                 'right': u < 1 - np.abs(2 * v - 1),          # base on the left, apex right
                 'left': u > np.abs(2 * v - 1)}
        mb = mask.astype(bool)
        self.iso = {}
        for k2, idl in ideal.items():
            inter = float((mb & idl).sum())
            uni = float((mb | idl).sum())
            self.iso[k2] = inter / uni if uni > 0 else 0.0

    @property
    def long(self):
        return max(self.W, self.H)

    @property
    def short(self):
        return min(self.W, self.H)

    def right_triangle(self):
        """A rectangle cut by its diagonal: returns the corner holding the right angle."""
        if not (0.33 < self.fill < 0.72):
            return None
        k = max(self.diag, key=self.diag.get)
        if self.diag[k] < 0.6:
            return None
        return k

    def isosceles(self):
        if not (0.33 < self.fill < 0.72):
            return None
        k = max(self.iso, key=self.iso.get)
        if self.iso[k] < 0.62:
            return None
        return k


def white_components(tsub, region, thr=0.10):
    sm = cv2.GaussianBlur(np.clip(tsub, 0, 1).astype(np.float32), (3, 3), 0)
    ink = (sm > thr).astype(np.uint8)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    white = (ink == 0) & region
    n, cl, st, cen = cv2.connectedComponentsWithStats(white.astype(np.uint8), connectivity=4)
    return n, cl, st, cen


def detect(tsub, region, ppf, dbg=None, thr=0.10):
    n, cl, st, cen = white_components(tsub, region, thr)
    comps = []
    maxa = 15 * ppf
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.08 * ppf * ppf:
            continue
        if w > maxa or h > maxa:
            continue
        m = cl[y:y + h, x:x + w] == i
        comps.append(Comp(i, st[i], cen[i], m, ppf))
    res = {}
    lw = 0.12  # typical line width (ft) between adjacent white parts

    # ---------------- gondola sections: two 2'-0" halves either side of the centre line
    halves = [c for c in comps if c.fill > 0.86 and 1.65 <= c.short <= 2.2 and 2.3 <= c.long <= 4.5]
    used = set()
    sections = []
    for c in halves:
        if c.i in used:
            continue
        vert = c.H > c.W   # run axis vertical => halves side by side in x
        best = None
        for d in halves:
            if d is c or d.i in used:
                continue
            if (d.H > d.W) != vert:
                continue
            if vert:
                same = abs(d.y - c.y) <= 0.2 * ppf and abs(d.h - c.h) <= 0.2 * ppf
                gap = max(d.x - (c.x + c.w), c.x - (d.x + d.w))
            else:
                same = abs(d.x - c.x) <= 0.2 * ppf and abs(d.w - c.w) <= 0.2 * ppf
                gap = max(d.y - (c.y + c.h), c.y - (d.y + d.h))
            if same and -2 <= gap <= 0.4 * ppf:
                if best is None or gap < best[0]:
                    best = (gap, d)
        if best is not None:
            used.add(c.i)
            used.add(best[1].i)
            L = 0.5 * (c.long + best[1].long) + lw
            sections.append((L, c, best[1]))
        else:
            # unpaired half: only if it is stacked with another half along a run
            stacked = False
            for d in halves:
                if d is c:
                    continue
                if vert and (d.H > d.W) and abs(d.x - c.x) <= 0.2 * ppf and \
                        (abs(d.y - (c.y + c.h)) < 0.4 * ppf or abs(c.y - (d.y + d.h)) < 0.4 * ppf):
                    stacked = True
                if (not vert) and (d.W > d.H) and abs(d.y - c.y) <= 0.2 * ppf and \
                        (abs(d.x - (c.x + c.w)) < 0.4 * ppf or abs(c.x - (d.x + d.w)) < 0.4 * ppf):
                    stacked = True
            used.add(c.i)
            if stacked:
                sections.append((c.long + lw, c, None))
    n4 = sum(1 for L, a, b in sections if L >= 3.5)
    n3 = sum(1 for L, a, b in sections if L < 3.5)
    res['gondola_sections_4ft'] = n4
    res['gondola_sections_3ft'] = n3

    # ---------------- endcaps: rectangle 4'-0" wide split by a diagonal
    tris = [(c, c.right_triangle()) for c in comps]
    tris = [(c, o) for c, o in tris if o is not None]
    end_halves = [(c, o) for c, o in tris if 2.8 <= c.long <= 4.3 and 0.9 <= c.short <= 2.0]
    endcaps = _pair_triangles(end_halves, ppf, ((3.3, 4.5), (1.0, 2.2)))
    res['endcaps'] = len(endcaps)

    # ---------------- reach-in cooler doors: small rectangle split by a diagonal, in rows
    #                  along a white cooler case
    door_halves = [(c, o) for c, o in tris if 0.9 <= c.long <= 2.8 and 0.25 <= c.short <= 0.85]
    doors, door_rows = _count_door_rows(door_halves, cl, ppf, tsub)
    res['cooler_doors'] = doors

    # ---------------- X-divided boxes (produce tables, checkout bag wells)
    xboxes = _find_x_boxes(comps, ppf)

    # ---------------- self-checkout kiosks: circle inside a square
    disks = []
    for c in comps:
        if 0.70 < c.fill < 0.88 and abs(c.W - c.H) < 0.1 * max(c.W, c.H) and 1.0 <= c.long <= 2.0:
            if _square_around(c, comps, ppf):
                disks.append(c)
    res['self_checkouts'] = len(disks)

    # ---------------- checkout stands: counter (about 2'-4" x 8'-9") with a bag well (X box)
    #                  at one end
    counters = []
    for c in comps:
        if 0.75 < c.fill < 0.97 and 1.9 <= c.short <= 3.0 and 7.0 <= c.long <= 11.0:
            if _xbox_at_end(c, xboxes, ppf):
                counters.append(c)
    res['checkouts'] = len(counters)

    # ---------------- produce tables: 5'-0" x 4'-0" with an X
    tables = [b for b in xboxes if (abs(b[2] - 5.0) < 0.6 and abs(b[3] - 4.0) < 0.6) or
              (abs(b[2] - 4.0) < 0.6 and abs(b[3] - 5.0) < 0.6)]
    res['produce_tables'] = len(tables)
    occ = set()
    for L_, a_, b_ in sections:
        occ.add(a_.i)
        if b_ is not None:
            occ.add(b_.i)
    for a_, b_ in endcaps:
        occ.add(a_.i)
        occ.add(b_.i)
    for c in disks + counters:
        occ.add(c.i)
    for (bx, by, bW, bH, parts) in xboxes:
        for q in parts:
            occ.add(q.i)
    for horiz, items, n_, nn in door_rows:
        for c, o in items:
            occ.add(c.i)
    for c in comps:
        # the square ring of a kiosk
        for d in disks:
            if c.x <= d.x and c.y <= d.y and c.x + c.w >= d.x + d.w and c.y + c.h >= d.y + d.h and c is not d:
                if c.long < 3.5:
                    occ.add(c.i)
    res_occ = sorted(occ)
    if dbg is not None:
        dbg['occluders'] = res_occ
        dbg['case_ids'] = [nn[4] for horiz, items, n_, nn in door_rows]
    if dbg is not None:
        dbg.update({'comps': comps, 'sections': sections, 'endcaps': endcaps, 'door_rows': door_rows,
                    'disks': disks, 'counters': counters, 'cl': cl, 'xboxes': xboxes, 'tables': tables})
    return res


def _pair_triangles(halves, ppf, size_rng, allow_single=False):
    """Group complementary right triangles that split one rectangle along its diagonal.
    size_rng = ((long_min, long_max), (short_min, short_max)) for the union box (ft)."""
    comp = {'tl': 'br', 'br': 'tl', 'tr': 'bl', 'bl': 'tr'}
    (lmin, lmax), (smin, smax) = size_rng
    used = set()
    out = []
    for c, o in halves:
        if c.i in used:
            continue
        best = None
        for d, o2 in halves:
            if d is c or d.i in used or o2 != comp[o]:
                continue
            # centroid displacement consistent with the shared diagonal
            a, b = (c, d) if o in ('tl', 'bl') else (d, c)   # a: left-hand triangle
            oa = o if a is c else o2
            if b.cx <= a.cx:
                continue
            if oa == 'bl' and not (b.cy < a.cy):
                continue
            if oa == 'tl' and not (b.cy > a.cy):
                continue
            x0 = min(c.x, d.x)
            y0 = min(c.y, d.y)
            x1 = max(c.x + c.w, d.x + d.w)
            y1 = max(c.y + c.h, d.y + d.h)
            W = (x1 - x0) / ppf
            H = (y1 - y0) / ppf
            L, S = max(W, H), min(W, H)
            if not (lmin <= L <= lmax and smin <= S <= smax):
                continue
            # strong overlap across the short side
            if W >= H:
                ov = min(c.y + c.h, d.y + d.h) - max(c.y, d.y)
                ref = min(c.h, d.h)
            else:
                ov = min(c.x + c.w, d.x + d.w) - max(c.x, d.x)
                ref = min(c.w, d.w)
            if ov < 0.6 * ref:
                continue
            dist = (c.cx - d.cx) ** 2 + (c.cy - d.cy) ** 2
            if best is None or dist < best[0]:
                best = (dist, d)
        if best is not None:
            used.add(c.i)
            used.add(best[1].i)
            out.append((c, best[1]))
        elif allow_single:
            used.add(c.i)
            out.append((c, None))
    return out


def _find_x_boxes(comps, ppf):
    """Boxes divided by an X into four isosceles triangles.  Returns (x0,y0,W_ft,H_ft,parts)."""
    tri = {'down': [], 'up': [], 'left': [], 'right': []}
    for c in comps:
        k = c.isosceles()
        if k is not None and c.long >= 0.8:
            tri[k].append(c)
    out = []
    used = set()
    tol = 0.45 * ppf
    for t in tri['down']:
        if t.i in used:
            continue
        # bottom partner: same x-range, directly below
        bots = [b for b in tri['up'] if b.i not in used and abs(b.x - t.x) < tol and
                abs(b.x + b.w - t.x - t.w) < tol and 0 <= b.y - (t.y + t.h) + 0.25 * ppf < t.h * 1.2 + tol]
        if not bots:
            continue
        b = min(bots, key=lambda q: abs(q.y - (t.y + t.h)))
        x0, x1 = min(t.x, b.x), max(t.x + t.w, b.x + b.w)
        y0, y1 = t.y, b.y + b.h
        lefts = [l for l in tri['right'] if l.i not in used and abs(l.x - x0) < tol and
                 abs(l.y - y0) < tol and abs(l.y + l.h - y1) < tol]
        rights = [r for r in tri['left'] if r.i not in used and abs(r.x + r.w - x1) < tol and
                  abs(r.y - y0) < tol and abs(r.y + r.h - y1) < tol]
        if not lefts or not rights:
            continue
        l = lefts[0]
        r = rights[0]
        for q in (t, b, l, r):
            used.add(q.i)
        X0 = min(x0, l.x)
        X1 = max(x1, r.x + r.w)
        out.append((X0, y0, (X1 - X0) / ppf, (y1 - y0) / ppf, (t, b, l, r)))
    return out


def _square_around(c, comps, ppf):
    """Is the disk c centred in a square ring (the self-checkout kiosk outline, ~3'-0")?"""
    for d in comps:
        if d is c:
            continue
        if d.x <= c.x and d.y <= c.y and d.x + d.w >= c.x + c.w and d.y + d.h >= c.y + c.h:
            if abs(d.W - d.H) < 0.15 * max(d.W, d.H) and 2.4 <= d.long <= 3.5 \
                    and 1.6 * c.long <= d.long <= 2.6 * c.long and 0.45 < d.fill < 0.9:
                dcx = d.x + d.w / 2.0
                dcy = d.y + d.h / 2.0
                ccx = c.x + c.w / 2.0
                ccy = c.y + c.h / 2.0
                if abs(dcx - ccx) < 0.25 * ppf and abs(dcy - ccy) < 0.25 * ppf:
                    return True
    return False


def _xbox_at_end(c, xboxes, ppf):
    """Is there an X box adjoining one end of the counter c?"""
    tol = 0.6 * ppf
    vert = c.H > c.W
    for (bx, by, bW, bH, parts) in xboxes:
        bw = bW * ppf
        bh = bH * ppf
        if vert:
            if abs(bx - c.x) < tol and abs(bx + bw - c.x - c.w) < tol:
                if abs(by - (c.y + c.h)) < tol or abs(by + bh - c.y) < tol:
                    return True
        else:
            if abs(by - c.y) < tol and abs(by + bh - c.y - c.h) < tol:
                if abs(bx - (c.x + c.w)) < tol or abs(bx + bw - c.x) < tol:
                    return True
    return False


DOOR_MODULE = 2.5   # ft, reach-in cooler door module of the chain's block


def _row_template_count(img, items, horiz, ppf, case_rng):
    """Count doors along a row by NCC with a template averaged from the row's own doors."""
    comp = {'tl': 'br', 'br': 'tl', 'tr': 'bl', 'bl': 'tr'}
    # well-formed doors: complementary halves overlapping along the row
    boxes = []
    used = set()
    for i, (c, o) in enumerate(items):
        if i in used:
            continue
        for j, (d, o2) in enumerate(items):
            if j <= i or j in used or o2 != comp[o]:
                continue
            if horiz:
                ov = min(c.x + c.w, d.x + d.w) - max(c.x, d.x)
            else:
                ov = min(c.y + c.h, d.y + d.h) - max(c.y, d.y)
            if ov > 0:
                x0 = min(c.x, d.x)
                y0 = min(c.y, d.y)
                x1 = max(c.x + c.w, d.x + d.w)
                y1 = max(c.y + c.h, d.y + d.h)
                L = (x1 - x0) if horiz else (y1 - y0)
                if 1.6 * ppf <= L <= 2.9 * ppf:
                    boxes.append((x0, y0, x1, y1))
                    used.add(i)
                    used.add(j)
                break
    if len(boxes) < 2:
        return None
    # template: median box size, padded
    pad = 4
    ws = int(np.median([b[2] - b[0] for b in boxes])) + 2 * pad
    hs = int(np.median([b[3] - b[1] for b in boxes])) + 2 * pad
    H, W = img.shape
    patches = []
    for (x0, y0, x1, y1) in boxes:
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        xa = cx - ws // 2
        ya = cy - hs // 2
        if xa < 0 or ya < 0 or xa + ws > W or ya + hs > H:
            continue
        patches.append(img[ya:ya + hs, xa:xa + ws])
    if len(patches) < 2:
        return None
    T = np.median(np.stack(patches), axis=0).astype(np.float32)
    # search band along the row
    cy = int(np.median([(b[1] + b[3]) // 2 for b in boxes])) if horiz else None
    cxr = int(np.median([(b[0] + b[2]) // 2 for b in boxes])) if not horiz else None
    lo, hi = case_rng
    m = 6
    if horiz:
        ya = cy - hs // 2 - m
        yb = cy + hs // 2 + m + 1
        xa = int(max(0, lo - ws))
        xb = int(min(W, hi + ws))
        band = img[max(0, ya):min(H, yb), xa:xb].astype(np.float32)
    else:
        xa_ = cxr - ws // 2 - m
        xb_ = cxr + ws // 2 + m + 1
        ya_ = int(max(0, lo - hs))
        yb_ = int(min(H, hi + hs))
        band = img[ya_:yb_, max(0, xa_):min(W, xb_)].astype(np.float32)
    if band.shape[0] < T.shape[0] or band.shape[1] < T.shape[1]:
        return None
    R = cv2.matchTemplate(band, T, cv2.TM_CCOEFF_NORMED)
    prof = R.max(axis=0) if horiz else R.max(axis=1)
    # reference level: NCC at the known doors
    sep = int(0.75 * DOOR_MODULE * ppf)
    peaks = []
    order = np.argsort(-prof)
    taken = np.zeros(len(prof), bool)
    for k in order:
        if prof[k] < 0.45:
            break
        if taken[max(0, k - sep):k + sep + 1].any():
            continue
        peaks.append(k)
        taken[k] = True
    if not peaks:
        return None
    vals = prof[peaks]
    ref = float(np.median(vals[:max(1, len(boxes))]))
    n = int(np.sum(vals >= max(0.45, 0.55 * ref)))
    return n


def _count_door_rows(halves, cl, ppf, img=None):
    """Cooler doors: rows of diagonal-split door rectangles in front of a white cooler case.
    Halves are grouped by the case component directly behind them; doors are counted by
    template matching along the row (template from the row's own doors), falling back to
    the half counts / run extent."""
    H, W = cl.shape
    groups = {}
    for c, o in halves:
        horiz = c.W >= c.H
        best = None
        for sgn in (-1, 1):
            ids = []
            for f in (0.25, 0.5, 0.75):
                if horiz:
                    px = int(c.x + f * c.w)
                    py = int(c.y - 0.35 * ppf) if sgn < 0 else int(c.y + c.h + 0.35 * ppf)
                else:
                    py = int(c.y + f * c.h)
                    px = int(c.x - 0.35 * ppf) if sgn < 0 else int(c.x + c.w + 0.35 * ppf)
                if 0 <= px < W and 0 <= py < H:
                    ids.append(int(cl[py, px]))
            ids = [i for i in ids if i > 0]
            if not ids:
                continue
            cid = max(set(ids), key=ids.count)
            sz = None
            if best is None:
                best = (sgn, cid)
        if best is None:
            continue
        key = (horiz, best[1], best[0])
        groups.setdefault(key, []).append((c, o))
    total = 0
    kept = []
    for (horiz, cid, sgn), items in groups.items():
        ys, xs = np.nonzero(cl == cid)
        if len(xs) == 0:
            continue
        ext_case = ((xs.max() - xs.min()) if horiz else (ys.max() - ys.min())) / ppf
        if ext_case < 2.0:
            continue
        fam = {}
        for c, o in items:
            fam[o] = fam.get(o, 0) + 1
        n_tri = max(fam.values())
        if horiz:
            a0 = min(c.x for c, o in items)
            a1 = max(c.x + c.w for c, o in items)
            back = (min(c.y for c, o in items) - 0.3 * ppf) if sgn < 0 else (max(c.y + c.h for c, o in items) + 0.3 * ppf)
        else:
            a0 = min(c.y for c, o in items)
            a1 = max(c.y + c.h for c, o in items)
            back = (min(c.x for c, o in items) - 0.3 * ppf) if sgn < 0 else (max(c.x + c.w for c, o in items) + 0.3 * ppf)
        n_ext = int(round((a1 - a0) / ppf / DOOR_MODULE + 0.04))
        b = int(round(back))
        mid = int(0.5 * (a0 + a1))
        if horiz and 0 <= b < H:
            row = cl[b, :] == cid
        elif (not horiz) and 0 <= b < W:
            row = cl[:, b] == cid
        else:
            row = None
        n_case = 0
        case_rng = (a0, a1)
        if row is not None and 0 <= mid < len(row) and row[mid]:
            lo = mid
            while lo > 0 and row[lo - 1]:
                lo -= 1
            hi = mid
            while hi < len(row) - 1 and row[hi + 1]:
                hi += 1
            Lc = (hi - lo + 1) / ppf
            n_case = int(round(Lc / DOOR_MODULE + 0.04))
            if n_case > n_tri + 3:
                n_case = 0
            else:
                case_rng = (min(lo, a0), max(hi, a1))
        n_tpl = None
        if img is not None:
            try:
                n_tpl = _row_template_count(img, items, horiz, ppf, case_rng)
            except Exception:
                n_tpl = None
        if n_tpl is not None and n_tpl >= max(1, n_tri - 1):
            n = n_tpl
        else:
            n = max(n_tri, n_ext, n_case)
        if n_tri < 2 and n_case == 0 and n_tpl is None:
            continue
        total += n
        kept.append((horiz, items, n, (n_tri, n_ext, n_case, n_tpl, cid)))
    return total, kept


def occluder_mask(cl, ids, case_ids, ppf):
    """Pixels hidden by fixtures (white-filled blocks), grown to cover their outlines."""
    m = np.isin(cl, np.array(sorted(set(ids) | set(case_ids)), dtype=cl.dtype)) if (ids or case_ids) else np.zeros(cl.shape, bool)
    r = max(3, int(round(0.25 * ppf)))
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def cooler_cases(tink, fr, lines, dbg=None):
    """Reach-in cooler runs found along wall faces: a white cooler case of constant depth
    (about 2'-4") in front of the wall, followed by a row of 2'-6" doors.  Returns a list of
    (line, side, start_ft, end_ft, depth_ft, n_doors)."""
    sm = cv2.GaussianBlur(np.clip(tink, 0, 1).astype(np.float32), (3, 3), 0)
    inkm = sm > 0.10
    out = []
    for L in lines.values():
        o = L.o
        pa = fr.pp(o)
        pc = fr.pp('v' if o == 'h' else 'h')
        sides = (-1, 1)
        if L.ext:
            # only the side facing the building interior
            mid = (fr.H if o == 'h' else fr.W) / 2.0
            sides = (1,) if L.c < mid else (-1,)
        for side in sides:
            face = L.c + side * L.t / 2.0
            for (s, e) in L.runs:
                a0 = int(round(fr.along_px(o, s)))
                a1 = int(round(fr.along_px(o, e)))
                if a1 - a0 < 2.5 * pa:
                    continue
                f_px = fr.cross_px(o, face)
                d0 = int(round(0.12 * pc))
                d1 = int(round(3.4 * pc))
                if side > 0:
                    r0 = int(np.ceil(f_px)) + d0
                    r1 = r0 + (d1 - d0)
                else:
                    r1 = int(np.floor(f_px)) - d0 + 1
                    r0 = r1 - (d1 - d0)
                H, W = inkm.shape
                if o == 'h':
                    if r0 < 0 or r1 > H:
                        continue
                    band = inkm[r0:r1, a0:a1]
                    tb = sm[r0:r1, a0:a1]
                else:
                    if r0 < 0 or r1 > W:
                        continue
                    band = inkm[a0:a1, r0:r1].T
                    tb = sm[a0:a1, r0:r1].T
                if side < 0:
                    band = band[::-1]
                    tb = tb[::-1]
                # white depth: first inked row after the face line (depth from the face)
                skip = int(round(0.35 * pc))
                lead = band[:skip]
                # rows still on the face line: leading inked rows
                clear = ~lead
                start = np.where(clear.any(0), clear.argmax(0), skip)
                rows = np.arange(band.shape[0])[:, None]
                after = band & (rows >= start[None, :])
                has = after.any(0)
                first = np.where(has, after.argmax(0), band.shape[0])
                depth = (first + d0) / pc
                ok = (depth >= 1.7) & (depth <= 3.0)
                # intervals of consistent depth
                i = 0
                n = len(ok)
                while i < n:
                    if not ok[i]:
                        i += 1
                        continue
                    j = i
                    while j < n and ok[j]:
                        j += 1
                    # allow short interruptions (tags, grid lines) of < 1 ft
                    k = j
                    while k < n:
                        g = k
                        while g < n and not ok[g]:
                            g += 1
                        if g < n and (g - k) < 1.0 * pa:
                            j2 = g
                            while j2 < n and ok[j2]:
                                j2 += 1
                            if abs(np.median(depth[g:j2]) - np.median(depth[i:j])) < 0.25:
                                j = j2
                                k = j2
                                continue
                        break
                    seg_len = (j - i) / pa
                    if seg_len >= 2.0:
                        D = float(np.median(depth[i:j]))
                        dd = depth[i:j]
                        cons = np.mean(np.abs(dd[np.isfinite(dd)] - D) < 0.2)
                        if cons > 0.8:
                            # door row band in front of the case
                            q0 = int(round(D * pc)) - d0
                            q1 = q0 + int(round(0.75 * pc))
                            prof = tb[max(0, q0):q1, i:j].mean(0)
                            per = _periodicity(prof, 2.5 * pa)
                            if per > 0.3:
                                n_len = int(round(seg_len / 2.5 + 0.04))
                                ext = int(round(1.2 * pa))
                                ie = max(0, i - ext)
                                je = min(tb.shape[1], j + ext)
                                prof_e = tb[max(0, q0):q1, ie:je].mean(0)
                                n_pk = _divider_count(prof_e, pa)
                                # a clean interval (strong 2'-6" periodicity) is measured by
                                # its length; a diluted one (it runs past the case) by the
                                # door dividers inside it
                                if per >= 0.75 or n_pk is None or not (n_len - 2 <= n_pk <= n_len + 1):
                                    nd = n_len
                                else:
                                    nd = n_pk
                                out.append((L, side, s + i / pa, s + j / pa, D, nd, per, n_len, n_pk))
                    i = j
    return out


def _periodicity(prof, period):
    """Normalised autocorrelation of a 1-D profile at the given period (px)."""
    p = prof - prof.mean()
    if len(p) < 1.5 * period or p.std() < 1e-6:
        return 0.0
    lag0 = int(round(period))
    best = 0.0
    for lag in range(int(0.9 * lag0), int(1.1 * lag0) + 1):
        if lag >= len(p) - 2:
            break
        a = p[:-lag]
        b = p[lag:]
        c = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
        best = max(best, c)
    return best


def _divider_count(prof, pa, module=2.5):
    """Number of doors from the door dividers (short sides) along the door row profile:
    the longest chain of peaks spaced by whole door modules."""
    p = cv2.GaussianBlur(prof.astype(np.float32).reshape(1, -1), (0, 0), 1.5).ravel()
    if len(p) < 3:
        return None
    base = float(np.median(p))
    top = float(np.percentile(p, 98))
    if top - base < 1e-3:
        return None
    thr = base + 0.35 * (top - base)
    sep = int(0.6 * module * pa)
    order = np.argsort(-p)
    peaks = []
    taken = np.zeros(len(p), bool)
    for k in order:
        if p[k] < thr:
            break
        if taken[max(0, k - sep):k + sep + 1].any():
            continue
        peaks.append(int(k))
        taken[k] = True
    if len(peaks) < 2:
        return None
    peaks.sort()
    mpx = module * pa
    best = 0
    for a in range(len(peaks)):
        n = 0
        last = peaks[a]
        for b in range(a + 1, len(peaks)):
            d = (peaks[b] - last) / mpx
            k = int(round(d))
            if k in (1, 2) and abs(d - k) < 0.15:
                n += k
                last = peaks[b]
            elif d > 2.2:
                break
        best = max(best, n)
    return best if best >= 1 else None

def case_rects_mask(shape, cv_x0, cv_y0, fr, cases):
    """Occluder rectangles for cooler cases + door rows (canvas coordinates)."""
    m = np.zeros(shape, bool)
    for (L, side, s, e, D, nd, per, n_len, n_pk) in cases:
        face = L.c + side * L.t / 2.0
        far = face + side * (D + 0.8)
        c0, c1 = sorted([face, far])
        if L.o == 'h':
            x0 = int(fr.ix(s)) - cv_x0
            x1 = int(fr.ix(e)) - cv_x0
            y0 = int(fr.iy(c0)) - cv_y0
            y1 = int(fr.iy(c1)) - cv_y0
        else:
            y0 = int(fr.iy(s)) - cv_y0
            y1 = int(fr.iy(e)) - cv_y0
            x0 = int(fr.ix(c0)) - cv_x0
            x1 = int(fr.ix(c1)) - cv_x0
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(shape[1], x1), min(shape[0], y1)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def detect_multi(tsub, region, ppf, thrs=(0.10, 0.06)):
    """Fixture counts at several ink thresholds (blurred scans need a high threshold, faint
    ones a low one); every detector verifies its block geometry, so the largest count per
    fixture type is kept.  Returns (counts, occluder mask)."""
    best = None
    occ = None
    for t in thrs:
        d = {}
        try:
            r = detect(tsub, region, ppf, d, thr=t)
        except Exception:
            continue
        m = occluder_mask(d['cl'], d.get('occluders', []), [], ppf)
        occ = m if occ is None else (occ | m)
        if best is None:
            best = dict(r)
        else:
            for k, v in r.items():
                best[k] = max(best[k], v)
    return best, occ
