"""Fixture detection by template matching against the chain's standard blocks."""
import numpy as np
import cv2
from scipy import ndimage as ndi

SS = 4  # supersampling for template rendering


def _rot_pts(pts, k, W, H):
    """Rotate points (ft) by k*90 deg clockwise inside a W x H box."""
    out = []
    for (x, y) in pts:
        for _ in range(k % 4):
            x, y = H - y, x
            W, H = H, W
        out.append((x, y))
    return out


def render(shapes, W, H, sx, sy, k=0, mirror=False, margin=0.35, lw=2.2):
    """Render line shapes (ft coords inside W x H box) into a template image.

    k: number of 90-degree clockwise rotations; mirror: flip x before rotating.
    Returns (template float32, (W_ft, H_ft) of rotated box, margin_px tuple).
    """
    def tx(p):
        x, y = p
        if mirror:
            x = W - x
        ww, hh = W, H
        for _ in range(k % 4):
            x, y = hh - y, x
            ww, hh = hh, ww
        return x, y
    ww, hh = (W, H) if k % 2 == 0 else (H, W)
    mx = margin * sx; my = margin * sy
    tw = int(np.ceil((ww * sx + 2 * mx)))
    th = int(np.ceil((hh * sy + 2 * my)))
    img = np.zeros((th * SS, tw * SS), np.uint8)
    t = max(1, int(round(lw * SS)))
    for s in shapes:
        if s[0] == 'line':
            a = tx((s[1], s[2])); b = tx((s[3], s[4]))
            p0 = (int(round((a[0] * sx + mx) * SS)), int(round((a[1] * sy + my) * SS)))
            p1 = (int(round((b[0] * sx + mx) * SS)), int(round((b[1] * sy + my) * SS)))
            cv2.line(img, p0, p1, 255, t, cv2.LINE_AA)
        elif s[0] == 'rect':
            x0, y0, x1, y1 = s[1:]
            for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                a = tx(a); b = tx(b)
                p0 = (int(round((a[0] * sx + mx) * SS)), int(round((a[1] * sy + my) * SS)))
                p1 = (int(round((b[0] * sx + mx) * SS)), int(round((b[1] * sy + my) * SS)))
                cv2.line(img, p0, p1, 255, t, cv2.LINE_AA)
        elif s[0] == 'circle':
            c = tx((s[1], s[2]))
            cc = (int(round((c[0] * sx + mx) * SS)), int(round((c[1] * sy + my) * SS)))
            rr = int(round(s[3] * 0.5 * (sx + sy) * SS))
            cv2.circle(img, cc, rr, 255, t, cv2.LINE_AA)
    tpl = cv2.resize(img.astype(np.float32) / 255.0, (tw, th), interpolation=cv2.INTER_AREA)
    tpl = cv2.GaussianBlur(tpl, (0, 0), 0.7)
    return tpl, (ww, hh), (mx, my)


# --- block definitions (ft) in canonical orientation ---------------------------

def produce_table():
    return [('rect', 0, 0, 5, 4), ('line', 0, 0, 5, 4), ('line', 5, 0, 0, 4)], 5.0, 4.0


def self_checkout():
    return [('rect', 0, 0, 3, 3), ('circle', 1.5, 1.5, 0.8)], 3.0, 3.0


def checkout():
    # vertical: belt end at top (y=0), X-box at bottom, side square on the right
    W, H = 3.7, 12.0
    return [('rect', 0, 0, 2.5, 12), ('line', 0, 9, 2.5, 9),
            ('line', 0, 9, 2.5, 12), ('line', 2.5, 9, 0, 12),
            ('line', 1.25, 0.5, 1.25, 8.5),
            ('line', 2.5, 4.8, 3.7, 4.8), ('line', 3.7, 4.8, 3.7, 6.0), ('line', 3.7, 6.0, 2.5, 6.0)], W, H


def gondola_section(L):
    return [('rect', 0, 0, 4, L), ('line', 2, 0, 2, L)], 4.0, float(L)


def endcap():
    # top end of a vertical run: 4 wide, 1.5 deep, diagonal
    return [('rect', 0, 0, 4, 1.5), ('line', 0, 1.5, 4, 0)], 4.0, 1.5


COOLER_GAP = 0.25  # template starts this far in front of the wall face


def cooler_door():
    # one reach-in door with the (blank) case behind it; case above (wall face
    # COOLER_GAP above the box), door strip 2.27 x 0.6 at the case front.
    a = 2.4 - COOLER_GAP
    return [('rect', 0, a, 2.27, a + 0.6), ('line', 0, a, 2.27, a + 0.6)], 2.27, a + 0.6


def match_all(ink, region, tpls, thr, nms_frac=0.6):
    """Run template variants over the region; return detections after NMS.

    tpls: list of (name, tpl_img, (wft,hft), (mx,my)).
    Returns list of dicts {name, score, x, y, w, h} in pixel coords (box of the block).
    """
    x0, y0, x1, y1 = region
    img = ink[y0:y1, x0:x1]
    dets = []
    for name, tpl, (wf, hf), (mx, my) in tpls:
        th, tw = tpl.shape
        if th >= img.shape[0] or tw >= img.shape[1]:
            continue
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        # local maxima above thr
        k = max(3, int(min(th, tw) * 0.5) | 1)
        mxf = ndi.maximum_filter(res, size=k)
        ys, xs = np.nonzero((res >= thr) & (res == mxf))
        if len(ys) > 3000:
            vals = res[ys, xs]
            o = np.argsort(-vals)[:3000]
            ys, xs = ys[o], xs[o]
        for yy, xx in zip(ys, xs):
            dets.append({'name': name, 'score': float(res[yy, xx]),
                         'x': x0 + xx + mx, 'y': y0 + yy + my,
                         'w': tw - 2 * mx, 'h': th - 2 * my})
    # NMS across all variants
    dets.sort(key=lambda d: -d['score'])
    kept = []
    for d in dets:
        ok = True
        for k_ in kept:
            ix = min(d['x'] + d['w'], k_['x'] + k_['w']) - max(d['x'], k_['x'])
            iy = min(d['y'] + d['h'], k_['y'] + k_['h']) - max(d['y'], k_['y'])
            if ix > 0 and iy > 0:
                inter = ix * iy
                if inter > nms_frac * min(d['w'] * d['h'], k_['w'] * k_['h']):
                    ok = False
                    break
        if ok:
            kept.append(d)
    return kept


def variants(defn, sx, sy, rots=(0, 1, 2, 3), mirrors=(False, True), name='', margin=0.35):
    shapes, W, H = defn
    out = []
    seen = set()
    for m in mirrors:
        for k in rots:
            tpl, wh, mxy = render(shapes, W, H, sx, sy, k=k, mirror=m, margin=margin)
            key = (tpl.shape, round(float(tpl.sum()), 1), round(float((tpl * np.arange(tpl.shape[1])).sum()), 0))
            if key in seen:
                continue
            seen.add(key)
            out.append(('%s_r%d%s' % (name, k, 'm' if m else ''), tpl, wh, mxy))
    return out


def door_strip_ok(ink, S, sref, d, sx, sy):
    """Verify a cooler-door candidate: blank case behind it, wall face at its back."""
    name = d['name']
    k = int(name.split('_r')[1][0])
    x, y, w, h = d['x'], d['y'], d['w'], d['h']
    # canonical (k=0): case at top, wall face COOLER_GAP above the box top.
    # The box depth (h for k=0) covers case (2.15) + door (0.6).
    if k in (0, 2):
        depth = h; g = COOLER_GAP * sy
    else:
        depth = w; g = COOLER_GAP * sx
    cfrac = (2.4 - COOLER_GAP) / (2.4 - COOLER_GAP + 0.6)
    if k == 0:
        case = (x + 0.1 * w, y + 0.05 * h, x + 0.9 * w, y + (cfrac - 0.06) * h); wall = ('h', y - g)
    elif k == 1:
        case = (x + (1 - cfrac + 0.06) * w, y + 0.1 * h, x + 0.95 * w, y + 0.9 * h); wall = ('v', x + w + g)
    elif k == 2:
        case = (x + 0.1 * w, y + (1 - cfrac + 0.06) * h, x + 0.9 * w, y + 0.95 * h); wall = ('h', y + h + g)
    else:
        case = (x + 0.05 * w, y + 0.1 * h, x + (cfrac - 0.06) * w, y + 0.9 * h); wall = ('v', x - g)
    cx0, cy0, cx1, cy1 = [int(round(v)) for v in case]
    if cx1 <= cx0 or cy1 <= cy0:
        return False
    cs = S[cy0:cy1, cx0:cx1]
    blank = float((cs > 0.3 * sref).mean())
    if blank > 0.02:
        return False
    if wall[0] == 'h':
        yy = int(round(wall[1]))
        band = S[max(yy - 5, 0):yy + 6, int(x + 0.2 * w):int(x + 0.8 * w)]
        if band.size == 0:
            return False
        strong = float(np.median(band.max(axis=0))) / sref
    else:
        xx = int(round(wall[1]))
        band = S[int(y + 0.2 * h):int(y + 0.8 * h), max(xx - 5, 0):xx + 6]
        if band.size == 0:
            return False
        strong = float(np.median(band.max(axis=1))) / sref
    return strong > 0.6


def detect_fixtures(rot, S, sref, region, sx, sy, floor_mask=None, debug=None):
    ink = cv2.GaussianBlur((1.0 - rot).astype(np.float32), (0, 0), 0.7)
    res = {}
    def keep_in_floor(dets):
        if floor_mask is None:
            return dets
        out = []
        for d in dets:
            cx = int(d['x'] + d['w'] / 2); cy = int(d['y'] + d['h'] / 2)
            if 0 <= cy < floor_mask.shape[0] and 0 <= cx < floor_mask.shape[1] and floor_mask[cy, cx]:
                out.append(d)
        return out
    def run(key, fn):
        try:
            res.update(fn())
        except Exception:
            import traceback
            traceback.print_exc()
            if isinstance(key, str):
                res[key] = []
            else:
                for k in key:
                    res[k] = []

    run('produce_tables', lambda: {'produce_tables': keep_in_floor(match_all(
        ink, region, variants(produce_table(), sx, sy, (0, 1), (False,), 'produce'), 0.70))})
    run('self_checkouts', lambda: {'self_checkouts': keep_in_floor(match_all(
        ink, region, variants(self_checkout(), sx, sy, (0,), (False,), 'selfco'), 0.75))})
    run('checkouts', lambda: {'checkouts': keep_in_floor(match_all(
        ink, region, variants(checkout(), sx, sy, (0, 1, 2, 3), (False, True), 'checkout'), 0.80))})
    run('endcaps', lambda: {'endcaps': keep_in_floor(match_all(
        ink, region, variants(endcap(), sx, sy, (0, 1, 2, 3), (False, True), 'endcap'), 0.75))})

    def gondolas():
        g = variants(gondola_section(4), sx, sy, (0, 1), (False,), 'g4') + \
            variants(gondola_section(3), sx, sy, (0, 1), (False,), 'g3')
        d = keep_in_floor(match_all(ink, region, g, 0.74, nms_frac=0.3))
        return {'gondola_sections_4ft': [x for x in d if x['name'].startswith('g4')],
                'gondola_sections_3ft': [x for x in d if x['name'].startswith('g3')]}
    run(('gondola_sections_4ft', 'gondola_sections_3ft'), gondolas)

    def coolers():
        d = match_all(ink, region, variants(cooler_door(), sx, sy, (0, 1, 2, 3), (False, True), 'cooler',
                                            margin=0.12), 0.72, nms_frac=0.25)
        return {'cooler_doors': [x for x in keep_in_floor(d) if door_strip_ok(ink, S, sref, x, sx, sy)]}
    run('cooler_doors', coolers)
    if debug is not None:
        debug.update(res)
    return {k: len(v) for k, v in res.items()}
