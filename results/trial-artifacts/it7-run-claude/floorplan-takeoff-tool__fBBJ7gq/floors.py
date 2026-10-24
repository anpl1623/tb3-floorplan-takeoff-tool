"""Floor finish areas: floor mask from the wall model, zone cells split by
heavy finish-transition lines, hatch classification against the legend."""
import numpy as np
import cv2
from scipy import ndimage as ndi

import layout
import texture

NOMINAL_T = {0: 8.0 / 12, 1: 8.0 / 12, 2: 7.25 / 12, 3: 4.875 / 12}


def wall_thickness(w, ppf):
    nom = NOMINAL_T[w['typ']] * ppf
    if abs(w['d'] - nom) < 0.15 * ppf:
        return nom
    return w['d']


def floor_mask(shape, ppf, wres):
    H, W = shape
    top, bot, lef, rig = wres['ext_box']
    dext = wres['dext']
    te = NOMINAL_T[0] * ppf if abs(dext - NOMINAL_T[0] * ppf) < 0.15 * ppf else dext
    F = np.zeros((H, W), np.uint8)
    ya, yb = int(round(top + te / 2)), int(round(bot - te / 2))
    xa, xb = int(round(lef + te / 2)), int(round(rig - te / 2))
    F[ya:yb, xa:xb] = 1
    for w in wres['segs']:
        t = wall_thickness(w, ppf)
        a0, a1 = int(round(w['e0'])), int(round(w['e1']))
        c0, c1 = int(round(w['c'] - t / 2)), int(round(w['c'] + t / 2))
        if w['o'] == 'h':
            F[c0:c1, a0:a1] = 0
        else:
            F[a0:a1, c0:c1] = 0
    return F, (xa, ya, xb, yb)


def line_map(img, sigma=1.5, r=4):
    """Smoothed ink with large dark areas removed (white top-hat): keeps strokes
    thinner than ~2r px, so solid fills don't count as lines."""
    sm = cv2.GaussianBlur(img, (0, 0), sigma)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(sm, cv2.MORPH_TOPHAT, k)


def ext_outline_strength(kn, ppf, wres, sigma=1.5):
    """Smoothed peak darkness of the exterior wall outlines (heavy-pen reference)."""
    top, bot, lef, rig = wres['ext_box']
    de = wres['dext']
    H, W = kn.shape
    vals = []
    for frac in (0.25, 0.45, 0.65):
        xa, xb = int(lef + frac * (rig - lef)), int(lef + (frac + 0.1) * (rig - lef))
        for yc in (top - de / 2, top + de / 2, bot - de / 2, bot + de / 2):
            y0 = int(round(yc)) - 6
            if y0 < 0 or y0 + 13 > H:
                continue
            sub = cv2.GaussianBlur(kn[y0:y0 + 13, xa:xb], (0, 0), sigma)
            vals.append(float(sub.mean(axis=1).max()))
    return float(np.median(vals)) if vals else 0.6


def zone_cells(kn, F, ppf, bbox, ref, rel=0.6):
    xa, ya, xb, yb = bbox
    m = 6
    xa0, ya0 = max(0, xa - m), max(0, ya - m)
    crop = kn[ya0:yb + m, xa0:xb + m]
    sm = cv2.GaussianBlur(crop, (0, 0), 1.5)
    sl = (slice(ya - ya0, ya - ya0 + (yb - ya)), slice(xa - xa0, xa - xa0 + (xb - xa)))
    sub = sm[sl]
    Fs = F[ya:yb, xa:xb]
    HL = ((sub > rel * ref) & (Fs > 0)).astype(np.uint8)
    # solid-filled zones (large fat dark areas) are not lines; small solids such
    # as columns stay so that lines running through them remain closed
    HLc = cv2.morphologyEx(HL, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dt = cv2.distanceTransform(HLc, cv2.DIST_L2, 3)
    core = (dt > 4.5).astype(np.uint8)
    if core.any():
        nc, lc, stc, _ = cv2.connectedComponentsWithStats(core, 8)
        bigc = np.zeros(nc, bool)
        bigc[1:] = stc[1:, 4] >= 4.0 * ppf * ppf
        if bigc.any():
            fat = bigc[lc].astype(np.uint8)
            fat = cv2.dilate(fat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            HL[fat > 0] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(HL, 8)
    keep = np.zeros(n, bool)
    big = np.maximum(st[:, 2], st[:, 3]) >= 2.0 * ppf
    keep[1:] = big[1:]
    HL2 = keep[lab].astype(np.uint8)
    # bridge short gaps between collinear straight line pieces (lines hidden
    # under white-masked fixtures or tags): splitting a zone is harmless,
    # merging two finishes is not
    Lk = max(3, int(round(1.5 * ppf)))
    Gk = max(3, int(round(6.0 * ppf)))
    hl = cv2.morphologyEx(HL2, cv2.MORPH_OPEN, np.ones((1, Lk), np.uint8))
    hl = cv2.morphologyEx(hl, cv2.MORPH_CLOSE, np.ones((1, Gk), np.uint8))
    vl = cv2.morphologyEx(HL2, cv2.MORPH_OPEN, np.ones((Lk, 1), np.uint8))
    vl = cv2.morphologyEx(vl, cv2.MORPH_CLOSE, np.ones((Gk, 1), np.uint8))
    HL2 = ((HL2 > 0) | (hl > 0) | (vl > 0)).astype(np.uint8)
    # extend dangling straight line ends to a wall within 6 ft (line ends hidden
    # under fixtures standing against the wall)
    Hh, Ww = Fs.shape
    for img_l, horiz in ((hl, True), (vl, False)):
        nl, ll, stl, _ = cv2.connectedComponentsWithStats(img_l, 8)
        for i in range(1, nl):
            x, y, w, h, a = stl[i]
            if horiz:
                if w < 2 * Lk:
                    continue
                r = y + h // 2
                for end, step in ((x, -1), (x + w - 1, 1)):
                    j = end + step
                    cnt = 0
                    while 0 <= j < Ww and Fs[r, j] > 0 and cnt <= Gk:
                        j += step
                        cnt += 1
                    if 0 < cnt <= Gk and (j < 0 or j >= Ww or Fs[r, j] == 0):
                        lo_, hi_ = sorted((end, min(max(j, 0), Ww - 1)))
                        HL2[max(0, r - 2):r + 3, lo_:hi_ + 1] = 1
            else:
                if h < 2 * Lk:
                    continue
                cx = x + w // 2
                for end, step in ((y, -1), (y + h - 1, 1)):
                    j = end + step
                    cnt = 0
                    while 0 <= j < Hh and Fs[j, cx] > 0 and cnt <= Gk:
                        j += step
                        cnt += 1
                    if 0 < cnt <= Gk and (j < 0 or j >= Hh or Fs[j, cx] == 0):
                        lo_, hi_ = sorted((end, min(max(j, 0), Hh - 1)))
                        HL2[lo_:hi_ + 1, max(0, cx - 2):cx + 3] = 1
    HL2[Fs == 0] = 0
    C = ((Fs > 0) & (HL2 == 0)).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(C, connectivity=4)
    return lab2, n2


def floor_areas(kn, ppf, lg, wres, debug=False):
    H, W = kn.shape
    F, bbox = floor_mask((H, W), ppf, wres)
    xa, ya, xb, yb = bbox
    ref = ext_outline_strength(kn, ppf, wres)
    lab, n = zone_cells(kn, F, ppf, bbox, ref)
    Fs = F[ya:yb, xa:xb]
    # texture matcher with a blank-paper reference
    bb = texture.find_blank_box(kn, 64, 48, region=(int(W * 0.05), int(H * 0.05), int(W * 0.95), int(H * 0.95)))
    tm = texture.TextureMatcher(kn, lg['swatches'], ppf, blank_box=bb, size=(48, 64))
    cls = np.full(n, -1, int)
    objs = ndi.find_objects(lab)
    info = []
    for i in range(1, n):
        sl = objs[i - 1]
        if sl is None:
            continue
        m = lab[sl] == i
        area = m.sum() / ppf ** 2
        if area < 3:
            continue
        mask = np.zeros((H, W), bool)
        mask[ya + sl[0].start:ya + sl[0].stop, xa + sl[1].start:xa + sl[1].stop] = m
        bbox_i = (xa + sl[1].start, ya + sl[0].start, xa + sl[1].stop, ya + sl[0].stop)
        r = tm.classify_mask(mask, bbox=bbox_i)
        if r is None:
            continue
        # a small cell with no hatch at all is a fixture/mask interior: it takes
        # the finish of its surroundings
        if area < 300 and r['n'] > 0 and r['n_blank'] >= 0.95 * r['n']:
            continue
        v = r['votes']
        j = int(np.argmax(v))
        cls[i] = j
        info.append((i, area, j, float(v[j])))
        if debug:
            print('cell', i, 'area %.1f' % area, layout.FINISH_CODES[j], np.round(v, 2))
    # propagate class labels to unclassified floor pixels (lines, small cells),
    # taking sources only from the same wall-enclosed region
    L = np.where(lab > 0, cls[lab], -1)
    L[Fs == 0] = -1
    rooms, nr = ndi.label(Fs > 0)
    for k, sl in enumerate(ndi.find_objects(rooms), start=1):
        if sl is None:
            continue
        rm = rooms[sl] == k
        Ls = L[sl]
        need = rm & (Ls < 0)
        if not need.any():
            continue
        src = rm & (Ls >= 0)
        if not src.any():
            continue
        d, (iy, ix) = ndi.distance_transform_edt(~src, return_indices=True)
        filled = Ls[iy, ix]
        Ls[need] = filled[need]
        L[sl] = Ls
    if (L[Fs > 0] < 0).any() and (L >= 0).any():
        d, (iy, ix) = ndi.distance_transform_edt(L < 0, return_indices=True)
        L = np.where((Fs > 0) & (L < 0), L[iy, ix], L)
    L[Fs == 0] = -1
    areas = {}
    for j, code in enumerate(layout.FINISH_CODES):
        areas[code] = float((L == j).sum()) / ppf ** 2
    return areas
