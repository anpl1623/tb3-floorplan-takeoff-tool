"""Finish hatch segmentation with fixtures and columns restored to the floor."""
import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import box as rectangle
from shapely.ops import unary_union


FINISHES = ["LVT-1", "PC-1", "SC-1", "EP-1", "VCT-1", "CT-1", "CPT-1", "WOM-1"]


def room_polygons(walls, box):
    left, top, right, bottom = box
    interior = rectangle(left+16, top+16, right-16, bottom-16)
    partitions = []
    for w in walls:
        if w.kind == "EXT":
            continue
        width = 10 if w.kind == "P1" else 16
        if w.axis == 0:
            partitions.append(rectangle(w.lo, w.c-width/2, w.hi, w.c+width/2))
        else:
            partitions.append(rectangle(w.c-width/2, w.lo, w.c+width/2, w.hi))
    free = interior.difference(unary_union(partitions))
    return list(free.geoms) if hasattr(free, "geoms") else [free]


def polygon_mask(poly, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(np.array(poly.exterior.coords)).astype(np.int32)], 1)
    for ring in poly.interiors:
        cv2.fillPoly(mask, [np.rint(np.array(ring.coords)).astype(np.int32)], 0)
    return mask


def hatch_features(a):
    ink = (a < 235).astype(np.uint8)
    size = 17
    kernels = [np.ones((1, size), np.uint8), np.ones((size, 1), np.uint8),
               np.eye(size, dtype=np.uint8), np.fliplr(np.eye(size, dtype=np.uint8))]
    return [cv2.morphologyEx(ink, cv2.MORPH_OPEN, k) for k in kernels]


def horizontal_pitch(a, safe):
    # Projection within the finish removes fixture outlines and annotations.
    # Its high-frequency component separates the closely spaced walk-off mat
    # lines from the seven-inch plank courses, independent of ink density.
    count = safe.sum(1)
    rows = np.flatnonzero(count > 40)
    if len(rows) < 40:
        return None
    lo, hi = rows[0], rows[-1]+1
    profile = (a*safe).sum(1)[lo:hi]/np.maximum(1, count[lo:hi])
    good = count[lo:hi] > 40
    profile = np.interp(np.arange(len(profile)), np.flatnonzero(good), profile[good])
    profile -= ndimage.gaussian_filter1d(profile, 12)
    spectrum = abs(np.fft.rfft(profile*np.hanning(len(profile)), n=8192))
    frequency = np.fft.rfftfreq(8192)
    band = (frequency > .045) & (frequency < .45)
    f = frequency[np.flatnonzero(band)[np.argmax(spectrum[band])]]
    return float(1/f)


def classify_hatch(fs, pitch=None, unoriented=0.):
    h, v, d1, d2 = fs
    if min(d1, d2) > .012 and min(d1, d2) > .3*max(d1, d2):
        return "CT-1"
    if max(d1, d2) > .012 and max(d1, d2) > 1.7*max(h, v):
        return "CPT-1" if d1 > d2 else "EP-1"
    if pitch is not None and pitch < 7 and h > v*2:
        return "WOM-1"
    if h > .02 and v > .02 and min(h, v) > .5*max(h, v):
        return "VCT-1"
    if h > .025 and h > v*2:
        return "LVT-1"
    return "SC-1" if unoriented > .018 else "PC-1"


def measure_floors(a, walls, box, debug=None, fixture_mask=None):
    polys = room_polygons(walls, box)
    features = hatch_features(a)
    residual = ((a < 235) & ~(features[0] | features[1] | features[2] | features[3]).astype(bool))
    ink = cv2.dilate((a < 170).astype(np.uint8), np.ones((3, 3), np.uint8))
    if fixture_mask is not None:
        ink[fixture_mask.astype(bool)] = 0
    records = []
    totals = dict.fromkeys(FINISHES, 0.)
    viz = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    for ri, poly in enumerate(polys):
        room = polygon_mask(poly, a.shape)
        free = room & (1-ink)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(free)
        seeds = np.zeros(a.shape, np.uint8)
        room_records = []
        for i in range(1, n):
            if stats[i, 4] < 24**2*20:
                continue
            region = (labels == i).astype(np.uint8)
            safe = cv2.erode(region, np.ones((17, 17), np.uint8)).astype(bool)
            if fixture_mask is not None:
                safe &= ~fixture_mask.astype(bool)
            if safe.sum() < 100:
                safe = region.astype(bool)
            fs = [float(f[safe].mean()) for f in features]
            unoriented = float(residual[safe].mean())
            pitch = horizontal_pitch(a, safe) if fs[0] > .025 and fs[0] > 2*fs[1] else None
            kind = classify_hatch(fs, pitch, unoriented)
            seeds[region.astype(bool)] = FINISHES.index(kind)+1
            rec = {"room": ri, "region": i, "area": stats[i, 4]/24**2,
                   "finish": kind, "features": fs, "horizontal_pitch": pitch, "unoriented": unoriented,
                   "center": centers[i].tolist(), "room_area": poly.area/24**2}
            records.append(rec)
            room_records.append(rec)
            color = np.random.default_rng(ri*10000+i).integers(40, 230, 3)
            viz[region.astype(bool)] = .45*viz[region.astype(bool)] + .55*color
            cv2.putText(viz, str(len(records)-1), tuple(np.rint(centers[i]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 230), 3)
        if not room_records:
            raise ValueError("A floor region could not be classified")
        kinds = {r["finish"] for r in room_records}
        if len(kinds) == 1:
            totals[next(iter(kinds))] += poly.area/24**2
        else:
            # Restore the ink of boundaries, annotations, fixtures and columns.
            # Each receives the finish of its nearest unambiguous floor pixel.
            l, t, r, b = map(int, np.rint(poly.bounds))
            local = seeds[t:b+1, l:r+1]
            room_local = room[t:b+1, l:r+1].astype(bool)
            nearest = ndimage.distance_transform_edt(local == 0, return_distances=False, return_indices=True)
            completed = local[tuple(nearest)]
            pixel_area = poly.area/room_local.sum()/24**2
            for code in kinds:
                totals[code] += float(((completed == FINISHES.index(code)+1) & room_local).sum())*pixel_area
    if debug is not None:
        cv2.imwrite(str(debug / "floor_regions.png"), viz)
    return totals, records
