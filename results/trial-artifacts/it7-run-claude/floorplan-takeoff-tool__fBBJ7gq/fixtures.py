"""Fixture detection: template matching for compact blocks, structural analysis
for gondola runs."""
import numpy as np
import cv2

SS = 4  # supersampling for template rendering

# Block geometry in feet (local coordinates, x right, y down)
BLOCKS = {
    'produce_tables': dict(size=(5.0, 4.0), rects=[(0, 0, 5.0, 4.0)],
                           lines=[(0, 0, 5.0, 4.0), (0, 4.0, 5.0, 0)], circles=[]),
    'self_checkouts': dict(size=(3.0, 3.0), rects=[(0, 0, 3.0, 3.0)], lines=[],
                           circles=[(1.5, 1.5, 0.8)]),
    'checkouts': dict(size=(3.7, 12.0),
                      rects=[(0, 0, 2.5, 12.0), (0, 9.0, 2.5, 12.0), (2.5, 4.8, 3.7, 6.0)],
                      lines=[(0, 9.0, 2.5, 12.0), (0, 12.0, 2.5, 9.0), (1.25, 0.4, 1.25, 8.6)],
                      circles=[]),
    'cooler_doors': dict(size=(2.3, 0.6), rects=[(0, 0, 2.3, 0.6)],
                         lines=[(0, 0, 2.3, 0.6)], circles=[]),
    'gondola_sections_4ft': dict(size=(4.0, 4.0), rects=[(0, 0, 4.0, 4.0)],
                                 lines=[(2.0, 0, 2.0, 4.0)], circles=[]),
    'gondola_sections_3ft': dict(size=(4.0, 3.0), rects=[(0, 0, 4.0, 3.0)],
                                 lines=[(2.0, 0, 2.0, 3.0)], circles=[]),
    'endcaps': dict(size=(4.0, 1.5), rects=[(0, 0, 4.0, 1.5)],
                    lines=[(0, 1.5, 4.0, 0)], circles=[]),
}


def render_block(spec, ppf, lw_px=1.8, blur=1.0, margin_px=3):
    w_ft, h_ft = spec['size']
    W = int(round(w_ft * ppf)) + 2 * margin_px + 1
    H = int(round(h_ft * ppf)) + 2 * margin_px + 1
    canvas = np.zeros((H * SS, W * SS), np.uint8)

    def P(x, y):
        return (int(round((x * ppf + margin_px + 0.5) * SS)), int(round((y * ppf + margin_px + 0.5) * SS)))
    t = max(1, int(round(lw_px * SS)))
    for (x0, y0, x1, y1) in spec['rects']:
        cv2.rectangle(canvas, P(x0, y0), P(x1, y1), 255, t, cv2.LINE_AA)
    for (x0, y0, x1, y1) in spec['lines']:
        cv2.line(canvas, P(x0, y0), P(x1, y1), 255, t, cv2.LINE_AA)
    for (cx, cy, r) in spec['circles']:
        cv2.circle(canvas, P(cx, cy), int(round(r * ppf * SS)), 255, t, cv2.LINE_AA)
    img = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def variants(t, sym_rot180=False, mirror=True):
    out = []
    for k in range(4):
        r = np.rot90(t, k)
        out.append(np.ascontiguousarray(r))
        if mirror:
            out.append(np.ascontiguousarray(np.fliplr(r)))
    # dedupe identical shapes
    uniq = []
    for v in out:
        if not any(u.shape == v.shape and np.abs(u - v).max() < 1e-3 for u in uniq):
            uniq.append(v)
    return uniq


def nms_peaks(R, thr, rad_x, rad_y):
    """Local maxima of R above thr with suppression window."""
    k = np.ones((2 * rad_y + 1, 2 * rad_x + 1), np.uint8)
    mx = cv2.dilate(R, k)
    ys, xs = np.nonzero((R >= mx) & (R > thr))
    return list(zip(xs, ys, R[ys, xs]))


def match_blocks(kimg, ppf, region, thr=0.45, debug=False):
    """Return detections dict name -> list of (cx, cy, score, w, h)."""
    x0, y0, x1, y1 = region
    sub = kimg[y0:y1, x0:x1]
    img = cv2.GaussianBlur(np.clip(sub, 0, 1).astype(np.float32), (0, 0), 1.0)
    dets = {}
    for name, spec in BLOCKS.items():
        t = render_block(spec, ppf)
        vs = variants(t)
        cand = []
        for v in vs:
            if v.shape[0] >= img.shape[0] or v.shape[1] >= img.shape[1]:
                continue
            R = cv2.matchTemplate(img, v, cv2.TM_CCOEFF_NORMED)
            h, w = v.shape
            pk = nms_peaks(R, thr, max(2, w // 3), max(2, h // 3))
            for (px, py, sc) in pk:
                cand.append((x0 + px + w / 2.0, y0 + py + h / 2.0, float(sc), w, h))
        # NMS across variants
        cand.sort(key=lambda c: -c[2])
        keep = []
        for c in cand:
            ok = True
            for k_ in keep:
                if abs(c[0] - k_[0]) < 0.5 * min(c[3], k_[3]) and abs(c[1] - k_[1]) < 0.5 * min(c[4], k_[4]):
                    ok = False
                    break
            if ok:
                keep.append(c)
        dets[name] = keep
    return dets


def _line_maps(img, ppf, thr=0.3):
    """Binary maps of thin vertical (Vb) and horizontal (Hb) strokes."""
    k = max(3, int(round(0.25 * ppf)))
    kb = max(3, int(round(0.4 * ppf)))
    V = cv2.blur(cv2.erode(img, np.ones((k, 1), np.uint8)), (1, kb))
    H = cv2.blur(cv2.erode(img, np.ones((1, k), np.uint8)), (kb, 1))
    return (V > thr).astype(np.uint8), (H > thr).astype(np.uint8)


def _peak_positions(prof, thr, min_sep):
    out = []
    n = len(prof)
    i = 0
    while i < n:
        if prof[i] > thr:
            j = i
            while j < n and prof[j] > thr:
                j += 1
            seg = prof[i:j]
            w = seg - thr + 1e-6
            out.append(float(i + np.sum(np.arange(len(seg)) * w) / np.sum(w)))
            i = j
        else:
            i += 1
    merged = []
    for p in out:
        if merged and p - merged[-1] < min_sep:
            merged[-1] = (merged[-1] + p) / 2.0
        else:
            merged.append(p)
    return merged


def gondola_runs(kimg, ppf, region, debug=False):
    """Detect double-sided gondola runs (4 ft deep with a centre line).
    Returns list of runs with section lengths and endcaps."""
    x0, y0, x1, y1 = region
    runs = []
    for orient in ('v', 'h'):
        sub = np.clip(kimg[y0:y1, x0:x1], 0, 1).astype(np.float32)
        if orient == 'h':
            sub = np.ascontiguousarray(sub.T)
        Vb, Hb = _line_maps(sub, ppf)
        Vd = cv2.dilate(Vb, np.ones((1, 5), np.uint8))
        s2 = int(round(2.0 * ppf))
        s4 = int(round(4.0 * ppf))
        Hh, Ww = Vd.shape
        T = np.zeros_like(Vd)
        T[:, :Ww - s4] = Vd[:, :Ww - s4] & Vd[:, s2:Ww - s4 + s2] & Vd[:, s4:]
        T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, np.ones((int(round(0.5 * ppf)), 1), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(T, 8)
        cands = []
        for i in range(1, n):
            xx, yy, w, h, a = st[i]
            if h < 2.5 * ppf:
                continue
            cands.append((xx, yy, w, h))
        # merge candidates belonging to the same run (same x within 6px, overlapping y)
        cands.sort()
        used = [False] * len(cands)
        for i, c in enumerate(cands):
            if used[i]:
                continue
            xx, yy, w, h = c
            ya, yb = yy, yy + h
            xs = [xx + w / 2.0]
            for j in range(i + 1, len(cands)):
                if used[j]:
                    continue
                c2 = cands[j]
                if abs((c2[0] + c2[2] / 2.0) - xs[0]) < 6 and c2[1] < yb + 2 * ppf and c2[1] + c2[3] > ya - 2 * ppf:
                    used[j] = True
                    ya, yb = min(ya, c2[1]), max(yb, c2[1] + c2[3])
            # refine left line position: column profile of Vb near xs
            xc = int(round(xs[0]))
            prof = Vb[ya:yb, max(0, xc - 8):xc + 9].mean(axis=0)
            xl = max(0, xc - 8) + int(np.argmax(prof))
            # cross lines: horizontal strokes spanning the run width
            m = int(round(0.25 * ppf))
            ylo = max(0, ya - int(3 * ppf))
            yhi = min(Hh, yb + int(3 * ppf))
            P = Hb[ylo:yhi, xl + m:xl + s4 - m].mean(axis=1)
            cl = [ylo + p for p in _peak_positions(P, 0.75, 0.5 * ppf)]
            # the run's long sides must exist between consecutive cross lines
            segs = []
            for a_, b_ in zip(cl[:-1], cl[1:]):
                ia, ib = int(round(a_ + 3)), int(round(b_ - 3))
                if ib <= ia:
                    continue
                left = Vd[ia:ib, max(0, xl - 2):xl + 3].max(axis=1).mean()
                right = Vd[ia:ib, xl + s4 - 2:xl + s4 + 3].max(axis=1).mean()
                center = Vd[ia:ib, xl + s2 - 2:xl + s2 + 3].max(axis=1).mean()
                L = (b_ - a_) / ppf
                segs.append(dict(a=a_, b=b_, L=L, left=float(left), right=float(right), center=float(center)))
            # keep the maximal chain of segments with both long sides present
            chain = [sg for sg in segs if sg['left'] > 0.8 and sg['right'] > 0.8]
            runs.append(dict(orient=orient, x=xl, ya=ya, yb=yb, segs=chain, all=segs, region=region))
            if debug:
                print(orient, 'run x', xl, 'y', ya, yb, 'segs', [(round(sg['L'], 2), round(sg['center'], 2)) for sg in chain])
    return runs


def count_gondolas(runs, ppf):
    n4 = n3 = ne = 0
    for r in runs:
        chain = r['segs']
        for k, sg in enumerate(chain):
            L = sg['L']
            if abs(L - 4.0) < 0.35 and sg['center'] > 0.6:
                n4 += 1
            elif abs(L - 3.0) < 0.35 and sg['center'] > 0.6:
                n3 += 1
            elif abs(L - 1.5) < 0.35:
                ne += 1
    return n4, n3, ne


def joint_nms(dets, thr=0.74, iou_thr=0.3, ppf=None):
    """Resolve overlaps across fixture types; keep the best-scoring template."""
    allc = []
    for name, dl in dets.items():
        for (cx, cy, sc, w, h) in dl:
            if sc >= thr - 0.15:
                allc.append((sc, name, cx, cy, w, h))
    allc.sort(key=lambda t: -t[0])
    kept = []
    for c in allc:
        sc, name, cx, cy, w, h = c
        bx = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
        bad = False
        for k in kept:
            kx = (k[2] - k[4] / 2.0, k[3] - k[5] / 2.0, k[2] + k[4] / 2.0, k[3] + k[5] / 2.0)
            ix = max(0.0, min(bx[2], kx[2]) - max(bx[0], kx[0]))
            iy = max(0.0, min(bx[3], kx[3]) - max(bx[1], kx[1]))
            inter = ix * iy
            if inter <= 0:
                continue
            a1 = w * h
            a2 = k[4] * k[5]
            if inter / min(a1, a2) > iou_thr:
                bad = True
                break
        if not bad:
            kept.append(c)
    acc = [c[0] >= thr for c in kept]
    # row completion: cooler doors come in rows at a 2.5 ft pitch; a weaker
    # match exactly one pitch from an accepted door in the same row is a door
    if ppf:
        pitch = 2.5 * ppf
        tol = 0.2 * ppf
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(kept):
                if acc[i] or c[1] != 'cooler_doors' or c[0] < 0.55:
                    continue
                for j, k in enumerate(kept):
                    if not acc[j] or k[1] != 'cooler_doors' or abs(k[4] - c[4]) > 2 or abs(k[5] - c[5]) > 2:
                        continue
                    horiz = c[4] > c[5]
                    along, across = (abs(c[2] - k[2]), abs(c[3] - k[3])) if horiz else (abs(c[3] - k[3]), abs(c[2] - k[2]))
                    if abs(along - pitch) < tol and across < tol:
                        acc[i] = True
                        changed = True
                        break
    counts = {name: 0 for name in dets}
    final = []
    for c, a in zip(kept, acc):
        if a:
            counts[c[1]] += 1
            final.append(c)
    return counts, final
