"""Hatch-pattern matching of plan regions against the legend swatches."""
import numpy as np
import cv2


class TextureMatcher:
    """Shift-invariant patch descriptor: mean & median ink, log spectral energy
    and the normalised power-spectrum shape.  References are built from many
    crops of each legend swatch (so random patterns get realistic tolerances)
    plus a blank-paper reference."""

    def __init__(self, kn, swatches, ppf, blank_box=None, size=(48, 64), inset=5):
        self.kn = kn
        self.ppf = ppf
        self.PH, self.PW = size
        self.win = np.outer(np.hanning(self.PH), np.hanning(self.PW)).astype(np.float32)
        self.noiseP = None
        blank_ds = []
        if blank_box is not None:
            x, y, w, h = blank_box
            Ps = []
            pos = []
            for i in range(3):
                for j in range(3):
                    xx = x + int(i * max(0, w - self.PW) / 2)
                    yy = y + int(j * max(0, h - self.PH) / 2)
                    p = kn[yy:yy + self.PH, xx:xx + self.PW]
                    if p.shape == (self.PH, self.PW):
                        Ps.append(self._power(p))
                        pos.append(p)
            if Ps:
                self.noiseP = np.mean(Ps, axis=0)
                self.noiseE = float(self.noiseP.sum())
                blank_ds = [self.describe(p) for p in pos]
        self.refs = []
        self.crops = []
        for (x, y, w, h) in swatches:
            ds = []
            xa, xb = x + inset, x + w - inset - self.PW
            ya, yb = y + inset, y + h - inset - self.PH
            nx = max(1, int((xb - xa) // max(4, self.PW // 4)) + 1)
            ny = max(1, int((yb - ya) // max(4, self.PH // 4)) + 1)
            for i in range(nx):
                for j in range(ny):
                    xx = int(round(xa + (xb - xa) * (i / max(1, nx - 1))))
                    yy = int(round(ya + (yb - ya) * (j / max(1, ny - 1))))
                    p = kn[yy:yy + self.PH, xx:xx + self.PW]
                    if p.shape == (self.PH, self.PW):
                        ds.append(self.describe(p))
            self.refs.append(self._ref_from(ds))
            self.crops.append(ds)
        self.blank = self._ref_from(blank_ds) if blank_ds else None
        # sparsity of each swatch: fraction of half-size windows without strokes;
        # only the sparsest pattern can explain blank patches in the plan
        sm = None
        sparse = []
        hh, ww = max(8, self.PH // 2), max(8, self.PW // 2)
        for (x, y, w, h) in swatches:
            sub = kn[y + inset:y + h - inset, x + inset:x + w - inset]
            if sub.size == 0:
                sparse.append(0.0)
                continue
            sub = cv2.GaussianBlur(sub.astype(np.float32), (0, 0), 1.0)
            bm = self.blank['m'] if self.blank is not None else 0.02
            thr = bm + 0.1
            tot = 0
            nb_ = 0
            for yy in range(0, sub.shape[0] - hh + 1, 4):
                for xx in range(0, sub.shape[1] - ww + 1, 4):
                    tot += 1
                    if float(sub[yy:yy + hh, xx:xx + ww].max()) < thr:
                        nb_ += 1
            sparse.append(nb_ / float(max(1, tot)))
        self.sparse = sparse
        j = int(np.argmax(sparse)) if sparse else -1
        self.can_blank = [(i == j and sparse[i] > 0.2) for i in range(len(sparse))]
        # drop crops that look like blank paper (sparse patterns): they carry
        # no evidence for their class
        # how often each swatch's own crops look blank (noise can hide sparse dots)
        self.crop_blank = [float(np.mean([self.is_blank(d) for d in ds])) if ds else 0.0
                           for ds in self.crops]
        self.crop_arr = []
        for ci, ds in enumerate(self.crops):
            keep = [d for d in ds if not self.is_blank(d)]
            if not keep:
                keep = ds
            self.crop_arr.append(keep)

    def _power(self, p):
        p = np.clip(p, 0, 1.2).astype(np.float32)
        q = (p - p.mean()) * self.win
        return np.fft.fftshift(np.abs(np.fft.fft2(q)) ** 2)

    def is_blank(self, d):
        if self.noiseP is None or getattr(self, 'blank', None) is None:
            return False
        b = self.blank
        return d[2] < b['e'] + max(0.3, 3.0 * b['se']) and d[0] < b['m'] + 0.02

    @staticmethod
    def _ref_from(ds):
        F = np.mean([d[1] for d in ds], axis=0)
        F /= F.sum() + 1e-12
        m = np.array([d[0] for d in ds])
        e = np.array([d[2] for d in ds])
        md = np.array([d[3] for d in ds])
        return dict(m=float(m.mean()), F=F, e=float(e.mean()), md=float(np.median(md)),
                    sm=float(m.std()), se=float(e.std()), n=len(ds))

    def describe(self, p):
        p = np.clip(p, 0, 1.2).astype(np.float32)
        m = float(p.mean())
        md = float(np.median(p))
        q = (p - m) * self.win
        F = np.fft.fftshift(np.abs(np.fft.fft2(q)) ** 2)
        if self.noiseP is not None:
            F = np.clip(F - self.noiseP, 0, None)
            e0 = 0.05 * self.noiseE
        else:
            e0 = 1e-9
        F = cv2.GaussianBlur(F, (0, 0), 0.8)
        E = float(F.sum())
        return m, F / (E + 1e-12), float(np.log10(E + e0)), md

    @staticmethod
    def sim(d, r):
        bc = float(np.sum(np.sqrt(d[1] * r['F'])))
        et = max(0.3, 2.0 * r['se'])
        mt = max(0.03, 2.0 * r['sm'], 0.15 * r['m'])
        return bc * np.exp(-abs(d[2] - r['e']) / et) * np.exp(-abs(d[0] - r['m']) / mt)

    @staticmethod
    def sim_crop(d, c, et=0.35, mt=0.03):
        bc = float(np.sum(np.sqrt(d[1] * c[1])))
        mt = max(mt, 0.15 * c[0])
        return bc * np.exp(-abs(d[2] - c[2]) / et) * np.exp(-abs(d[0] - c[0]) / mt)

    def patch_scores(self, p):
        d = self.describe(p)
        s = []
        for ds in self.crop_arr:
            v = sorted((self.sim_crop(d, c) for c in ds), reverse=True)
            s.append(float(np.mean(v[:3])))
        s = np.array(s)
        sb = self.sim(d, self.blank) if self.blank is not None else 0.0
        return s, sb, d

    def sample_positions(self, mask, bbox, max_patches=150, rng=None):
        PH, PW = self.PH, self.PW
        x0, y0, x1, y1 = bbox
        sub = mask[y0:y1, x0:x1].astype(np.uint8)
        if sub.shape[0] < PH or sub.shape[1] < PW:
            return None
        er = cv2.erode(sub, np.ones((PH, PW), np.uint8), anchor=(PW // 2, PH // 2),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
        ys, xs = np.nonzero(er)
        if len(ys) == 0:
            return None
        sy, sx = max(1, PH // 3), max(1, PW // 3)
        keep = ((ys - ys.min()) % sy == 0) & ((xs - xs.min()) % sx == 0)
        cy, cx = ys[keep], xs[keep]
        if len(cy) < 10:
            sel = np.linspace(0, len(ys) - 1, min(len(ys), 16)).astype(int)
            cy, cx = ys[sel], xs[sel]
        if len(cy) > max_patches:
            rng = rng or np.random.default_rng(0)
            sel = rng.choice(len(cy), max_patches, replace=False)
            cy, cx = cy[sel], cx[sel]
        return y0 + cy - PH // 2, x0 + cx - PW // 2

    def classify_mask(self, mask, bbox=None, max_patches=150, rng=None):
        """Score a region against the references.  Returns dict with per-class
        scores (upper-quantile of patch similarities), blank fraction etc."""
        if bbox is None:
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                return None
            bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        pos = self.sample_positions(mask, bbox, max_patches, rng)
        if pos is None:
            return None
        PH, PW = self.PH, self.PW
        allsc, blank = [], []
        for Y, X in zip(*pos):
            s, sb, d = self.patch_scores(self.kn[Y:Y + PH, X:X + PW])
            allsc.append(s)
            blank.append(self.is_blank(d))
        allsc = np.array(allsc)
        blank = np.array(blank)
        isblank = blank
        nb = int(isblank.sum())
        sc = allsc[~isblank] if (~isblank).sum() >= 3 else allsc
        q = np.quantile(sc, 0.8, axis=0)
        best = np.bincount(np.argmax(sc, axis=1), minlength=len(self.refs)) / float(len(sc))
        bf = nb / float(max(1, len(allsc)))
        # blank patches support the sparse pattern only if the inked patches
        # do not clearly prefer another pattern
        qmax = float(q.max()) if len(q) else 0.0
        bonus = np.zeros(len(self.refs))
        for i, cb in enumerate(self.can_blank):
            if not cb or q[i] < 0.8 * qmax:
                continue
            # competitors that are themselves often blank explain blank patches
            others = [self.crop_blank[j] for j in range(len(self.refs))
                      if j != i and q[j] >= 0.8 * qmax]
            excess = bf - (max(others) if others else 0.0)
            bonus[i] = 0.5 * max(0.0, excess)
        return dict(votes=q + bonus, raw=q, frac=best, n_blank=nb, n=len(allsc), scores=allsc, blank=blank,
                    blank_frac=bf)


def find_blank_box(kn, PW, PH, region=None):
    """Find a blank paper area (for the noise reference)."""
    H, W = kn.shape
    x0, y0, x1, y1 = region if region is not None else (0, 0, W, H)
    bw, bh = 3 * PW, 3 * PH
    best = None
    small = cv2.resize(kn[y0:y1, x0:x1], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    mx = cv2.dilate(small, np.ones((max(1, bh // 4), max(1, bw // 4)), np.uint8))
    sh, sw = small.shape
    for yy in range(bh // 8, sh - bh // 8, max(1, bh // 8)):
        for xx in range(bw // 8, sw - bw // 8, max(1, bw // 8)):
            v = mx[yy, xx]
            if best is None or v < best[0]:
                best = (v, xx, yy)
    if best is None:
        return None
    v, xx, yy = best
    cx, cy = x0 + xx * 4, y0 + yy * 4
    return (int(cx - bw // 2), int(cy - bh // 2), int(bw), int(bh))
