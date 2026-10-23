"""Texture descriptors for hatch classification (masked autocorrelation)."""
import numpy as np
import cv2

R_LAG = 40


def _fft_acf(y, m, R):
    """Masked autocovariance of y (already zero outside mask m) for lags |dx|,|dy|<=R."""
    h, w = y.shape
    H = cv2.getOptimalDFTSize(max(h + R + 1, 2 * R + 3))
    W = cv2.getOptimalDFTSize(max(w + R + 1, 2 * R + 3))
    Y = np.fft.rfft2(y, s=(H, W))
    M = np.fft.rfft2(m, s=(H, W))
    num = np.fft.irfft2(Y * np.conj(Y), s=(H, W))
    den = np.fft.irfft2(M * np.conj(M), s=(H, W))
    num = np.fft.fftshift(num)
    den = np.fft.fftshift(den)
    cy, cx = H // 2, W // 2
    out_n = np.zeros((2 * R + 1, 2 * R + 1))
    out_d = np.zeros((2 * R + 1, 2 * R + 1))
    ry = min(R, cy)
    rx = min(R, cx)
    out_n[R - ry:R + ry + 1, R - rx:R + rx + 1] = num[cy - ry:cy + ry + 1, cx - rx:cx + rx + 1]
    out_d[R - ry:R + ry + 1, R - rx:R + rx + 1] = den[cy - ry:cy + ry + 1, cx - rx:cx + rx + 1]
    return out_n, out_d


def describe(D, mask, R=R_LAG, max_win=512, rng=None):
    """Descriptor dict for region `mask` (bool, same shape as D)."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return None
    vals = D[ys, xs]
    mu = float(vals.mean())
    sd = float(vals.std())
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    num_tot = np.zeros((2 * R + 1, 2 * R + 1))
    den_tot = np.zeros((2 * R + 1, 2 * R + 1))
    # tile big regions into windows
    hh, ww = y1 - y0, x1 - x0
    ny = max(1, int(np.ceil(hh / max_win)))
    nx = max(1, int(np.ceil(ww / max_win)))
    for iy in range(ny):
        for ix in range(nx):
            a0 = y0 + iy * hh // ny
            a1 = y0 + (iy + 1) * hh // ny
            b0 = x0 + ix * ww // nx
            b1 = x0 + (ix + 1) * ww // nx
            m = mask[a0:a1, b0:b1].astype(np.float64)
            if m.sum() < 50:
                continue
            if m.shape[0] < 4 or m.shape[1] < 4:
                continue
            y = (D[a0:a1, b0:b1].astype(np.float64) - mu) * m
            num, den = _fft_acf(y, m, R)
            num_tot += num
            den_tot += den
    acov = num_tot / np.maximum(den_tot, 1.0)
    var = acov[R, R]
    rho = acov / max(var, 1e-9)
    return dict(mu=mu, sd=sd, rho=rho, acov=acov, n=int(len(ys)), den=den_tot)


def acov_distance(a, b, excl=2):
    R = a['acov'].shape[0] // 2
    w = np.minimum(a['den'], b['den'])
    w = w / max(w.max(), 1.0)
    w = np.where(w > 0.05, w, 0.0)
    w[R - excl:R + excl + 1, R - excl:R + excl + 1] = 0
    return float(((a['acov'] - b['acov']) ** 2 * w).sum() / max(w.sum(), 1.0))


def distance(a, b):
    """Distance between two descriptors."""
    R = a['rho'].shape[0] // 2
    w = np.minimum(a['den'], b['den'])
    w = w / max(w.max(), 1.0)
    w[R, R] = 0
    w = np.where(w > 0.05, w, 0.0)
    dr = ((a['rho'] - b['rho']) ** 2 * w).sum() / max(w.sum(), 1.0)
    dmu = (a['mu'] - b['mu']) ** 2
    dsd = (a['sd'] - b['sd']) ** 2
    return dict(rho=float(dr), mu=float(dmu), sd=float(dsd))
