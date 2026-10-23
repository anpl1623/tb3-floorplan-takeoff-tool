"""Scan-dependent parameters shared by the detectors.

K is the thin-line contrast factor measured on the graphic scale's outline
(1.0 for the reference scans; lower for blurrier/lighter scans).  Thresholds
for thin linework are expressed as darkness below paper (248) and scaled by K.
"""
PAPER = 248.0
K = 1.0


def thr(t_ref):
    """Scale a reference intensity threshold (tuned at K=1) to this scan."""
    return PAPER - (PAPER - t_ref) * K
