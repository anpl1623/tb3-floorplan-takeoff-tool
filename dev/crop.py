#!/usr/bin/env python3
"""Crop a generated store's scan (as the reference tool sees it) around plan coordinates.

Usage: python dev/crop.py SEED X0 Y0 X1 Y1 OUT.png   (plan feet)
"""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks" / "floorplan-takeoff-tool" / "solution"))
import takeoff as T  # noqa: E402
from generate import make, write_pdf  # noqa: E402

seed, fx0, fy0, fx1, fy1, out = int(sys.argv[1]), *map(float, sys.argv[2:6]), sys.argv[6]
p, page, _ = make(seed)
pdf = Path(out).with_suffix(".pdf")
write_pdf(page, pdf, p.store)
g, _ = T.deskew(T.load(str(pdf)))
foot, outer = T.building(g)
ppf = T.scale_bar(g, outer)
x0, _, _, y1 = outer
cv2.imwrite(out, g[int(y1 - fy1 * ppf):int(y1 - fy0 * ppf), int(x0 + fx0 * ppf):int(x0 + fx1 * ppf)])
