#!/usr/bin/env python3
"""Calibrate the verifier's area and length tolerances.

Simulates an estimator taking off the scan in a PDF measuring tool: they
calibrate the sheet against a printed dimension and then click polygon
vertices and wall endpoints, each click landing a few pixels off. Every
quantity is measured this way many times; the report gives the worst
relative error seen, next to the error the reference raster method makes
and the error of skipping the scale check.

Floors and partitions are measured on A101; ceilings, base and pendant rows
on A102, which was printed at a different reduction. Base is the clicked room
perimeter less each door opening, itself measured with two clicks.

Usage: python dev/tolerance.py [--trials 2000] [--click-px 2.0]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon

import plan as P

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "store-remodel-takeoff"
PPF = 200 * 0.973 / 8                # pixels per foot on the A101 scan
PPF_RCP = 200 * 0.964 / 8            # and on the A102 scan


def noisy_area(poly, rng, click_px, scale_err, ppf=PPF):
    total = 0.0
    for part in getattr(poly, "geoms", [poly]):
        pts = np.asarray(part.exterior.coords[:-1])
        pts = pts + rng.normal(0, click_px / ppf, pts.shape)
        a = Polygon(pts).area
        for hole in part.interiors:
            h = np.asarray(hole.coords[:-1])
            a -= Polygon(h + rng.normal(0, click_px / ppf, h.shape)).area
        total += a
    return total * (1 + scale_err) ** 2


def noisy_perimeter(poly, rng, click_px, ppf):
    pts = np.asarray(poly.exterior.coords[:-1])
    pts = pts + rng.normal(0, click_px / ppf, pts.shape)
    return Polygon(pts).exterior.length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--click-px", type=float, default=2.0)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    p = P.asbuilt(P.build())
    truth = P.ground_truth(p)

    # calibration error: clicking the two ends of the 160'-0" dimension
    def scale_err(ppf=PPF):
        measured = 160 * ppf + rng.normal(0, args.click_px, 2).sum() * np.sqrt(0.5)
        return measured / (160 * ppf) - 1

    worst = {}
    finishes = list(truth["floor_sf"])
    for f in finishes:
        poly = p.floor(f)
        errs = [noisy_area(poly, rng, args.click_px, scale_err()) / truth["floor_sf"][f] - 1
                for _ in range(args.trials)]
        worst[f] = (float(np.percentile(np.abs(errs), 99)), float(np.max(np.abs(errs))))
    for kind in truth["wall_lf"]:
        walls = [w for w in p.walls if w.kind == kind]
        errs = []
        for _ in range(args.trials):
            s = 1 + scale_err()
            total = 0.0
            for w in walls:
                ends = np.array([[w.x1, w.y1], [w.x2, w.y2]]) + rng.normal(0, args.click_px / PPF, (2, 2))
                total += np.abs(ends[1] - ends[0]).sum()
            errs.append(total * s / truth["wall_lf"][kind] - 1)
        worst[kind] = (float(np.percentile(np.abs(errs), 99)), float(np.max(np.abs(errs))))

    rooms = []
    for poly, numbers in P.room_polygons(p):
        row = p.schedule[numbers[0]]
        doors = [d.width for d in p.doors if any(poly.buffer(0.05).contains(pt) for pt in P.door_sides(p, d))]
        rooms.append((poly, row["ceiling"], row["base"], doors))
    for clg in truth["ceilings_sf"]:
        polys = [r[0] for r in rooms if r[1] == clg]
        errs = []
        for _ in range(args.trials):
            s = scale_err(PPF_RCP)
            errs.append(sum(noisy_area(q, rng, args.click_px, s, PPF_RCP) for q in polys) / truth["ceilings_sf"][clg] - 1)
        worst[clg] = (float(np.percentile(np.abs(errs), 99)), float(np.max(np.abs(errs))))
    for bas in truth["base_lf"]:
        picked = [r for r in rooms if r[2] == bas]
        errs = []
        for _ in range(args.trials):
            s = 1 + scale_err(PPF_RCP)
            total = 0.0
            for poly, _, _, doors in picked:
                total += noisy_perimeter(poly, rng, args.click_px, PPF_RCP)
                total -= sum(w + rng.normal(0, args.click_px / PPF_RCP, 2).sum() * np.sqrt(0.5) for w in doors)
            errs.append(total * s / truth["base_lf"][bas] - 1)
        worst[bas] = (float(np.percentile(np.abs(errs), 99)), float(np.max(np.abs(errs))))
    errs = []
    for _ in range(args.trials):
        s = 1 + scale_err(PPF_RCP)
        total = 0.0
        for row in p.l3_rows:
            ends = np.asarray(row, dtype=float) + rng.normal(0, args.click_px / PPF_RCP, (len(row), 2))
            total += LineString(ends).length
        errs.append(total * s / truth["lights_lf"]["L3"] - 1)
    worst["L3"] = (float(np.percentile(np.abs(errs), 99)), float(np.max(np.abs(errs))))

    oracle = json.loads(subprocess.run(
        [sys.executable, str(TASK / "solution" / "takeoff.py"), str(TASK / "environment/data/drawings/store-0417-asbuilt.pdf"),
         "--redlines", str(TASK / "solution" / "redlines.json"), "--rooms", str(TASK / "solution" / "rooms.json")],
        capture_output=True, text=True, check=True).stdout)
    print(f"{'quantity':8} {'truth':>9} {'digitize p99':>13} {'worst':>7} {'raster':>8} {'unscaled':>9} {'tolerance':>10}")
    nominal = 25 / PPF - 1            # using 200 dpi and 1/8" = 1'-0" without checking the scale
    nominal_rcp = 25 / PPF_RCP - 1
    for f in finishes:
        r = oracle["floor_sf"][f] / truth["floor_sf"][f] - 1
        tol = max(0.02, 8 / truth["floor_sf"][f])
        print(f"{f:8} {truth['floor_sf'][f]:9.1f} {worst[f][0]:12.2%} {worst[f][1]:7.2%} {r:+8.2%} "
              f"{(1 + nominal) ** 2 - 1:+9.2%} {tol:10.2%}")
    for k in truth["wall_lf"]:
        r = oracle["wall_lf"][k] / truth["wall_lf"][k] - 1
        tol = max(0.015, 2 / truth["wall_lf"][k])
        print(f"{k:8} {truth['wall_lf'][k]:9.2f} {worst[k][0]:12.2%} {worst[k][1]:7.2%} {r:+8.2%} {nominal:+9.2%} {tol:10.2%}")
    for k in truth["ceilings_sf"]:
        r = oracle["ceilings_sf"][k] / truth["ceilings_sf"][k] - 1
        tol = max(0.02, 8 / truth["ceilings_sf"][k])
        print(f"{k:8} {truth['ceilings_sf'][k]:9.1f} {worst[k][0]:12.2%} {worst[k][1]:7.2%} {r:+8.2%} "
              f"{(1 + nominal_rcp) ** 2 - 1:+9.2%} {tol:10.2%}")
    for sec, k in [("base_lf", "RB-1"), ("base_lf", "CTB-1"), ("lights_lf", "L3")]:
        r = oracle[sec][k] / truth[sec][k] - 1
        tol = max(0.015, 2 / truth[sec][k])
        print(f"{k:8} {truth[sec][k]:9.2f} {worst[k][0]:12.2%} {worst[k][1]:7.2%} {r:+8.2%} {nominal_rcp:+9.2%} {tol:10.2%}")


if __name__ == "__main__":
    main()
