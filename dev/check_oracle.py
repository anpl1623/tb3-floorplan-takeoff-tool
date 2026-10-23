#!/usr/bin/env python3
"""Run the reference takeoff tool on generated stores and compare with ground truth.

Every quantity is checked the way the verifier checks it: areas within 2% or
8 SF, lengths within 1.5% or 2 LF, counts exact.

Usage: python dev/check_oracle.py SEED [SEED ...] [--jobs 8]
       python dev/check_oracle.py --range 5000 5040
"""

import argparse
import json
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))
TOOL = ROOT / "tasks" / "floorplan-takeoff-tool" / "solution" / "takeoff.py"

import plan as plan_mod  # noqa: E402
from generate import make, write_pdf  # noqa: E402

AREA_TOL = (0.02, 8.0)
LENGTH_TOL = (0.015, 2.0)


def compare(truth, got):
    bad = []
    for sec, tol in (("floor_sf", AREA_TOL), ("wall_lf", LENGTH_TOL)):
        for k, v in truth[sec].items():
            g = got.get(sec, {}).get(k)
            if g is None or abs(g - v) > max(tol[0] * v, tol[1]):
                bad.append(f"{k} {g} vs {v}")
    for sec in ("doors", "fixtures"):
        for k, v in truth[sec].items():
            g = got.get(sec, {}).get(k)
            if g != v:
                bad.append(f"{k} {g} vs {v}")
    return bad


def check(seed, tool=None):
    p, page, params = make(seed)
    truth = plan_mod.ground_truth(p)
    with tempfile.TemporaryDirectory() as tmp:
        pdf, out = Path(tmp) / "a101.pdf", Path(tmp) / "out.json"
        write_pdf(page, pdf, p.store)
        r = subprocess.run([sys.executable, str(tool or TOOL), str(pdf), str(out)], capture_output=True, text=True,
                           cwd=tmp)
        if r.returncode:
            return seed, params, [f"tool failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}"]
        got = json.loads(out.read_text())
    return seed, params, compare(truth, got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="*")
    ap.add_argument("--range", type=int, nargs=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tool", help="another takeoff tool to check, e.g. an agent's /app/takeoff/takeoff.py")
    args = ap.parse_args()
    seeds = args.seeds + (list(range(*args.range)) if args.range else [])
    if args.tool:
        global TOOL
        TOOL = Path(args.tool).resolve()
    n_bad = 0
    with Pool(args.jobs) as pool:
        from functools import partial
        for seed, params, bad in pool.imap(partial(check, tool=TOOL), seeds):
            n_bad += bool(bad)
            print(f"{'ok ' if not bad else 'BAD'} {seed} scale {params['print_scale']:.3f} skew {params['skew']:+.2f}  "
                  + "; ".join(bad), flush=True)
    print(f"{len(seeds) - n_bad}/{len(seeds)} stores fully correct")


if __name__ == "__main__":
    main()
