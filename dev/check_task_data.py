#!/usr/bin/env python3
"""Run a takeoff tool on the task's own scans: the three examples (against their
hand takeoffs) and the hidden stores (against their plan models), with the
verifier's tolerances. Prints one line per scan and the run time.

Usage: python dev/check_task_data.py [--tool PATH] [--jobs N]
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "floorplan-takeoff-tool"
sys.path.insert(0, str(ROOT / "dev"))

import plan as plan_mod  # noqa: E402
from check_oracle import compare  # noqa: E402


def run(args):
    tool, pdf, truth = args
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        t = time.time()
        r = subprocess.run([sys.executable, str(tool), str(pdf), str(out)], capture_output=True, text=True, cwd=tmp)
        dt = time.time() - t
        if r.returncode:
            return pdf.name, dt, [f"tool failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}"]
        return pdf.name, dt, compare(truth, json.loads(out.read_text()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default=str(TASK / "solution" / "takeoff.py"))
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    jobs = []
    for pdf in sorted((TASK / "environment" / "data" / "examples").glob("*.pdf")):
        jobs.append((a.tool, pdf, json.loads(pdf.with_suffix("").with_suffix(".takeoff.json").read_text())))
    for pdf in sorted((TASK / "tests" / "stores").glob("*.pdf")):
        jobs.append((a.tool, pdf, plan_mod.ground_truth(plan_mod.build(int(pdf.stem)))))
    bad = 0
    with Pool(a.jobs) as pool:
        for name, dt, wrong in pool.imap(run, jobs):
            bad += bool(wrong)
            print(f"{'ok ' if not wrong else 'BAD'} {name:22} {dt:6.1f}s  " + "; ".join(wrong), flush=True)
    print(f"{len(jobs) - bad}/{len(jobs)} scans fully correct")


if __name__ == "__main__":
    main()
