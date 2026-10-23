#!/usr/bin/env python3
"""Probe the verifier with crafted takeoff files.

Builds the verifier image from tests/Dockerfile and runs tests/test.sh in a
fresh container for each case, with the file mounted where harbor places the
artifact. Exploit attempts and near misses must score 0; correct answers in
any reasonable formatting must score 1.

Usage: python dev/verifier_probe.py
"""

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import plan as P

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "store-remodel-takeoff"
IMAGE = "store-remodel-takeoff-verifier:probe"


def cases(truth, template):
    def scaled(f_area, f_len):
        t = copy.deepcopy(truth)
        for sec in ("floor_sf", "ceilings_sf"):
            t[sec] = {k: v * f_area for k, v in t[sec].items()}
        for sec in ("wall_lf", "base_lf", "lights_lf"):
            t[sec] = {k: v * f_len for k, v in t[sec].items()}
        return t

    def tweak(section, key, delta):
        t = copy.deepcopy(truth)
        t[section][key] += delta
        return t

    yield "empty file", "", 0
    yield "template left as nulls", json.dumps(template), 0
    yield "numbers as strings", json.dumps({s: {k: str(v) for k, v in d.items()} for s, d in truth.items()}), 0
    yield "NaN everywhere", json.dumps({s: {k: float("nan") for k in d} for s, d in truth.items()}), 0
    yield "Infinity everywhere", json.dumps({s: {k: float("inf") for k in d} for s, d in truth.items()}), 0
    yield "booleans for counts", json.dumps({**truth, "doors": {k: True for k in truth["doors"]}}), 0
    yield "nominal scale, not calibrated", json.dumps(scaled((25 / 24.325) ** 2, 25 / 24.325)), 0
    yield "one cooler door short", json.dumps(tweak("fixtures", "cooler_doors", -1)), 0
    yield "pair counted as two singles", json.dumps(
        {**truth, "doors": {**truth["doors"], "pair": 0, "single": truth["doors"]["single"] + 4}}), 0
    yield "gross area of one finish (+3%)", json.dumps(tweak("floor_sf", "LVT-1", truth["floor_sf"]["LVT-1"] * 0.03)), 0
    yield "exact takeoff of the drawing as drawn (redlines ignored)", json.dumps(P.ground_truth(P.build())), 0
    yield "ceilings taken off A102 as drawn (A101 wall changes ignored)", json.dumps(
        {**truth, "ceilings_sf": {**truth["ceilings_sf"], "GYP": truth["ceilings_sf"]["GYP"] - 124.1}}), 0
    yield "EM troffers counted as plain troffers", json.dumps(
        {**truth, "lights": {**truth["lights"], "L1": truth["lights"]["L1"] + truth["lights"]["L1E"], "L1E": 0}}), 0
    yield "exact answer", json.dumps(truth), 1
    yield "within tolerance (-1.5% areas, +1% lengths)", json.dumps(scaled(0.985, 1.01)), 1
    yield "counts as floats", json.dumps({**truth, "fixtures": {k: float(v) for k, v in truth["fixtures"].items()}}), 1


def run_case(content):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        out.mkdir()
        (out / "takeoff.json").write_text(content)
        cmd = ["docker", "run", "--rm", "-v", f"{out}:/app/output:ro", IMAGE, "bash", "-c",
               "mkdir -p /logs/verifier && bash /tests/test.sh >/dev/null 2>&1; cat /logs/verifier/reward.txt"]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def main():
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(TASK / "tests")], check=True, capture_output=True)
    truth = P.ground_truth(P.asbuilt(P.build()))
    template = json.loads((TASK / "environment" / "data" / "takeoff_template.json").read_text())
    failures = 0
    for name, content, want in cases(truth, template):
        got = run_case(content)
        ok = got == str(want)
        failures += not ok
        print(f"{'ok ' if ok else 'BAD'} reward={got} expected={want}  {name}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
