#!/usr/bin/env python3
"""Probe the verifier with crafted takeoff tools.

Builds the verifier image from tests/Dockerfile and runs tests/test.sh in a
fresh container per case, with the candidate tool directory mounted where
harbor places the artifact (/app/takeoff). Tools that are missing, broken,
hard-coded, out of tolerance or that try to read the verifier's files must
score 0; the reference tool must score 1, also when it is slightly off within
tolerance.

Usage: python dev/verifier_probe.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "floorplan-takeoff-tool"
IMAGE = "floorplan-takeoff-tool-verifier:probe"
REFERENCE = (TASK / "solution" / "takeoff.py").read_text()
EXAMPLE = next((TASK / "environment" / "data" / "examples").glob("*.takeoff.json")).read_text()

WRAP = '''
import json, sys, runpy
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2] + ".ref"]
runpy.run_path(__file__.replace("takeoff.py", "reference.py"), run_name="__main__")
q = json.load(open(sys.argv[2]))
{edit}
json.dump(q, open(sys.argv[2][:-4], "w"))
'''


def cases():
    yield "no tool", {}, 0
    yield "tool exits with an error", {"takeoff.py": "raise SystemExit(3)\n"}, 0
    yield "tool writes an example's hand takeoff", {
        "takeoff.py": "import sys\nopen(sys.argv[2], 'w').write(open(__file__.replace('takeoff.py', 'example.json')).read())\n",
        "example.json": EXAMPLE}, 0
    yield "tool reads the verifier's plan models and stores", {
        "takeoff.py": ("import sys, json, glob\nsys.path.insert(0, '/tests')\nimport plan\n"
                       "scans = glob.glob('/tests/stores/*.pdf')\n"
                       "json.dump(plan.ground_truth(plan.build(int(scans[0].split('/')[-1][:-4]))), open(sys.argv[2], 'w'))\n")}, 0
    yield "tool leaves a process behind to overwrite the reward", {
        "takeoff.py": ("import subprocess, sys\n"
                       "subprocess.Popen([sys.executable, '-c', 'import time\\n"
                       "for _ in range(3000):\\n"
                       "    time.sleep(0.2)\\n"
                       "    try: open(\\'/logs/verifier/reward.txt\\', \\'w\\').write(\\'1\\')\\n"
                       "    except OSError: pass'], start_new_session=True)\n"
                       "open(sys.argv[2], 'w').write('{}')\n")}, 0
    yield "reference tool, one cooler door short", {
        "reference.py": REFERENCE,
        "takeoff.py": WRAP.format(edit="q['fixtures']['cooler_doors'] -= 1")}, 0
    yield "reference tool, areas 3% high", {
        "reference.py": REFERENCE,
        "takeoff.py": WRAP.format(edit="q['floor_sf'] = {k: v * 1.03 for k, v in q['floor_sf'].items()}")}, 0
    yield "reference tool", {"takeoff.py": REFERENCE}, 1
    yield "reference tool, areas 1% low, lengths 1% high", {
        "reference.py": REFERENCE,
        "takeoff.py": WRAP.format(edit="q['floor_sf'] = {k: v * 0.99 for k, v in q['floor_sf'].items()}\n"
                                       "q['wall_lf'] = {k: v * 1.01 for k, v in q['wall_lf'].items()}")}, 1


def run_case(files):
    with tempfile.TemporaryDirectory() as tmp:
        tool = Path(tmp) / "takeoff"
        tool.mkdir()
        for name, text in files.items():
            (tool / name).write_text(text)
        # harbor uploads the artifact into the container as root; copy it in the same way
        cmd = ["docker", "run", "--rm", "--cpus", "2", "-v", f"{tool}:/mnt/tool:ro", IMAGE, "bash", "-c",
               "rm -rf /app/takeoff && cp -r /mnt/tool /app/takeoff && mkdir -p /logs/verifier && chmod 777 /logs/verifier && "
               "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1; cat /logs/verifier/reward.txt; "
               "grep -E '^(FAILED|PASSED|ERROR)' /logs/verifier/test-stdout.txt | head -3"]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().split("\n")


def main():
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(TASK / "tests")], check=True, capture_output=True)
    failures = 0
    for name, files, want in cases():
        out = run_case(files)
        got = out[0] if out else "?"
        ok = got == str(want)
        failures += not ok
        print(f"{'ok ' if ok else 'BAD'} reward={got} expected={want}  {name}", flush=True)
        if not ok:
            print("    " + "\n    ".join(out[1:]))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
