#!/usr/bin/env python3
"""Probe the verifier with crafted plan files.

Builds the verifier image from tests/Dockerfile and runs tests/test.sh in a
fresh container for each case, with the file mounted where harbor places the
artifact. Each rule is broken on its own by a single-store change found by
search, so a pass/fail mix-up in any one test shows up. Plans that break a rule
or miss the target must score 0; the reference plan must score 1 however it is
formatted.

Usage: python dev/verifier_probe.py --plan reference_plan.csv
"""

import argparse
import io
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import rollout as R
from greedy import greedy

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "remodel-rollout-plan"
IMAGE = "remodel-rollout-plan-verifier:probe"
RULES = {"windows": R.check_store_windows, "crews": R.check_crews,
         "districts": R.check_districts, "starts": R.check_starts}


def to_csv(plan, order=None, line_end="\n", fmt="{}"):
    out = io.StringIO()
    out.write("store,crew,start_week" + line_end)
    for s in order or sorted(plan):
        c, t = plan[s]
        out.write(f"{s},{c},{fmt.format(t)}{line_end}")
    return out.getvalue()


def break_one(inst, plan, rule):
    """A copy of plan with one store changed so that only `rule` is violated."""
    for s in sorted(plan):
        for c in sorted(inst.crews):
            for t in range(1, R.LAST_WORK_WEEK + 1):
                p = dict(plan)
                p[s] = (c, t)
                broken = [name for name, check in RULES.items() if check(inst, p)]
                if broken == [rule]:
                    return p, f"{s} -> {c} week {t}"
    raise RuntimeError(f"no single-store change breaks only {rule}")


def cases(inst, ref):
    stores = sorted(ref)
    yield "no plan file", None, 0
    yield "empty file", "", 0
    yield "header only", "store,crew,start_week\n", 0
    yield "wrong header", to_csv(ref).replace("start_week", "start", 1), 0
    yield "one store missing", to_csv({s: ref[s] for s in stores[1:]}), 0
    yield "a store listed twice", to_csv(ref) + f"{stores[0]},{ref[stores[0]][0]},{ref[stores[0]][1]}\n", 0
    yield "unknown crew", to_csv({**ref, stores[0]: ("C10", ref[stores[0]][1])}), 0
    yield "start week not a number", to_csv(ref).replace(f"{stores[0]},{ref[stores[0]][0]},{ref[stores[0]][1]}",
                                                         f"{stores[0]},{ref[stores[0]][0]},wk{ref[stores[0]][1]}"), 0
    for rule in RULES:
        p, how = break_one(inst, ref, rule)
        yield f"breaks only the {rule} rule ({how})", to_csv(p), 0
    g = greedy(inst)
    yield f"greedy plan, feasible, ${R.cost(inst, g)['total']:,.0f}", to_csv(g), 0
    yield f"reference plan, ${R.cost(inst, ref)['total']:,.0f}", to_csv(ref), 1
    yield "reference plan, shuffled rows, CRLF, weeks as 12.0, BOM", \
        "﻿" + to_csv(ref, order=list(reversed(stores)), line_end="\r\n", fmt="{}.0"), 1


def run_case(content):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        out.mkdir()
        if content is not None:
            (out / "plan.csv").write_text(content, newline="")
        cmd = ["docker", "run", "--rm", "-v", f"{out}:/app/output:ro", IMAGE, "bash", "-c",
               "mkdir -p /logs/verifier && bash /tests/test.sh >/dev/null 2>&1; cat /logs/verifier/reward.txt"]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="the reference plan")
    args = ap.parse_args()
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(TASK / "tests")], check=True, capture_output=True)
    inst = R.load(TASK / "environment" / "data")
    ref = R.read_plan(args.plan)
    assert not R.evaluate(inst, ref)[0]
    failures = 0
    for name, content, want in cases(inst, ref):
        got = run_case(content)
        ok = got == str(want)
        failures += not ok
        print(f"{'ok ' if ok else 'BAD'} reward={got} expected={want}  {name}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
