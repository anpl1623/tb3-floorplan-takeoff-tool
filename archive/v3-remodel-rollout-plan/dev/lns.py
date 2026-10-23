#!/usr/bin/env python3
"""Large-neighbourhood search with exact CP-SAT repairs.

Each step frees part of the incumbent plan and re-optimizes it exactly with the
CP-SAT model restricted to that part (cpsat_baseline.build with `allowed`):
  crews     - every store of 2-3 crews whose homes are close; those crews only
  window    - every store under construction in a 6-10 week window; any crew
  district  - every store of one or two districts; any crew

Usage: python dev/lns.py --init plan.csv [--seconds 1800] [--out plan.csv] [--seed 1]
"""

import argparse
import random
import time
from pathlib import Path

from ortools.sat.python import cp_model

import rollout as R
from cpsat_baseline import build, extract

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tasks" / "remodel-rollout-plan" / "environment" / "data"


def neighbourhood(inst, plan, rng):
    kind = rng.choice(["crews", "crews", "window", "district"])
    C = sorted(inst.crews)
    if kind == "crews":
        c0 = rng.choice(C)
        home = inst.crews[c0]["home"]
        near = sorted(C, key=lambda c: inst.miles[home, inst.crews[c]["home"]])
        crews = set(near[:rng.choice([2, 3, 3])])
        free = {s for s, (c, t) in plan.items() if c in crews}
        return kind, {s: (crews, None) for s in free}
    if kind == "window":
        a = rng.randint(1, R.LAST_WORK_WEEK - 6)
        b = a + rng.randint(6, 10)
        free = {s for s, (c, t) in plan.items() if t <= b and t + inst.stores[s]["weeks"] - 1 >= a}
        return kind, {s: (set(C), None) for s in free}
    ds = sorted({st["district"] for st in inst.stores.values()})
    pick = set(rng.sample(ds, rng.choice([1, 2])))
    free = {s for s in inst.stores if inst.stores[s]["district"] in pick}
    return kind, {s: (set(C), None) for s in free}


def solve_sub(inst, plan, free, seconds, workers):
    allowed = {}
    for s in inst.stores:
        if s in free:
            crews, _ = free[s]
            allowed[s] = (crews, [t for (x, t) in inst.lost if x == s])
        else:
            c, t = plan[s]
            allowed[s] = ({c}, [t])
    m, x, y, S, C, starts = build(inst, hint=plan, allowed=allowed)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_workers = workers
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return extract(solver, x, y, S, C, starts), st == cp_model.OPTIMAL
    return None, False


def lns(inst, plan, seconds, workers, seed, step_seconds=20.0, log=print):
    rng = random.Random(seed)
    _, cost = R.evaluate(inst, plan)
    best = cost["total"]
    t0 = time.time()
    it = 0
    while time.time() - t0 < seconds:
        it += 1
        kind, free = neighbourhood(inst, plan, rng)
        new, optimal = solve_sub(inst, plan, free, min(step_seconds, seconds - (time.time() - t0)), workers)
        if new is None:
            continue
        bad, cost = R.evaluate(inst, new)
        assert not bad, bad[:3]
        if cost["total"] < best - 0.5:
            best = cost["total"]
            plan = new
            log(f"{time.time() - t0:7.1f}s it {it:4d} {kind:8s} free {len(free):2d} -> {best:,.0f}"
                f"{'' if optimal else ' (time limit)'}")
    return plan, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--seconds", type=float, default=1800)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    inst = R.load(DATA)
    plan = R.read_plan(args.init)
    plan, best = lns(inst, plan, args.seconds, args.workers, args.seed, log=lambda m: print(m, flush=True))
    bad, cost = R.evaluate(inst, plan)
    print("violations", bad[:3], {k: round(v) for k, v in cost.items()})
    if args.out:
        R.write_plan(args.out, plan)


if __name__ == "__main__":
    main()
