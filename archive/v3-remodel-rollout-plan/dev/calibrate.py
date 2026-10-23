#!/usr/bin/env python3
"""Run one planning method on an instance and report its cost and the gap to the
column-generation lower bound. Used to place the cost target between what generic
methods reach and what the reference method reaches.

  greedy  - dev/greedy.py
  sa      - dev/anneal.py from the greedy plan (t_start 3000, the best schedule found)
  cpsat   - dev/cpsat_baseline.py with the greedy plan as a hint, 2 workers
  oracle  - solution/plan_rollout.py (three dives, polish, crew-pair re-optimization)

Usage: python dev/calibrate.py METHOD --data DIR [--seconds 3600] [--seed 1] [--out plan.csv]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks" / "remodel-rollout-plan" / "solution"))

import rollout as R  # noqa: E402
import plan_rollout as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["greedy", "sa", "cpsat", "oracle", "bound"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--seconds", type=float, default=3600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--starts-cap", type=int, help="override the DC start limit (scaled instances)")
    args = ap.parse_args()
    if args.starts_cap:
        R.STARTS_CAP = args.starts_cap
    inst = R.load(args.data)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    seed_plan = P.greedy(inst)
    if args.method == "bound":
        pr = P.Pricing(inst)
        ms = P.Master(pr)
        for c, route in P.plan_routes(inst, pr, seed_plan).items():
            ms.add(c, route)
        lp, lb, _ = P.generate(pr, ms, 3000)
        log(f"BOUND {lb:.0f}")
        return
    if args.method == "greedy":
        plan = seed_plan
    elif args.method == "sa":
        import anneal
        plan, _ = anneal.anneal(inst, seed_plan, args.seconds, args.seed, t_start=3000.0, t_end=10.0,
                                log=lambda m: None)
    elif args.method == "cpsat":
        from ortools.sat.python import cp_model
        import cpsat_baseline as B
        m, x, y, S, C, starts = B.build(inst, seed_plan)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.seconds
        solver.parameters.num_workers = 2
        st = solver.Solve(m)
        plan = B.extract(solver, x, y, S, C, starts)
    else:
        best = None
        for k in range(3):
            lb, plan = P.dive_once(inst, seed_plan, k)
            if plan is None:
                continue
            plan = P.polish(inst, plan, moves=40_000_000, seed=k)
            total = R.cost(inst, plan)["total"]
            log(f"dive {k} polished {total:,.0f}")
            if best is None or total < best[0]:
                best = (total, plan)
        plan = P.improve(inst, best[1], rounds=4, log=log)
    bad, cost = R.evaluate(inst, plan)
    log(f"FINAL {args.method} {cost['total']:.0f} violations {len(bad)}")
    if args.out:
        R.write_plan(args.out, plan)


if __name__ == "__main__":
    main()
