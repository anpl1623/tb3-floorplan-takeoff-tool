#!/usr/bin/env python3
"""Direct CP-SAT model of the remodel program: the formulation a capable agent
would most likely write first. Used to calibrate the cost target.

One-hot start week per store (lost margin, district and DC limits are linear in
it), one circuit per crew over its home and the stores (travel, sequencing and
contract span).

Usage: python dev/cpsat_baseline.py [--seconds 600] [--workers 2] [--out plan.csv] [--hint plan.csv]
"""

import argparse
import time
from pathlib import Path

from ortools.sat.python import cp_model

import rollout as R

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tasks" / "remodel-rollout-plan" / "environment" / "data"


def build(inst, hint=None, allowed=None):
    """allowed: optional {store: (crews, start weeks)} restricting each store's choices (LNS)."""
    m = cp_model.CpModel()
    S = sorted(inst.stores)
    C = sorted(inst.crews)
    starts = {s: [t for (x, t) in inst.lost if x == s] for s in S}
    crews_of = {s: C for s in S}
    if allowed:
        for s, (cs, ts) in allowed.items():
            crews_of[s] = sorted(cs)
            starts[s] = sorted(ts)
    y = {(s, t): m.NewBoolVar(f"y_{s}_{t}") for s in S for t in starts[s]}
    start = {}
    for s in S:
        m.AddExactlyOne(y[s, t] for t in starts[s])
        start[s] = m.NewIntVar(min(starts[s]), max(starts[s]), f"start_{s}")
        m.Add(start[s] == sum(t * y[s, t] for t in starts[s]))
    for w in range(1, R.LAST_WORK_WEEK + 1):
        m.Add(sum(y[s, w] for s in S if (s, w) in y) <= R.STARTS_CAP)
        for d in {st["district"] for st in inst.stores.values()}:
            active = [y[s, t] for s in S if inst.stores[s]["district"] == d for t in starts[s]
                      if t <= w < t + inst.stores[s]["weeks"]]
            if len(active) > R.DISTRICT_CAP:
                m.Add(sum(active) <= R.DISTRICT_CAP)

    x = {(s, c): m.NewBoolVar(f"x_{s}_{c}") for s in S for c in crews_of[s]}
    for s in S:
        m.AddExactlyOne(x[s, c] for c in crews_of[s])
    obj = [round(inst.lost[s, t]) * y[s, t] for s in S for t in starts[s]]
    H = R.LAST_WORK_WEEK + 1
    for c in C:
        home = inst.crews[c]["home"]
        rate = round(inst.crews[c]["rate"])
        mine = [s for s in S if (s, c) in x]
        if not mine:
            continue
        used = m.NewBoolVar(f"used_{c}")
        m.AddMaxEquality(used, [x[s, c] for s in mine])
        first = m.NewIntVar(0, H, f"first_{c}")
        last = m.NewIntVar(0, H, f"last_{c}")
        m.Add(first == 0).OnlyEnforceIf(used.Not())
        m.Add(last == 0).OnlyEnforceIf(used.Not())
        arcs = [(0, 0, used.Not())]
        idx = {s: i + 1 for i, s in enumerate(mine)}
        for s in mine:
            arcs.append((idx[s], idx[s], x[s, c].Not()))
            a = m.NewBoolVar(f"in_{c}_{s}")
            arcs.append((0, idx[s], a))
            m.Add(first == start[s]).OnlyEnforceIf(a)
            obj.append(round(R.MILE_RATE * inst.miles[home, s]) * a)
            b = m.NewBoolVar(f"out_{c}_{s}")
            arcs.append((idx[s], 0, b))
            m.Add(last == start[s] + inst.stores[s]["weeks"]).OnlyEnforceIf(b)
            obj.append(round(R.MILE_RATE * inst.miles[s, home]) * b)
            for u in mine:
                if u == s:
                    continue
                # u can follow s only if it can start after s finishes plus the move
                if max(starts[u]) < min(starts[s]) + inst.stores[s]["weeks"] + inst.gap(s, u):
                    continue
                l = m.NewBoolVar(f"a_{c}_{s}_{u}")
                arcs.append((idx[s], idx[u], l))
                m.Add(start[u] >= start[s] + inst.stores[s]["weeks"] + inst.gap(s, u)).OnlyEnforceIf(l)
                obj.append(round(R.MILE_RATE * inst.miles[s, u]) * l)
        m.AddCircuit(arcs)
        span = m.NewIntVar(0, H, f"span_{c}")
        m.Add(span == last - first)
        obj.append(rate * span)
    m.Minimize(sum(obj))
    if hint:
        for s, (c, t) in hint.items():
            for tt in starts[s]:
                m.AddHint(y[s, tt], tt == t)
            for cc in crews_of[s]:
                m.AddHint(x[s, cc], cc == c)
    return m, x, y, S, C, starts


def extract(solver, x, y, S, C, starts):
    plan = {}
    for s in S:
        c = next(c for c in C if (s, c) in x and solver.Value(x[s, c]))
        t = next(t for t in starts[s] if solver.Value(y[s, t]))
        plan[s] = (c, t)
    return plan


class Log(cp_model.CpSolverSolutionCallback):
    def __init__(self, t0):
        super().__init__()
        self.t0 = t0

    def on_solution_callback(self):
        print(f"  {time.time() - self.t0:7.1f}s  obj {self.ObjectiveValue():,.0f}  bound {self.BestObjectiveBound():,.0f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out")
    ap.add_argument("--hint")
    args = ap.parse_args()
    inst = R.load(DATA)
    hint = R.read_plan(args.hint) if args.hint else None
    t0 = time.time()
    m, x, y, S, C, starts = build(inst, hint)
    print(f"model built in {time.time() - t0:.1f}s")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_workers = args.workers
    st = solver.Solve(m, Log(time.time()))
    print(solver.StatusName(st))
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        plan = extract(solver, x, y, S, C, starts)
        bad, cost = R.evaluate(inst, plan)
        print("violations", bad[:5], "cost", {k: round(v) for k, v in cost.items()})
        if args.out:
            R.write_plan(args.out, plan)


if __name__ == "__main__":
    main()
