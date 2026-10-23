#!/usr/bin/env python3
"""Simulated annealing over feasible plans: the other approach an agent is likely
to write. Moves: shift a store's start week, move a store to another crew and
week, swap the crews of two stores (each keeping or re-picking its week).
Infeasible neighbours are rejected; costs are recomputed for the crews touched.

Usage: python dev/anneal.py --init plan.csv [--seconds 600] [--seed 1] [--out plan.csv]
"""

import argparse
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import rollout as R

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tasks" / "remodel-rollout-plan" / "environment" / "data"


class State:
    def __init__(self, inst, plan):
        self.inst = inst
        self.plan = dict(plan)
        self.starts = {s: sorted(t for (x, t) in inst.lost if x == s) for s in inst.stores}
        self.dload = defaultdict(int)
        self.nstart = defaultdict(int)
        for s, (c, t) in self.plan.items():
            self._mark(s, t, +1)
        self.routes = R.crew_routes(self.plan)
        self.crew_cost = {c: self._crew(c, self.routes.get(c, [])) for c in inst.crews}
        self.total = sum(self.crew_cost.values()) + sum(inst.lost[s, t] for s, (c, t) in self.plan.items())

    def _mark(self, s, t, k):
        st = self.inst.stores[s]
        self.nstart[t] += k
        for w in range(t, t + st["weeks"]):
            self.dload[st["district"], w] += k

    def _crew(self, c, jobs):
        """Contract + moves for one crew's sorted jobs, or None if the sequence is infeasible."""
        if not jobs:
            return 0.0
        inst = self.inst
        for (s1, a), (s2, b) in zip(jobs, jobs[1:]):
            if s2 < s1 + inst.stores[a]["weeks"] + inst.gap(a, b):
                return None
        home = inst.crews[c]["home"]
        stops = [home] + [s for _, s in jobs] + [home]
        span = jobs[-1][0] + inst.stores[jobs[-1][1]]["weeks"] - jobs[0][0]
        return inst.crews[c]["rate"] * span + R.MILE_RATE * sum(inst.miles[x, y] for x, y in zip(stops, stops[1:]))

    def caps_ok(self, changes):
        """changes: list of (store, old start, new start)."""
        for s, old, new in changes:
            self._mark(s, old, -1)
        ok = True
        for s, old, new in changes:
            self._mark(s, new, +1)
        for s, old, new in changes:
            st = self.inst.stores[s]
            if self.nstart[new] > R.STARTS_CAP or any(
                    self.dload[st["district"], w] > R.DISTRICT_CAP for w in range(new, new + st["weeks"])):
                ok = False
        if not ok:
            for s, old, new in changes:
                self._mark(s, new, -1)
                self._mark(s, old, +1)
        return ok

    def try_move(self, assign, temp, rng):
        """assign: {store: (crew, start)} for the stores that change."""
        inst = self.inst
        crews = {self.plan[s][0] for s in assign} | {c for c, _ in assign.values()}
        new_routes = {}
        for c in crews:
            jobs = [(t, s) for t, s in self.routes.get(c, []) if s not in assign]
            jobs += [(t, s) for s, (cc, t) in assign.items() if cc == c]
            new_routes[c] = sorted(jobs)
        new_cost = {}
        for c in crews:
            v = self._crew(c, new_routes[c])
            if v is None:
                return False
            new_cost[c] = v
        delta = sum(new_cost[c] - self.crew_cost[c] for c in crews)
        delta += sum(inst.lost[s, t] - inst.lost[s, self.plan[s][1]] for s, (c, t) in assign.items())
        if delta > 0 and rng.random() >= math.exp(-delta / temp):
            return False
        if not self.caps_ok([(s, self.plan[s][1], t) for s, (c, t) in assign.items()]):
            return False
        for s, ct in assign.items():
            self.plan[s] = ct
        for c in crews:
            self.routes[c] = new_routes[c]
            self.crew_cost[c] = new_cost[c]
        self.total += delta
        return True


def anneal(inst, plan, seconds, seed, t_start=20000.0, t_end=50.0, log=print):
    rng = random.Random(seed)
    st = State(inst, plan)
    S = sorted(inst.stores)
    C = sorted(inst.crews)
    best, best_plan = st.total, dict(st.plan)
    t0 = time.time()
    n = 0
    while True:
        frac = (time.time() - t0) / seconds
        if frac >= 1:
            break
        temp = t_start * (t_end / t_start) ** frac
        n += 1
        kind = rng.random()
        s = rng.choice(S)
        c, t = st.plan[s]
        if kind < 0.4:
            assign = {s: (c, rng.choice(st.starts[s]))}
        elif kind < 0.75:
            assign = {s: (rng.choice(C), rng.choice(st.starts[s]))}
        else:
            u = rng.choice(S)
            cu, tu = st.plan[u]
            if cu == c:
                continue
            assign = {s: (cu, rng.choice([t, tu] + st.starts[s][:0]) if tu in st.starts[s] else t),
                      u: (c, tu if t not in st.starts[u] else rng.choice([t, tu]))}
            assign = {k: (cc, tt) if tt in st.starts[k] else (cc, st.plan[k][1]) for k, (cc, tt) in assign.items()}
        st.try_move(assign, temp, rng)
        if st.total < best - 0.5:
            best, best_plan = st.total, dict(st.plan)
        if n % 200000 == 0:
            log(f"{time.time() - t0:7.1f}s moves {n:9d} temp {temp:8.0f} current {st.total:14,.0f} best {best:14,.0f}")
    return best_plan, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    inst = R.load(DATA)
    plan, best = anneal(inst, R.read_plan(args.init), args.seconds, args.seed, log=lambda m: print(m, flush=True))
    bad, cost = R.evaluate(inst, plan)
    print("violations", bad[:3], {k: round(v) for k, v in cost.items()})
    if args.out:
        R.write_plan(args.out, plan)


if __name__ == "__main__":
    main()
