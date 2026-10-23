#!/usr/bin/env python3
"""Greedy construction: the quick plan a program manager would sketch by hand.

Stores are taken in order of their latest feasible start; each goes to the
crew and week that adds the least cost right now, subject to every rule.

Usage: python dev/greedy.py [--out plan.csv]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import rollout as R

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tasks" / "remodel-rollout-plan" / "environment" / "data"


def greedy(inst, order_key=None, weight_idle=1.0):
    S = sorted(inst.stores)
    starts = {s: sorted(t for (x, t) in inst.lost if x == s) for s in S}
    order = sorted(S, key=order_key or (lambda s: (max(starts[s]), len(starts[s]))))
    district_load = defaultdict(int)
    n_starts = defaultdict(int)
    routes = {c: [] for c in inst.crews}         # sorted (start, store)
    plan = {}
    for s in order:
        d, wks = inst.stores[s]["district"], inst.stores[s]["weeks"]
        best = None
        for t in starts[s]:
            if n_starts[t] >= R.STARTS_CAP or any(district_load[d, w] >= R.DISTRICT_CAP for w in range(t, t + wks)):
                continue
            for c, jobs in routes.items():
                home = inst.crews[c]["home"]
                # insert (t, s) into the crew's sequence if it fits between neighbours
                pos = sum(1 for j in jobs if j[0] < t)
                prev = jobs[pos - 1] if pos else None
                nxt = jobs[pos] if pos < len(jobs) else None
                if prev and t < prev[0] + inst.stores[prev[1]]["weeks"] + inst.gap(prev[1], s):
                    continue
                if nxt and nxt[0] < t + wks + inst.gap(s, nxt[1]):
                    continue
                a = prev[1] if prev else home
                b = nxt[1] if nxt else home
                dtravel = R.MILE_RATE * (inst.miles[a, s] + inst.miles[s, b] - inst.miles[a, b])
                if jobs:
                    first = min(jobs[0][0], t)
                    last = max(jobs[-1][0] + inst.stores[jobs[-1][1]]["weeks"], t + wks)
                    dspan = (last - first) - (jobs[-1][0] + inst.stores[jobs[-1][1]]["weeks"] - jobs[0][0])
                else:
                    dspan = wks
                cost = inst.lost[s, t] + dtravel + weight_idle * inst.crews[c]["rate"] * dspan
                if best is None or cost < best[0]:
                    best = (cost, c, t, pos)
        if best is None:
            return None
        _, c, t, pos = best
        routes[c].insert(pos, (t, s))
        plan[s] = (c, t)
        n_starts[t] += 1
        for w in range(t, t + wks):
            district_load[d, w] += 1
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    inst = R.load(DATA)
    plan = greedy(inst)
    if plan is None:
        print("greedy failed to place every store")
        return
    bad, cost = R.evaluate(inst, plan)
    print("violations", bad[:5])
    print({k: round(v) for k, v in cost.items()})
    if args.out:
        R.write_plan(args.out, plan)


if __name__ == "__main__":
    main()
