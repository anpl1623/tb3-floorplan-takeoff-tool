#!/usr/bin/env python3
"""Cross-check the reference engine's deal optimiser against a brute force.

The brute force works on individual units instead of per-SKU counts, tries
every assignment of every unit to a deal (or to no deal), and for BOGO and
MULTI deals tries every partition of the deal's units into groups, keeping
the grouping that saves the customer the most. It checks two things:

  1. the canonical grouping in the manual (BOGO: most expensive units
     grouped together, cheapest discounted) is the customer-optimal one;
  2. the engine's chosen deal savings equal the brute-force optimum,
     including the priority tie-break.

Runs over (a) every real department-level problem in the history and
pending baskets small enough to brute force, and (b) random stress cases.

Usage: python3 dev/crosscheck.py [n_random]
"""

import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "pos-promo-parity"
DATA = TASK / "environment" / "data"
spec = importlib.util.spec_from_file_location("engine", TASK / "solution" / "engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

MAX_UNITS = 8


def partitions_into_groups(items, size):
    """All ways to split `items` (a list) into unordered groups of `size`."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for others in itertools.combinations(range(len(rest)), size - 1):
        group = [first] + [rest[i] for i in others]
        remaining = [rest[i] for i in range(len(rest)) if i not in others]
        for tail in partitions_into_groups(remaining, size):
            yield [group] + tail


_cache = {}


def brute_deal_savings(deal, prices):
    key = (json.dumps(deal, sort_keys=True), tuple(sorted(prices)))
    if key not in _cache:
        _cache[key] = _brute_deal_savings(deal, prices)
    return _cache[key]


def _brute_deal_savings(deal, prices):
    if deal["type"] == "PCT":
        return sum(engine.pct_off(p, deal["pct"]) for p in prices)
    size = deal["qty"] if deal["type"] == "MULTI" else deal["buy"] + deal["get"]
    groups, rest = divmod(len(prices), size)
    limit = deal.get("limit")
    if rest or (limit is not None and groups > limit):
        return None
    best = None
    for split in partitions_into_groups(list(prices), size):
        if deal["type"] == "MULTI":
            value = sum(prices) - groups * deal["price"]
        else:
            value = sum(engine.pct_off(p, deal["pct"]) for g in split
                        for p in sorted(g, reverse=True)[deal["buy"]:])
        best = value if best is None else max(best, value)
    canonical = engine.deal_savings(*engine.deal_key(deal), tuple(sorted(prices)))
    assert canonical == best, (deal["id"], prices, canonical, best)
    return best


def brute_department(gross, units, item_deals, spend):
    flat = []                     # one entry per physical unit
    for price, count, eligible in units:
        flat.extend([(price, eligible)] * count)
    ranked = sorted(list(item_deals.values()) + ([spend] if spend else []),
                    key=lambda d: d["priority"], reverse=True)
    best_key, best = None, None
    options = [[None] + list(elig) for _, elig in flat]
    for choice in itertools.product(*options):
        assigned = {d: [] for d in item_deals}
        for (price, _), deal_id in zip(flat, choice):
            if deal_id is not None:
                assigned[deal_id].append(price)
        saved, ok = {}, True
        for deal_id, prices in assigned.items():
            value = brute_deal_savings(item_deals[deal_id], prices)
            if value is None:
                ok = False
                break
            if value:
                saved[deal_id] = value
        if not ok:
            continue
        if spend and gross - sum(saved.values()) >= spend["threshold"]:
            saved[spend["id"]] = spend["save"]
        key = (sum(saved.values()), tuple(saved.get(d["id"], 0) for d in ranked))
        if best_key is None or key > best_key:
            best_key, best = key, saved
    return best


class Recorder:
    """Wraps engine.optimise_department to capture department problems."""

    def __init__(self):
        self.problems = []
        self.real = engine.optimise_department

    def __call__(self, gross, units, item_deals, spend):
        result = self.real(gross, units, item_deals, spend)
        self.problems.append((gross, units, item_deals, spend, result))
        return result


def real_problems():
    catalog = engine.load_catalog(DATA / "legacy" / "catalog.csv")
    stores = json.loads((DATA / "legacy" / "stores.json").read_text())["stores"]
    recorder = Recorder()
    engine.optimise_department = recorder
    try:
        for period in ("history", "pending"):
            promos = json.loads((DATA / period / "promotions.json").read_text())
            for raw in open(DATA / period / "baskets.jsonl"):
                basket = json.loads(raw)
                engine.price_basket(basket, catalog, stores[str(basket["store"])]["rates"], promos)
    finally:
        engine.optimise_department = recorder.real
    return recorder.problems


def random_problem(rng, n):
    deals = {}
    kinds = ["MULTI", "BOGO", "PCT"]
    for i in range(rng.randint(1, 3)):
        kind = rng.choice(kinds)
        d = {"id": f"X{i}", "type": kind, "priority": rng.sample(range(10, 99), 1)[0] + i * 100}
        if kind == "MULTI":
            d.update(qty=rng.choice([2, 3]), price=rng.choice([300, 400, 500, 600]), limit=rng.choice([None, 1, 2]))
        elif kind == "BOGO":
            d.update(buy=rng.choice([1, 2]), get=1, pct=rng.choice([50, 100]), limit=rng.choice([None, 1, 2]))
        else:
            d.update(pct=rng.choice([10, 15, 20, 25, 30]))
        deals[d["id"]] = d
    units, left = [], n
    while left:
        count = rng.randint(1, min(3, left))
        price = rng.choice([149, 199, 249, 250, 299, 349, 399])
        eligible = sorted(rng.sample(sorted(deals), rng.randint(1, len(deals))))
        # MULTI deal prices must stay below the price of any group.
        units.append((price, count, eligible))
        left -= count
    for d in deals.values():
        if d["type"] == "MULTI":
            cheapest = min(p for p, _, e in units if d["id"] in e) if any(d["id"] in e for _, _, e in units) else 999
            d["price"] = min(d["price"], cheapest * d["qty"] - 1)
    gross = sum(p * c for p, c, _ in units) + rng.choice([0, 500, 1500])
    spend = None
    if rng.random() < 0.6:
        spend = {"id": "SP", "type": "SPEND", "priority": rng.choice([5, 99, 199, 999]),
                 "threshold": gross - rng.randint(0, 900), "save": rng.choice([200, 300, 500])}
    return gross, units, deals, spend


def main():
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    checked = skipped = 0
    for gross, units, item_deals, spend, result in real_problems():
        n = sum(c for _, c, e in units if e)
        if not item_deals or n > MAX_UNITS:
            skipped += bool(item_deals)
            continue
        expect = brute_department(gross, units, item_deals, spend)
        assert result == expect, (gross, units, item_deals, spend, result, expect)
        checked += 1
    print(f"real department problems checked={checked} skipped_too_large={skipped}")
    rng = random.Random(7)
    for _ in range(n_random):
        gross, units, deals, spend = random_problem(rng, rng.randint(2, MAX_UNITS))
        expect = brute_department(gross, units, deals, spend)
        got = engine.optimise_department(gross, units, deals, spend)
        assert got == expect, (gross, units, deals, spend, got, expect)
    print(f"random stress problems checked={n_random}")


if __name__ == "__main__":
    main()
