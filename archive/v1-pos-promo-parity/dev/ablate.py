#!/usr/bin/env python3
"""Rule ablations for pos-promo-parity.

Each variant changes one pricing rule to a plausible wrong alternative and
counts how many records change: September journal footers (is the rule
observable from the data the agent gets?) and full October receipts (does
the verifier catch an engine that gets this rule wrong?).

Usage: python3 dev/ablate.py [--json results/ablations.json]
"""

import argparse
import importlib.util
import json
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "pos-promo-parity"
DATA = TASK / "environment" / "data"


def fresh_engine():
    spec = importlib.util.spec_from_file_location("engine", TASK / "solution" / "engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def round_with(mode):
    return lambda value: int(value.quantize(Decimal(1), rounding=mode))


# ---------------------------------------------------------------- variants --
def v_pct_per_line(e):
    real = e.deal_savings.__wrapped__

    def deal_savings(kind, params, prices):
        if kind != "PCT":
            return real(kind, params, prices)
        counts = {}
        for p in prices:
            counts[p] = counts.get(p, 0) + 1
        return sum(e.pct_off(p * n, params[0]) for p, n in counts.items())
    e.deal_savings = deal_savings


def v_pct_truncate(e):
    e.pct_off = lambda cents, pct: int((Decimal(cents) * pct / 100).quantize(Decimal(1), rounding=ROUND_DOWN))
    e.deal_savings.cache_clear()


def v_weight_half_even(e):
    e.weighed_amount = lambda w, p: round_with(ROUND_HALF_EVEN)(Decimal(str(w)) * p)


def v_weight_truncate(e):
    e.weighed_amount = lambda w, p: round_with(ROUND_DOWN)(Decimal(str(w)) * p)


def v_tax_half_even(e):
    e.tax_amount = lambda b, r: round_with(ROUND_HALF_EVEN)(Decimal(b) * r / 10000)


def v_tax_truncate(e):
    e.tax_amount = lambda b, r: round_with(ROUND_DOWN)(Decimal(b) * r / 10000)


def v_coupons_reduce_tax_base(e):
    orig = e.price_basket

    def price_basket(basket, catalog, rates, promos):
        r = orig(basket, catalog, rates, promos)
        if not r["coupons"]:
            return r
        coupons = {c["code"]: c for c in promos["coupons"]}
        base = _class_bases(e, basket, catalog, promos, r)
        for code, value in r["coupons"].items():
            base[catalog[coupons[code]["skus"][0]]["tax_class"]] -= value
        tax = {c: e.tax_amount(b, rates[c]) for c, b in sorted(base.items())}
        tax = {c: v for c, v in tax.items() if v}
        total = r["subtotal"] - sum(r["deals"].values()) - sum(r["coupons"].values()) + sum(tax.values())
        return dict(r, tax=tax, total=total)
    e.price_basket = price_basket


def _class_bases(e, basket, catalog, promos, receipt):
    dept_class = {i["dept"]: i["tax_class"] for i in catalog.values()}
    base = {}
    for line in basket["lines"]:
        item = catalog[line["sku"]]
        if item["sold_by"] == "lb":
            amount = e.weighed_amount(line["weight"], item["price"])
        else:
            unit = item["price"]
            if basket.get("member") is not None and item["member_price"] is not None:
                unit = min(unit, item["member_price"])
            amount = unit * line["qty"]
        base[item["tax_class"]] = base.get(item["tax_class"], 0) + amount
    deals = {d["id"]: d for d in promos["deals"]}
    for deal_id, value in receipt["deals"].items():
        d = deals[deal_id]
        if d["funding"] != "STORE" and not getattr(e, "_all_deals_reduce", False):
            continue
        cls = dept_class[d["dept"]] if d["type"] == "SPEND" else catalog[d["skus"][0]]["tax_class"]
        base[cls] -= value
    return base


def v_vendor_deals_reduce_tax_base(e):
    orig = e.price_basket
    e._all_deals_reduce = True

    def price_basket(basket, catalog, rates, promos):
        r = orig(basket, catalog, rates, promos)
        base = _class_bases(e, basket, catalog, promos, r)
        tax = {c: e.tax_amount(b, rates[c]) for c, b in sorted(base.items())}
        tax = {c: v for c, v in tax.items() if v}
        total = r["subtotal"] - sum(r["deals"].values()) - sum(r["coupons"].values()) + sum(tax.values())
        return dict(r, tax=tax, total=total)
    e.price_basket = price_basket


def v_spend_all_or_nothing(e):
    """Spend coupling only by dropping whole deals from the unconstrained optimum."""
    import itertools
    real = e.optimise_department

    def optimise(gross, units, item_deals, spend):
        free = real(gross, units, item_deals, None)
        if not spend:
            return free
        ranked = sorted(list(item_deals.values()) + [spend], key=lambda d: d["priority"], reverse=True)
        best_key, best = None, None
        ids = sorted(free)
        for r in range(len(ids) + 1):
            for keep in itertools.combinations(ids, r):
                saved = {k: free[k] for k in keep}
                if gross - sum(saved.values()) >= spend["threshold"]:
                    saved[spend["id"]] = spend["save"]
                key = (sum(saved.values()), tuple(saved.get(d["id"], 0) for d in ranked))
                if best_key is None or key > best_key:
                    best_key, best = key, saved
        return best
    e.optimise_department = optimise


def v_greedy_all_orders(e):
    """Try every order of deals; each takes its most valuable units greedily."""
    import itertools

    def optimise(gross, units, item_deals, spend):
        ranked = sorted(list(item_deals.values()) + ([spend] if spend else []),
                        key=lambda d: d["priority"], reverse=True)
        best_key, best = None, None
        for order in itertools.permutations(item_deals.values()):
            pool = []
            for price, count, eligible in units:
                pool.extend([(price, frozenset(eligible))] * count)
            saved = {}
            for deal in order:
                mine = sorted([u for u in pool if deal["id"] in u[1]], key=lambda u: -u[0])
                kind, params = e.deal_key(deal)
                best_n, best_v = 0, 0
                for n in range(len(mine) + 1):
                    v = e.deal_savings(kind, params, tuple(sorted(p for p, _ in mine[:n])))
                    if v is not None and v > best_v:
                        best_n, best_v = n, v
                for u in mine[:best_n]:
                    pool.remove(u)
                if best_v:
                    saved[deal["id"]] = best_v
            if spend and gross - sum(saved.values()) >= spend["threshold"]:
                saved[spend["id"]] = spend["save"]
            key = (sum(saved.values()), tuple(saved.get(d["id"], 0) for d in ranked))
            if best_key is None or key > best_key:
                best_key, best = key, saved
        return best
    e.optimise_department = optimise


def v_spend_sequential(e):
    """Optimise item deals ignoring SPEND, then test the threshold."""
    real = e.optimise_department

    def optimise(gross, units, item_deals, spend):
        saved = dict(real(gross, units, item_deals, None))
        if spend and gross - sum(saved.values()) >= spend["threshold"]:
            saved[spend["id"]] = spend["save"]
        return saved
    e.optimise_department = optimise


def v_spend_on_gross(e):
    real = e.optimise_department

    def optimise(gross, units, item_deals, spend):
        saved = dict(real(gross, units, item_deals, None))
        if spend and gross >= spend["threshold"]:
            saved[spend["id"]] = spend["save"]
        return saved
    e.optimise_department = optimise


def v_tie_by_deal_id(e):
    real = e.optimise_department

    def optimise(gross, units, item_deals, spend):
        renumbered = {k: dict(v, priority=-int(k[1:])) for k, v in item_deals.items()}
        sp = dict(spend, priority=-int(spend["id"][1:])) if spend else None
        return real(gross, units, renumbered, sp)
    e.optimise_department = optimise


def v_tie_low_priority_first(e):
    real = e.optimise_department

    def optimise(gross, units, item_deals, spend):
        flipped = {k: dict(v, priority=-v["priority"]) for k, v in item_deals.items()}
        sp = dict(spend, priority=-spend["priority"]) if spend else None
        return real(gross, units, flipped, sp)
    e.optimise_department = optimise


def v_bogo_cheap_together(e):
    real = e.deal_savings.__wrapped__

    def deal_savings(kind, params, prices):
        if kind != "BOGO":
            return real(kind, params, prices)
        buy, get, pct, limit = params
        size = buy + get
        groups, rest = divmod(len(prices), size)
        if rest or (limit is not None and groups > limit):
            return None
        ordered = sorted(prices)
        return sum(e.pct_off(p, pct) for g in range(groups) for p in ordered[g * size:g * size + get])
    e.deal_savings = deal_savings


def v_greedy_priority(e):
    """Apply item deals one at a time in priority order, each as fully as it can."""
    def optimise(gross, units, item_deals, spend):
        pool = []
        for price, count, eligible in units:
            pool.extend([(price, set(eligible))] * count)
        saved = {}
        for deal in sorted(item_deals.values(), key=lambda d: d["priority"], reverse=True):
            mine = sorted([u for u in pool if deal["id"] in u[1]], key=lambda u: -u[0])
            kind, params = e.deal_key(deal)
            best_n, best_v = 0, 0
            for n in range(len(mine) + 1):
                v = e.deal_savings(kind, params, tuple(sorted(p for p, _ in mine[:n])))
                if v is not None and v > best_v:
                    best_n, best_v = n, v
            for u in mine[:best_n]:
                pool.remove(u)
            if best_v:
                saved[deal["id"]] = best_v
        if spend and gross - sum(saved.values()) >= spend["threshold"]:
            saved[spend["id"]] = spend["save"]
        return saved
    e.optimise_department = optimise


def v_coupon_needs_undiscounted(e):
    """Coupons only count units that did not take part in an item deal."""
    orig = e.price_basket

    def price_basket(basket, catalog, rates, promos):
        r = orig(basket, catalog, rates, promos)
        if not r["coupons"]:
            return r
        deal_skus = {s for d in promos["deals"] if d["id"] in r["deals"] and d["type"] != "SPEND"
                     for s in d["skus"]}
        coupons = {c["code"]: c for c in promos["coupons"]}
        new = {}
        for code, value in r["coupons"].items():
            c = coupons[code]
            if any(s in deal_skus for s in c["skus"]):
                continue
            new[code] = value
        if new == r["coupons"]:
            return r
        total = r["total"] + sum(r["coupons"].values()) - sum(new.values())
        return dict(r, coupons=new, total=total)
    e.price_basket = price_basket


VARIANTS = {
    "pct_rounded_per_line": v_pct_per_line,
    "pct_truncated": v_pct_truncate,
    "weight_half_even": v_weight_half_even,
    "weight_truncated": v_weight_truncate,
    "tax_half_even": v_tax_half_even,
    "tax_truncated": v_tax_truncate,
    "coupons_reduce_tax_base": v_coupons_reduce_tax_base,
    "vendor_deals_reduce_tax_base": v_vendor_deals_reduce_tax_base,
    "spend_after_item_deals": v_spend_sequential,
    "spend_on_gross": v_spend_on_gross,
    "tie_break_by_deal_id": v_tie_by_deal_id,
    "tie_break_low_priority": v_tie_low_priority_first,
    "greedy_by_priority": v_greedy_priority,
    "greedy_all_deal_orders": v_greedy_all_orders,
    "spend_drop_whole_deals_only": v_spend_all_or_nothing,
    "coupon_needs_undiscounted_units": v_coupon_needs_undiscounted,
}


def footer(r):
    return {"basket_id": r["basket_id"], "subtotal": r["subtotal"], "deal_savings": sum(r["deals"].values()),
            "coupon_savings": sum(r["coupons"].values()), "tax": r["tax"], "total": r["total"]}


def run(e, period):
    catalog = e.load_catalog(DATA / "legacy" / "catalog.csv")
    stores = json.loads((DATA / "legacy" / "stores.json").read_text())["stores"]
    promos = json.loads((DATA / period / "promotions.json").read_text())
    out = []
    for raw in open(DATA / period / "baskets.jsonl"):
        b = json.loads(raw)
        out.append(e.price_basket(b, catalog, stores[str(b["store"])]["rates"], promos))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    base = {p: run(fresh_engine(), p) for p in ("history", "pending")}
    recorded = [json.loads(x) for x in open(DATA / "history" / "journal.jsonl")]
    assert recorded == [footer(r) for r in base["history"]], "journal out of date; rerun dev/generate.py"
    report = {}
    print(f"{'variant':34} {'history diffs':>14} {'pending diffs':>14}")
    for name, patch in VARIANTS.items():
        row = {}
        for period in ("history", "pending"):
            e = fresh_engine()
            patch(e)
            got = run(e, period)
            if period == "history":   # the agent only sees journal footers for September
                row[period] = sum(footer(a) != footer(b) for a, b in zip(got, base[period]))
            else:
                row[period] = sum(a != b for a, b in zip(got, base[period]))
        report[name] = row
        print(f"{name:34} {row['history']:>14} {row['pending']:>14}")
    print(f"baskets: history={len(base['history'])} pending={len(base['pending'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
