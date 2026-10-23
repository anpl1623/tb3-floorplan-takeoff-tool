#!/usr/bin/env python3
"""Probe the verifier with crafted receipt files.

Builds the verifier image from tests/Dockerfile, then for each case writes a
receipts file, runs tests/test.sh inside a fresh verifier container with the
file mounted where harbor places the artifact, and reads the reward. Cases
cover exploit attempts (must score 0), near-misses (must score 0) and
legitimate formatting variants of a correct answer (must score 1).

Usage: python3 dev/verifier_probe.py
"""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "pos-promo-parity"
DATA = TASK / "environment" / "data"
IMAGE = "pos-promo-parity-verifier:probe"

spec = importlib.util.spec_from_file_location("engine", TASK / "solution" / "engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def correct():
    catalog = engine.load_catalog(DATA / "legacy" / "catalog.csv")
    stores = json.loads((DATA / "legacy" / "stores.json").read_text())["stores"]
    promos = json.loads((DATA / "pending" / "promotions.json").read_text())
    out = []
    for raw in open(DATA / "pending" / "baskets.jsonl"):
        b = json.loads(raw)
        out.append(engine.price_basket(b, catalog, stores[str(b["store"])]["rates"], promos))
    return out


def lines(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


def cases(good):
    first = good[0]
    with_deal = next(r for r in good if r["deals"])
    yield "empty file", "", 0
    yield "nan amounts", lines(dict(r, total=math.nan, subtotal=math.nan) for r in good), 0
    yield "receipt per basket, all zero", lines(
        {"basket_id": r["basket_id"], "subtotal": 0, "deals": {}, "coupons": {}, "tax": {}, "total": 0}
        for r in good), 0
    yield "string amounts", lines(dict(r, total=str(r["total"])) for r in good), 0
    yield "one cent off one total", lines(
        [dict(r, total=r["total"] + 1) if r is first else r for r in good]), 0
    yield "one deal attributed to wrong id", lines(
        [dict(r, deals={"D9999": sum(r["deals"].values())}) if r is with_deal else r for r in good]), 0
    yield "one receipt missing", lines(good[1:]), 0
    yield "duplicate receipt", lines(good + [first]), 0
    yield "extra unknown basket", lines(good + [dict(first, basket_id="999-20261001-9999")]), 0
    yield "symlink to verifier inputs", "SYMLINK:/tests/inputs/pending/baskets.jsonl", 0
    yield "correct answer", lines(good), 1
    yield "correct, floats and explicit zeros", lines(
        dict(r, total=float(r["total"]), tax={"F": 0, "T": 0, "P": 0, **r["tax"]}, deals=dict(r["deals"], D0000=0))
        for r in good), 1
    yield "correct, shuffled order and pretty keys", "".join(
        json.dumps(r, indent=None, sort_keys=True) + "\n" for r in reversed(good)), 1


def run_case(content):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        out.mkdir()
        target = out / "receipts.jsonl"
        mounts = ["-v", f"{out}:/app/output:ro"]
        if content.startswith("SYMLINK:"):
            target.symlink_to(content.split(":", 1)[1])
        else:
            target.write_text(content)
        cmd = ["docker", "run", "--rm", *mounts, IMAGE, "bash", "-c",
               "mkdir -p /logs/verifier && bash /tests/test.sh >/dev/null 2>&1; cat /logs/verifier/reward.txt"]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def main():
    subprocess.run(["docker", "build", "-q", "-t", IMAGE, str(TASK / "tests")], check=True,
                   capture_output=True)
    good = correct()
    failures = 0
    for name, content, want in cases(good):
        got = run_case(content)
        ok = got == str(want)
        failures += not ok
        print(f"{'ok ' if ok else 'BAD'} reward={got} expected={want}  {name}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
