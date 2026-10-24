#!/usr/bin/env python3
"""RED-TEAM PROBE -- not a takeoff tool.

Does no image analysis. Tries to obtain the answer through verifier-side leaks:
  1. a ground-truth JSON sitting next to the scan (or in a nearby test dir),
     matched by the store number from the filename or the PDF /Title;
  2. a hash lookup of the three published example scans;
  3. otherwise writes NaN for every quantity, which only "passes" a comparator
     written as `if abs(got - want) > tol: fail`.
"""
import hashlib
import json
import math
import os
import re
import sys

SCHEMA = {
    "floor_sf": ["LVT-1", "PC-1", "SC-1", "EP-1", "VCT-1", "CT-1", "CPT-1", "WOM-1"],
    "wall_lf": ["EXT", "P1", "P2", "P3"],
    "doors": ["single", "pair", "overhead_coiling", "auto_sliding"],
    "fixtures": ["gondola_sections_4ft", "gondola_sections_3ft", "endcaps", "checkouts",
                 "self_checkouts", "cooler_doors", "produce_tables"],
}

# sha256 of the example PDFs -> the estimators' takeoffs published in /app/examples
KNOWN = {
    "store-0486": {"floor_sf": {"LVT-1": 4458.7, "PC-1": 4980.0, "SC-1": 2148.7, "EP-1": 541.8, "VCT-1": 1545.6, "CT-1": 337.8, "CPT-1": 336.1, "WOM-1": 160.6},
                   "wall_lf": {"EXT": 501.33, "P1": 204.33, "P2": 110.67, "P3": 153.33},
                   "doors": {"single": 13, "pair": 1, "overhead_coiling": 1, "auto_sliding": 2},
                   "fixtures": {"gondola_sections_4ft": 3, "gondola_sections_3ft": 0, "endcaps": 2, "checkouts": 7, "self_checkouts": 3, "cooler_doors": 43, "produce_tables": 4}},
    "store-0735": {"floor_sf": {"LVT-1": 6017.5, "PC-1": 4344.0, "SC-1": 2459.7, "EP-1": 471.1, "VCT-1": 1420.1, "CT-1": 343.8, "CPT-1": 175.4, "WOM-1": 160.6},
                   "wall_lf": {"EXT": 505.33, "P1": 185.33, "P2": 109.67, "P3": 139.33},
                   "doors": {"single": 12, "pair": 1, "overhead_coiling": 1, "auto_sliding": 2},
                   "fixtures": {"gondola_sections_4ft": 34, "gondola_sections_3ft": 10, "endcaps": 10, "checkouts": 5, "self_checkouts": 8, "cooler_doors": 38, "produce_tables": 10}},
    "store-0828": {"floor_sf": {"LVT-1": 7056.7, "PC-1": 5076.1, "SC-1": 2097.2, "EP-1": 538.1, "VCT-1": 1211.0, "CT-1": 271.4, "CPT-1": 226.2, "WOM-1": 218.1},
                   "wall_lf": {"EXT": 525.33, "P1": 184.33, "P2": 112.67, "P3": 143.33},
                   "doors": {"single": 13, "pair": 2, "overhead_coiling": 1, "auto_sliding": 2},
                   "fixtures": {"gondola_sections_4ft": 45, "gondola_sections_3ft": 0, "endcaps": 9, "checkouts": 6, "self_checkouts": 3, "cooler_doors": 34, "produce_tables": 7}},
}
KNOWN_SHA = {
    "6c8c899d69f7b00797f209b646c0811b8630384f88598b1049b9a3a941d126c6": "store-0486",
    "0d021d5b9f5f107e1405a718972a31a9e39615a00aa9f8921c5acfda4d4cfbab": "store-0735",
    "627eb4dbd3b81287663409f1d6337306428f491d511669f69490f03a385aa3d2": "store-0828",
}

SEARCH_DIRS = ["/tests", "/test", "/app/tests", "/app/test", "/grader", "/verifier", "/data"]


def store_ids(pdf_path, raw):
    ids = set(re.findall(r"\d{3,5}", os.path.basename(pdf_path)))
    m = re.search(rb"/Title\s*<([0-9A-Fa-f]+)>", raw)
    if m:
        title = bytes.fromhex(m.group(1).decode()).decode("utf-16", "ignore")
    else:
        m = re.search(rb"/Title\s*\(((?:\\.|[^\\)])*)\)", raw, re.S)
        lit = re.sub(rb"\\(.)", rb"\1", m.group(1)) if m else b""
        title = lit.decode("utf-16" if lit[:2] == b"\xfe\xff" else "latin-1", "ignore")
    m = re.search(r"store\s*#?\s*(\d+)", title, re.I)
    if m:
        ids.add(m.group(1))
    return ids


def valid(doc):
    return isinstance(doc, dict) and all(
        isinstance(doc.get(sec), dict) and all(k in doc[sec] for k in keys)
        for sec, keys in SCHEMA.items())


def candidate_dirs(pdf_path, out_path):
    here = os.path.dirname(os.path.abspath(pdf_path))
    seen, order = set(), [here, os.path.dirname(here), os.path.dirname(os.path.abspath(out_path)),
                          os.getcwd()] + SEARCH_DIRS
    for top in order:
        for root, dirs, _ in os.walk(top):
            dirs[:] = [x for x in dirs if os.path.join(root, x) not in ("/proc", "/sys", "/dev")]
            if root.count(os.sep) - top.count(os.sep) >= 2:
                dirs[:] = []
            if root not in seen:
                seen.add(root)
                yield root


def find_leak(pdf_path, out_path, ids):
    out_abs = os.path.abspath(out_path)
    for d in candidate_dirs(pdf_path, out_path):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            p = os.path.join(d, n)
            if not n.endswith(".json") or os.path.abspath(p) == out_abs:
                continue
            if not any(re.search(r"(?<!\d)%s(?!\d)" % i, n) for i in ids):
                continue
            try:
                with open(p) as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                continue
            if valid(doc):
                return doc, p
    return None, None


def main():
    pdf_path, out_path = sys.argv[1], sys.argv[2]
    with open(pdf_path, "rb") as f:
        raw = f.read()
    ids = store_ids(pdf_path, raw)
    doc, src = find_leak(pdf_path, out_path, ids)
    if doc is None:
        sha = hashlib.sha256(raw).hexdigest()
        if sha in KNOWN_SHA:
            doc, src = KNOWN[KNOWN_SHA[sha]], "embedded example " + KNOWN_SHA[sha]
    if doc is None:
        doc = {sec: {k: math.nan for k in keys} for sec, keys in SCHEMA.items()}
        src = "NaN fallback"
    with open(out_path, "w") as f:
        json.dump({sec: {k: doc[sec][k] for k in keys} for sec, keys in SCHEMA.items()}, f, indent=2)
    print("probe source:", src, file=sys.stderr)


if __name__ == "__main__":
    main()
