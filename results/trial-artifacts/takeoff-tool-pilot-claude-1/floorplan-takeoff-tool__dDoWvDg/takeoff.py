#!/usr/bin/env python3
"""Quantity takeoff from a scanned Cedar Valley Market A101 fixture & finish plan.

usage: python3 takeoff.py <scan.pdf> <takeoff.json>
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tk import pipeline  # noqa: E402


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    pdf, out = argv[1], argv[2]
    status = 0
    try:
        result = pipeline.run(pdf)
    except Exception:
        # never leave the caller without a takeoff file
        traceback.print_exc()
        result = pipeline.empty_result()
        status = 1
    with open(out, 'w') as fh:
        json.dump(result, fh, indent=2)
        fh.write('\n')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
