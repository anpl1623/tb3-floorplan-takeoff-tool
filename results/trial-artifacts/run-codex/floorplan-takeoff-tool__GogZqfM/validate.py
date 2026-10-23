#!/usr/bin/env python3
"""Check takeoff measurements against an examples directory."""
import argparse
import json
from pathlib import Path
import resource
import time
from takeoff import takeoff


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',type=Path)
    args=parser.parse_args()
    cases=sorted(args.examples.glob('*.takeoff.json'))
    if not cases:parser.error('No *.takeoff.json reference files found')
    failures=[]
    max_area=0.;max_length=0.;max_seconds=0.
    for reference in cases:
        scan=reference.with_name(reference.name.replace('.takeoff.json','.pdf'))
        expected=json.loads(reference.read_text())
        started=time.monotonic()
        actual=takeoff(scan)
        elapsed=time.monotonic()-started
        max_seconds=max(max_seconds,elapsed)
        errors=[]
        for section,values in expected.items():
            if set(actual[section])!=set(values):errors.append(section+': keys differ')
            for key,value in values.items():
                measured=actual[section][key]
                if section=='floor_sf':
                    tolerance=max(8.,.02*value)
                    max_area=max(max_area,abs(measured-value)/value*100)
                elif section=='wall_lf':
                    tolerance=max(2.,.015*value)
                    max_length=max(max_length,abs(measured-value)/value*100)
                else:tolerance=0
                if abs(measured-value)>tolerance:
                    errors.append(f'{section}.{key}: {measured} vs {value}')
        if errors:failures.extend([str(scan)+': '+e for e in errors])
        print(f'{scan.name}: {"FAIL" if errors else "PASS"} ({elapsed:.2f} s)',flush=True)
    print(f'Maximum area error: {max_area:.2f}%; maximum length error: {max_length:.2f}%')
    print(f'Maximum time: {max_seconds:.2f} s; process peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MiB')
    if failures:raise SystemExit('\n'.join(failures))
    print('All counts exact; all measurements within tolerance.')


if __name__=='__main__':main()
