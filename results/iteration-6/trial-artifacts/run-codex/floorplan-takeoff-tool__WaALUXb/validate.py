#!/usr/bin/env python3
"""Run the CLI against an example directory and enforce the takeoff tolerances."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',nargs='?',type=Path,default=Path(__file__).resolve().parent.parent/'examples')
    args=parser.parse_args()
    tool=Path(__file__).resolve().parent/'takeoff.py'
    failures=[]
    cases=[scan for scan in sorted(args.examples.glob('*.pdf'))
           if scan.with_suffix('.takeoff.json').exists()]
    if not cases:
        parser.error('No PDF/takeoff reference pairs were found in the example directory')
    with tempfile.TemporaryDirectory(prefix='takeoff-validation-') as temp:
        for scan in cases:
            reference=scan.with_suffix('.takeoff.json')
            if not reference.exists():continue
            output=Path(temp)/(scan.stem+'.json')
            start=time.monotonic()
            subprocess.run([sys.executable,str(tool),str(scan),str(output)],check=True,timeout=300)
            elapsed=time.monotonic()-start
            actual=json.loads(output.read_text());expected=json.loads(reference.read_text())
            assert actual.keys()==expected.keys()
            print(f'{scan.name}: {elapsed:.2f}s')
            for group in expected:
                assert actual[group].keys()==expected[group].keys()
                worst=0.
                for key,want in expected[group].items():
                    got=actual[group][key]
                    tolerance=max(8,.02*want) if group=='floor_sf' else max(2,.015*want) if group=='wall_lf' else 0
                    error=abs(got-want)
                    if error>tolerance:failures.append(f'{scan.name} {group}.{key}: {got} != {want} (tolerance {tolerance})')
                    if want:worst=max(worst,error/want*100)
                print(f'  {group}: '+(f'max error {worst:.3f}%' if group in ['floor_sf','wall_lf'] else 'exact' if worst==0 else f'max error {worst:.3f}%'))
    if failures:
        print('\n'.join(failures),file=sys.stderr)
        return 1
    print('All reference quantities pass.')
    return 0

if __name__=='__main__':sys.exit(main())
