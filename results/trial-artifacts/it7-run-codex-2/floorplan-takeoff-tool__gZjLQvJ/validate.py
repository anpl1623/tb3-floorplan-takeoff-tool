#!/usr/bin/env python3
"""Compare the CLI with an external directory of scans and estimator takeoffs."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    references=sorted(args.examples.glob('*.takeoff.json'))
    if not references:parser.error('No *.takeoff.json reference files found')
    results=[]
    with tempfile.TemporaryDirectory(prefix='takeoff-validation-') as temp:
        for reference in references:
            stem=reference.name[:-len('.takeoff.json')]
            scan=reference.with_name(stem+'.pdf')
            destination=Path(temp)/(stem+'.json')
            started=time.monotonic()
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('takeoff.py')),str(scan),str(destination)],capture_output=True,text=True,timeout=300)
            elapsed=time.monotonic()-started
            failures=[];metrics={'area_max_percent':0.,'length_max_percent':0.,'counts_exact':True}
            if run.returncode:
                failures.append(run.stderr.strip())
            else:
                expected=json.loads(reference.read_text())
                actual=json.loads(destination.read_text())
                if set(actual)!=set(expected):failures.append('Top-level schema differs')
                for category,items in expected.items():
                    if set(actual.get(category,{}))!=set(items):
                        failures.append(category+': keys differ');continue
                    for key,target in items.items():
                        value=actual[category][key]
                        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                            failures.append(category+'.'+key+': invalid number');continue
                        delta=abs(value-target)
                        if category=='floor_sf':
                            tolerance=max(8.,.02*target)
                            if target:metrics['area_max_percent']=max(metrics['area_max_percent'],100*delta/target)
                        elif category=='wall_lf':
                            tolerance=max(2.,.015*target)
                            if target:metrics['length_max_percent']=max(metrics['length_max_percent'],100*delta/target)
                        else:
                            tolerance=0.
                            if not isinstance(value,int) or value!=target:metrics['counts_exact']=False
                        if delta>tolerance:failures.append(f'{category}.{key}: {value}, expected {target}, tolerance {tolerance:.3f}')
            entry=dict(scan=scan.name,passed=not failures,seconds=round(elapsed,3),**metrics,failures=failures)
            results.append(entry)
            status='PASS' if entry['passed'] else 'FAIL'
            print(f"{stem}: {status}; {elapsed:.2f}s; max area error {metrics['area_max_percent']:.3f}%; max length error {metrics['length_max_percent']:.3f}%; counts {'exact' if metrics['counts_exact'] else 'differ'}")
            for failure in failures:print('  '+failure)
    if args.report:
        args.report.write_text(json.dumps(results,indent=2)+'\n')
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
