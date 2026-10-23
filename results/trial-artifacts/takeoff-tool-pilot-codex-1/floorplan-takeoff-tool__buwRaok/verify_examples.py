#!/usr/bin/env python3
"""Check the CLI against the supplied hand takeoffs, using two CPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def compare(actual,expected):
    errors=[]
    if set(actual)!=set(expected):return ['Output categories differ from the reference schema']
    for category,values in expected.items():
        if set(actual[category])!=set(values):
            errors.append(f'{category}: output keys differ from the reference schema')
            continue
        for code,true in values.items():
            measured=actual[category][code]
            if category=='floor_sf':tolerance=max(8,.02*true)
            elif category=='wall_lf':tolerance=max(2,.015*true)
            else:
                tolerance=0
                if type(measured)is not int:errors.append(f'{category}.{code}: count is not an integer')
            if not isinstance(measured,(int,float)) or measured<0 or abs(measured-true)>tolerance:
                errors.append(f'{category}.{code}: got {measured}, expected {true}, tolerance {tolerance:g}')
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',nargs='?',type=Path,default=Path('/app/examples'))
    parser.add_argument('--unprivileged',action='store_true',help='when root, run the CLI as uid/gid 65534')
    args=parser.parse_args()
    scans=sorted(args.examples.glob('*.pdf'))
    if not scans:
        parser.error('No PDF examples found')
    allowed=sorted(os.sched_getaffinity(0))[:2] if hasattr(os,'sched_getaffinity') else None
    def restrict():
        if allowed is not None:os.sched_setaffinity(0,allowed)
        if args.unprivileged and os.geteuid()==0:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
    failures=0
    script=Path(__file__).resolve().parent/'takeoff.py'
    with tempfile.TemporaryDirectory(prefix='takeoff-verification-') as temporary:
        work=Path(temporary)
        if args.unprivileged and os.geteuid()==0:work.chmod(0o777)
        for scan in scans:
            reference=scan.with_suffix('.takeoff.json')
            if not reference.exists():continue
            output=work/(scan.stem+'.json')
            started=time.monotonic()
            try:
                result=subprocess.run([sys.executable,str(script),str(scan.resolve()),str(output)],capture_output=True,text=True,timeout=300,preexec_fn=restrict)
                if result.returncode:errors=[result.stderr.strip() or f'CLI exited {result.returncode}']
                else:
                    actual=json.loads(output.read_text())
                    expected=json.loads(reference.read_text())
                    errors=compare(actual,expected)
            except (subprocess.TimeoutExpired,OSError,ValueError) as error:errors=[str(error)]
            elapsed=time.monotonic()-started
            print(f'{scan.name}: {"FAIL" if errors else "PASS"} ({elapsed:.1f} s)',flush=True)
            for error in errors:print('  '+error)
            failures+=bool(errors)
    return 1 if failures else 0

if __name__=='__main__':
    raise SystemExit(main())
