#!/usr/bin/env python3
"""Integration check against paired example PDFs and estimator JSON files."""
import argparse,json,os,subprocess,sys,tempfile,time
from pathlib import Path


def compare(actual,expected):
    errors=[];area_pct=0.;length_pct=0.
    if actual.keys()!=expected.keys():errors.append('Top-level schema differs')
    for group,wanted in expected.items():
        got=actual.get(group,{})
        if got.keys()!=wanted.keys():errors.append(group+': schema differs')
        for key,value in wanted.items():
            measured=got.get(key)
            if not isinstance(measured,(int,float)):
                errors.append(group+'.'+key+': missing numeric quantity');continue
            delta=abs(measured-value)
            if group=='floor_sf':
                tolerance=max(8,.02*value);area_pct=max(area_pct,100*delta/max(value,1))
            elif group=='wall_lf':
                tolerance=max(2,.015*value);length_pct=max(length_pct,100*delta/max(value,1))
            else:
                tolerance=0
                if not isinstance(measured,int):errors.append(group+'.'+key+': count is not an integer')
            if delta>tolerance+1e-8:errors.append('%s.%s: got %s, expected %s, tolerance %s'%(group,key,measured,value,tolerance))
    return errors,area_pct,length_pct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples',nargs='?',type=Path,default=Path('/app/examples'))
    parser.add_argument('--unprivileged',action='store_true',help='when root, run the CLI as uid/gid 65534')
    args=parser.parse_args();script=Path(__file__).with_name('takeoff.py')
    def restrict():
        cpus=sorted(os.sched_getaffinity(0))[:2]
        os.sched_setaffinity(0,cpus)
        if args.unprivileged and os.geteuid()==0:
            os.setgroups([]);os.setgid(65534);os.setuid(65534)
    report=[]
    with tempfile.TemporaryDirectory(prefix='takeoff-validation-') as tmp:
        os.chmod(tmp,0o777)
        for pdf in sorted(args.examples.glob('*.pdf')):
            reference=pdf.with_suffix('.takeoff.json')
            if not reference.exists():continue
            output=Path(tmp)/(pdf.stem+'.json');start=time.monotonic()
            p=subprocess.run([sys.executable,str(script),str(pdf),str(output)],capture_output=True,text=True,timeout=300,preexec_fn=restrict,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            elapsed=time.monotonic()-start
            if p.returncode:raise RuntimeError('%s failed: %s'%(pdf.name,p.stderr))
            actual=json.loads(output.read_text());expected=json.loads(reference.read_text());errors,area_pct,length_pct=compare(actual,expected)
            row=dict(scan=pdf.name,passed=not errors,max_area_error_percent=round(area_pct,3),max_length_error_percent=round(length_pct,3),seconds=round(elapsed,2),errors=errors)
            report.append(row);print(json.dumps(row),flush=True)
    if not report:raise RuntimeError('No PDF/JSON example pairs found')
    return 0 if all(row['passed'] for row in report) else 1

if __name__=='__main__':raise SystemExit(main())
