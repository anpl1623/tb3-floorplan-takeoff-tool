#!/usr/bin/env python3
"""Quantity takeoff for scanned Cedar Valley Market A101 plans.

Usage: python3 takeoff.py scan.pdf takeoff.json
All recognition runs locally, without executables, downloads, or model services.
"""
import os
import sys
# Set these before importing numerical libraries, including on machines with many
# host CPUs. The runtime contract allows only two CPUs.
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
sys.dont_write_bytecode=True

import argparse
import json
import tempfile
from pathlib import Path

from vision import normalize
from walls import get_walls
from calibration import calibrate_grid
from floors import floor_quantities,wall_geometry
from fixtures import fixture_quantities
from doors import door_quantities


def takeoff(scan_path,debug_dir=None):
    padded,meta=normalize(scan_path,margin=6)
    padded,meta=calibrate_grid(padded,meta)
    padding=meta['padding']
    plan=padded[padding:-padding,padding:-padding]
    walls=get_walls(plan)
    wall_lf={code:0.0 for code in ('EXT','P1','P2','P3')}
    for wall in wall_geometry(plan,walls):
        wall_lf[wall['typ']]+=wall['hi']-wall['lo']
    wall_lf={code:round(float(length),2) for code,length in wall_lf.items()}
    floor_debug={} if debug_dir else None
    fixture_debug={} if debug_dir else None
    door_debug={} if debug_dir else None
    result={
        'floor_sf':floor_quantities(plan,walls,floor_debug),
        'wall_lf':wall_lf,
        'doors':door_quantities(plan,walls,padded,padding,door_debug),
        'fixtures':fixture_quantities(plan,walls,fixture_debug),
    }
    if debug_dir:
        import cv2
        debug_dir=Path(debug_dir)
        debug_dir.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(debug_dir/'normalized-plan.png'),plan)
        labels=floor_debug.pop('_labels')
        render_finishes(plan,labels,result['floor_sf'],debug_dir)
        (debug_dir/'measurements.json').write_text(json.dumps(dict(scan=meta,walls=wall_geometry(plan,walls),floors=floor_debug,doors=door_debug,fixtures=fixture_debug),indent=2,default=lambda x:float(x)))
        render_debug(plan,padded,walls,fixture_debug,door_debug,padding,debug_dir)
    return result


def render_finishes(plan,labels,quantities,directory):
    import cv2
    import numpy as np
    palette=np.array([[255,255,255],[205,180,235],[235,220,190],[190,215,195],[170,205,245],[160,230,235],[225,200,165],[215,180,200],[190,190,190]],dtype='uint8')
    image=cv2.addWeighted(cv2.cvtColor(plan,cv2.COLOR_GRAY2BGR),.7,palette[labels],.3,0)
    image=cv2.copyMakeBorder(image,0,0,0,320,cv2.BORDER_CONSTANT,value=(255,255,255))
    x=plan.shape[1]+15
    for row,(code,area) in enumerate(quantities.items()):
        y=35+row*45
        cv2.rectangle(image,(x,y-18),(x+25,y+6),palette[row+1].tolist(),-1)
        cv2.putText(image,f'{code}: {area:.1f} sf',(x+35,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(40,40,40),1,cv2.LINE_AA)
    cv2.imwrite(str(directory/'floor-finishes.png'),image)


def render_debug(plan,padded,walls,fixtures,doors,padding,directory):
    import cv2
    import numpy as np
    image=cv2.cvtColor(padded,cv2.COLOR_GRAY2BGR)
    for wall in wall_geometry(plan,walls):
        lo=round(wall['lo']*24)+padding;hi=round(wall['hi']*24)+padding;pos=round(wall['pos']*24)+padding
        a=(lo,pos) if wall['axis']==0 else(pos,lo)
        b=(hi,pos) if wall['axis']==0 else(pos,hi)
        cv2.line(image,a,b,{'EXT':(220,0,200),'P1':(0,170,0),'P2':(220,0,0),'P3':(0,0,220)}[wall['typ']],2)
    for key in ('produce','checkout','self_checkout','endcap'):
        for score,x,y,w,h,rot in fixtures[key]:
            a=(round(x)+padding,round(y)+padding);b=(round(x+w)+padding,round(y+h)+padding)
            cv2.rectangle(image,a,b,(0,160,230),2)
    for detection,size in fixtures['sections']:
        score,x,y,w,h,rot=detection
        a=(round(x)+padding,round(y)+padding);b=(round(x+w)+padding,round(y+h)+padding)
        cv2.rectangle(image,a,b,(180,130,0),2)
        cv2.putText(image,str(size),a,cv2.FONT_HERSHEY_SIMPLEX,.6,(180,130,0),2)
    for score,cx,cy,rot,detection in fixtures['coolers']:
        cv2.circle(image,(round(cx)+padding,round(cy)+padding),8,(0,120,255),2)
    for door in doors['swings']:
        hinge=np.array(door['hinge']);leaf=np.array(door['leaf']);swing=np.array(door['swing'])
        t=np.linspace(0,np.pi/2,70)
        arc=hinge+door['radius']*(np.cos(t)[:,None]*leaf+np.sin(t)[:,None]*swing)
        cv2.polylines(image,[np.round(arc).astype('int32')],False,(0,0,255),2)
    geometry=wall_geometry(plan,walls)
    for wall_index,center,score in doors['sliding']:
        wall=geometry[wall_index]
        point=(round(center*24)+padding,round(wall['pos']*24)+padding) if wall['axis']==0 else (round(wall['pos']*24)+padding,round(center*24)+padding)
        cv2.circle(image,point,20,(220,100,0),3)
        cv2.putText(image,'SL',point,cv2.FONT_HERSHEY_SIMPLEX,.65,(220,100,0),2)
    for wall_index,low,high in doors['coiling']:
        wall=geometry[wall_index];pos=round(wall['pos']*24)+padding
        a=(round(low*24)+padding,pos) if wall['axis']==0 else (pos,round(low*24)+padding)
        b=(round(high*24)+padding,pos) if wall['axis']==0 else (pos,round(high*24)+padding)
        cv2.line(image,a,b,(220,0,180),5)
    cv2.imwrite(str(directory/'recognition.png'),image)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan_pdf',type=Path)
    parser.add_argument('takeoff_json',type=Path)
    parser.add_argument('--debug',type=Path,help='write measurement details and a recognition overlay to this directory')
    args=parser.parse_args()
    try:
        if args.scan_pdf.resolve()==args.takeoff_json.resolve():
            raise ValueError('Input and output paths must differ')
        result=takeoff(args.scan_pdf,args.debug)
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        # Atomic replacement leaves an existing takeoff intact if recognition or
        # writing fails. No writable installation directory is needed.
        temporary=None
        args.takeoff_json.parent.mkdir(parents=True,exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=args.takeoff_json.parent,prefix='.'+args.takeoff_json.name+'.',suffix='.tmp',delete=False) as out:
                temporary=Path(out.name)
                out.write(text)
            os.replace(temporary,args.takeoff_json)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()
    except Exception as error:
        print(f'takeoff: {error}',file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
