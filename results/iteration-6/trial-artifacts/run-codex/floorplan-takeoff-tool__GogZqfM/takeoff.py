#!/usr/bin/env python3
"""Quantity takeoff from a scanned Cedar Valley A101 fixture / finish plan.

Usage: python3 takeoff.py input.pdf output.json
Optional diagnostics: --debug-dir directory
"""
import os
# Stay inside the two-CPU runtime allocation, including BLAS helpers.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='1'

import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

from vision import load_sheet,find_scale,find_building,wall_candidates,refine_walls,floor_regions
from texture import sample_region,classify
from legend import wall_codes
from fixtures import count_fixtures
from doors import count_doors

cv2.setNumThreads(2)
ASSETS=Path(__file__).resolve().parent/'assets'
FINISHES=['LVT-1','PC-1','SC-1','EP-1','VCT-1','CT-1','CPT-1','WOM-1']


def takeoff(path,debug_dir=None):
    started=time.monotonic()
    sheet,angle=load_sheet(path)
    s,bar=find_scale(sheet)
    source_scale=s
    factor=24/s
    sheet=cv2.resize(sheet,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)
    bar=tuple(round(v*factor) for v in bar)
    s=24.0
    bbox=find_building(sheet,s)
    codes=wall_codes(sheet,s)
    x0,y0,x1,y1=map(round,bbox)
    pad=round(s*5)
    origin=(max(0,x0-pad),max(0,y0-pad))
    im=sheet[origin[1]:min(sheet.shape[0],y1+pad+1),origin[0]:min(sheet.shape[1],x1+pad+1)]
    candidates,lines=wall_candidates(im,s)
    walls=refine_walls(im,candidates,s)
    if len(walls)<4:raise ValueError('Incomplete wall geometry')
    # Exterior runs are the extreme horizontal and vertical enclosing walls.
    exterior=set()
    for orientation in ['h','v']:
        ix=[i for i,w in enumerate(walls) if w['ori']==orientation]
        low=min(walls[i]['c'] for i in ix);high=max(walls[i]['c'] for i in ix)
        exterior.update(i for i in ix if min(abs(walls[i]['c']-low),abs(walls[i]['c']-high))<s*.2)
    wall_lf={key:0.0 for key in ['EXT','P1','P2','P3']}
    for i,w in enumerate(walls):
        code='EXT' if i in exterior else codes[w['style']]
        w['code']=code
        wall_lf[code]+=(w['hi']-w['lo'])/s
    regions,ids,wm,trans=floor_regions(im,walls,s)
    model=np.load(ASSETS/'hatches.npz',allow_pickle=False)
    train=model['features'];labels=model['labels']
    floors={code:0.0 for code in FINISHES}
    floor_debug=[]
    for i in ids:
        mask=regions==i
        features=sample_region(im,mask,s)
        if not len(features):raise ValueError('A floor region is too small to recognize its hatch')
        finish,votes=classify(features,train,labels)
        area=float(mask.sum()/s**2)
        floors[finish]+=area
        floor_debug.append(dict(region=int(i),finish=finish,area=area,votes=votes))
    fixtures,fixture_debug=count_fixtures(im,s)
    doors,door_debug=count_doors(im,s,walls)
    result=dict(floor_sf={k:round(v,1) for k,v in floors.items()},
                wall_lf={k:round(v,2) for k,v in wall_lf.items()},doors=doors,fixtures=fixtures)
    if debug_dir:
        d=Path(debug_dir);d.mkdir(parents=True,exist_ok=True)
        diagnostics=dict(pixels_per_foot=s,source_pixels_per_foot=source_scale,deskew_degrees=angle,scale_bar=bar,
                         building_bbox=list(map(float,bbox)),walls=walls,
                         floors=floor_debug,fixtures=fixture_debug,doors=door_debug,
                         elapsed_seconds=time.monotonic()-started)
        (d/'measurements.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
        vis=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        rng=np.random.default_rng(42)
        for r in floor_debug:
            mask=regions==r['region'];ys,xs=np.where(mask)
            color=rng.integers(40,220,size=3)
            vis[mask]=(.55*vis[mask]+.45*color).astype(np.uint8)
            cv2.putText(vis,r['finish'],(int(np.median(xs)),int(np.median(ys))),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
        cv2.imwrite(str(d/'floors.png'),vis)
        vis=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        for w in walls:
            if w['ori']=='h':p=(round(w['lo']),round(w['c']));q=(round(w['hi']),round(w['c']))
            else:p=(round(w['c']),round(w['lo']));q=(round(w['c']),round(w['hi']))
            cv2.line(vis,p,q,(0,0,255),3)
        cv2.imwrite(str(d/'walls.png'),vis)
        vis=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        colors=[(255,0,0),(0,128,0),(255,0,255),(0,0,255),(0,128,255)]
        for color,key in zip(colors,['produce_tables','checkouts','self_checkouts','endcaps','cells']):
            for pos,(x,y,w,h) in fixture_debug[key]:
                cv2.rectangle(vis,(x,y),(x+w,y+h),color,2)
        for x,y,score,k in fixture_debug['cooler_doors']:cv2.circle(vis,(round(x),round(y)),6,(0,0,255),-1)
        cv2.imwrite(str(d/'fixtures.png'),vis)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan_pdf',type=Path)
    parser.add_argument('takeoff_json',type=Path)
    parser.add_argument('--debug-dir',type=Path)
    args=parser.parse_args()
    result=takeoff(args.scan_pdf,args.debug_dir)
    # Write atomically beside the requested destination; no write access to the
    # installed tool directory is needed.
    import tempfile
    parent=args.takeoff_json.parent
    parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+args.takeoff_json.name+'.',suffix='.tmp',dir=parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        os.replace(tmp,args.takeoff_json)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


if __name__=='__main__':
    main()
