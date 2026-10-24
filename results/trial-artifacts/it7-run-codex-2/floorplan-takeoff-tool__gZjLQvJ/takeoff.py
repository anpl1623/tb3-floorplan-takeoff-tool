#!/usr/bin/env python3
"""Take off a Cedar Valley Market scanned A101 fixture and finish plan.

Usage: python3 takeoff.py scan.pdf takeoff.json
Optional diagnostics: set TAKEOFF_DEBUG to a writable directory.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
from vision import read_scan,rectify,find_scale,find_legend,find_wall_swatches,normalize_gray,correct_aspect,edge_blur
from geometry import wall_candidates,find_exterior,interior_candidates,finish_walls,wall_lengths,wall_mask,recover_weak_walls
from floors import measure_floors,paper_model
from fixtures import gondolas,basic_fixtures,cooler_doors,endcaps,fixture_mask
from doors import swing_doors,sliding_doors,coiling_doors


def analyze(im,debug=None):
    if im.shape[0]>im.shape[1]:
        im=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
    first_error=None
    for attempt in range(2):
        try:
            return analyze_upright(im,debug)
        except ValueError as error:
            if first_error is None:first_error=error
            im=cv2.rotate(im,cv2.ROTATE_180)
    raise first_error


def analyze_upright(im,debug=None):
    if im.shape[1]!=7200:
        im=cv2.resize(im,(7200,round(im.shape[0]*7200/im.shape[1])),interpolation=cv2.INTER_CUBIC if im.shape[1]<7200 else cv2.INTER_AREA)
    im,frame=rectify(im)
    im,frame,aspect=correct_aspect(im,frame)
    raw=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    g=normalize_gray(raw)
    ink=cv2.subtract(cv2.GaussianBlur(g,(0,0),12),g)
    contrast=float(np.percentile(ink,99.8))
    scale,bar=find_scale(g,ink,frame)
    blur=edge_blur(raw,bar)
    if blur>1.45:
        additional=np.sqrt(max(0.,(blur/.90)**2-1.05**2))
        g=normalize_gray(raw,deblur_sigma=min(2.8,additional))
        ink=cv2.subtract(cv2.GaussianBlur(g,(0,0),12),g)
        contrast=float(np.percentile(ink,99.8))
        scale,bar=find_scale(g,ink,frame)
    legend=find_legend(ink,contrast)
    swatches=find_wall_swatches(ink,contrast,legend)
    candidates,edges=wall_candidates(ink,contrast,scale)
    rect,exterior=find_exterior(candidates,g.shape,scale,edges)
    interior=interior_candidates(candidates,rect,scale)
    interior=recover_weak_walls(ink,contrast,scale,interior,exterior,rect)
    walls,refs=finish_walls(g,ink,interior,exterior,scale,contrast,swatches)
    wall_lf=wall_lengths(walls,scale)
    wm=wall_mask(walls,g.shape)
    runs=gondolas(g,ink,rect,scale,contrast)
    caps=endcaps(g,ink,rect,scale,contrast,runs,wm)
    basic=basic_fixtures(g,ink,rect,scale,contrast,wm)
    coolers=cooler_doors(g,ink,rect,scale,contrast,wm)
    occlusions=fixture_mask(g.shape,runs,basic,coolers,scale)
    floor_sf,floor_info=measure_floors(g,ink,walls,rect,scale,contrast,legend,occlusions=occlusions)
    singles,pairs,swing_candidates,dws=swing_doors(g,ink,walls,scale,contrast)
    sliders,slide_candidates=sliding_doors(g,ink,walls,scale,contrast)
    rolling,opening_candidates=coiling_doors(g,ink,walls,scale,contrast)
    result={
        'floor_sf':{k:float(v) for k,v in floor_sf.items()},
        'wall_lf':wall_lf,
        'doors':{'single':len(singles),'pair':len(pairs),'overhead_coiling':len(rolling),'auto_sliding':len(sliders)},
        'fixtures':{
            'gondola_sections_4ft':sum(r['sections_4ft'] for r in runs),
            'gondola_sections_3ft':sum(r['sections_3ft'] for r in runs),
            'endcaps':len(caps),
            'checkouts':len(basic['checkouts']),
            'self_checkouts':len(basic['self_checkouts']),
            'cooler_doors':len(coolers),
            'produce_tables':len(basic['produce_tables'])}}
    if debug:
        diagnostic_output(debug,im,scale,bar,legend,walls,floor_info,runs,basic,coolers,singles,pairs,sliders,rolling,result)
    return result


def diagnostic_output(directory,im,scale,bar,legend,walls,fi,runs,basic,coolers,singles,pairs,sliders,rolling,result):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    overlay=im.copy()
    colors={'EXT':(0,0,220),'P1':(30,170,30),'P2':(220,80,20),'P3':(20,130,230)}
    for t in walls:
        if t.axis==0:a=(round(t.a),round(t.p));b=(round(t.b),round(t.p))
        else:a=(round(t.p),round(t.a));b=(round(t.p),round(t.b))
        cv2.line(overlay,a,b,colors[t.kind],5)
    for kind,ds in dict(basic,cooler_doors=coolers).items():
        for t in ds:
            x0,y0,x1,y1=np.round(t['box']).astype(int)
            cv2.rectangle(overlay,(x0,y0),(x1,y1),(220,20,180),3)
    for run in runs:
        p=run['p'];a=run['a'];b=run['b'];width=run['width']
        if run['axis']==1:box=(p-width/2,a,p+width/2,b)
        else:box=(a,p-width/2,b,p+width/2)
        x0,y0,x1,y1=np.round(box).astype(int);cv2.rectangle(overlay,(x0,y0),(x1,y1),(220,20,180),4)
    for kind,ds in [('single',singles),('pair',[q for p in pairs for q in p]),('auto_sliding',sliders),('overhead_coiling',rolling)]:
        for q in ds:
            s=q.get('hinge',q.get('center'));p=q['p']
            xy=(s,p) if q['axis']==0 else (p,s)
            cv2.circle(overlay,tuple(np.round(xy).astype(int)),round(scale*.8),(0,0,255),4)
    cv2.rectangle(overlay,(bar[0],bar[1]),(bar[2],bar[3]),(0,0,255),3)
    for x,y,w,h in legend:cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imwrite(str(d/'recognized.png'),overlay)
    pal=np.array([[110,210,190],[230,185,145],[125,205,245],[200,160,240],[170,230,180],[240,170,120],[170,160,200],[220,220,90]],np.uint8)
    labelim=np.full((*fi['labels'].shape,3),255,np.uint8)
    for i,k in fi['assignments'].items():labelim[fi['labels']==i]=pal[k]
    cv2.imwrite(str(d/'finishes.png'),labelim)
    with open(d/'measurements.json','w') as f:json.dump(result,f,indent=2);f.write('\n')
    with open(d/'calibration.json','w') as f:json.dump({'pixels_per_foot':scale,'scale_bar':bar,'finish_swatches':legend},f,indent=2,default=lambda q:q.item() if isinstance(q,np.generic) else q)


def main():
    if len(sys.argv)!=3:
        print('Usage: python3 /app/takeoff/takeoff.py <scan.pdf> <takeoff.json>',file=sys.stderr)
        return 2
    source,dest=sys.argv[1:]
    started=time.monotonic()
    result=analyze(read_scan(source),os.environ.get('TAKEOFF_DEBUG'))
    # Do not leave a partial JSON if analysis or writing fails.
    target=Path(dest)
    import tempfile
    fd,tmp=tempfile.mkstemp(prefix='.'+target.name+'.',dir=str(target.parent))
    try:
        with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print(f'Takeoff written in {time.monotonic()-started:.1f}s',file=sys.stderr)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
