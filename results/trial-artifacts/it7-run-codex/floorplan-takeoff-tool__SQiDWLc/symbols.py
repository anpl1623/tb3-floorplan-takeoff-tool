"""CAD block recognition in calibrated plan coordinates; no store-specific data."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

DIMS={'produce':(5,4),'sco':(3,3),'cooler':(2.5,.6),'endcap':(4,1.5),'gond4':(4,4),'gond3':(4,3),'checkout':(3.75,12)}

@dataclass
class Detection:
    kind: str
    score: float
    x: float
    y: float
    rotation: int
    w: float
    h: float

def block_pattern(kind,ppf):
    supersample=4;pad=4
    w,h=DIMS[kind]
    out=np.zeros((round((h*ppf+2*pad)*supersample),round((w*ppf+2*pad)*supersample)),np.uint8)
    def p(x,y):return (round((pad+x*ppf)*supersample),round((pad+y*ppf)*supersample))
    def line(x,y,X,Y):cv2.line(out,p(x,y),p(X,Y),255,round(1.8*supersample),cv2.LINE_AA)
    def rect(x,y,X,Y):cv2.rectangle(out,p(x,y),p(X,Y),255,round(1.8*supersample),cv2.LINE_AA)
    if kind=='checkout':
        rect(0,0,2.5,12);line(0,9,2.5,9);line(0,9,2.5,12);line(2.5,9,0,12)
        line(1.25,.5,1.25,8.5);rect(2.5,4.8,3.75,6.05)
    else:
        rect(0,0,w,h)
        if kind=='produce':line(0,0,w,h);line(w,0,0,h)
        elif kind=='sco':cv2.circle(out,p(w/2,h/2),round(.8*ppf*supersample),255,round(1.8*supersample),cv2.LINE_AA)
        elif kind=='cooler':line(0,0,w,h)
        elif kind=='endcap':line(0,h,w,0)
        elif kind in ['gond4','gond3']:line(w/2,0,w/2,h)
    out=cv2.resize(out,(round(w*ppf+2*pad),round(h*ppf+2*pad)),interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(out,(0,0),.45).astype(np.float32)/255

def overlap(a,b):
    area=max(0,min(a.x+a.w/2,b.x+b.w/2)-max(a.x-a.w/2,b.x-b.w/2))*max(0,min(a.y+a.h/2,b.y+b.h/2)-max(a.y-a.h/2,b.y-b.h/2))
    return area/max(min(a.w*a.h,b.w*b.h),1)

def suppress(ds,overlap_fraction=.5):
    selected=[]
    for d in sorted(ds,key=lambda d:d.score,reverse=True):
        if not any(overlap(d,q)>overlap_fraction for q in selected):selected.append(d)
    return selected

def near_wall(d,walls,ppf):
    for q in walls:
        c,a,b=(d.x,d.y-d.h/2,d.y+d.h/2) if q.axis else (d.y,d.x-d.w/2,d.x+d.w/2)
        if abs(c-q.c)<q.width/2+ppf*.15 and b>=q.a and a<=q.b:return True
    return False

def match_block(sheet,bounds,kind,threshold=.7,scale=.5):
    left,top,right,bottom=bounds
    x0=max(int(left-sheet.ppf*3),0);y0=max(int(top-sheet.ppf*3),0)
    ink=sheet.ink[y0:int(bottom+sheet.ppf*3),x0:int(right+sheet.ppf*3)]
    image=cv2.resize(ink,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    image=cv2.GaussianBlur(image,(0,0),.4)
    base=block_pattern(kind,sheet.ppf)
    detections=[]
    for rot in range(4):
        pattern=np.rot90(base,rot).copy()
        pattern=cv2.resize(pattern,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        h,w=pattern.shape
        score=cv2.matchTemplate(image,pattern,cv2.TM_CCOEFF_NORMED)
        local=score==cv2.dilate(score,np.ones((5,5),np.uint8))
        ys,xs=np.where(local&(score>threshold))
        for y,x in zip(ys,xs):
            cx=(x+w/2)/scale+x0;cy=(y+h/2)/scale+y0
            if not left-8<cx<right+8 or not top-8<cy<bottom+8:continue
            detections.append(Detection(kind,float(score[y,x]),cx,cy,rot,w/scale,h/scale))
    return suppress(detections)

def gondola_geometry(sheet,d):
    ppf=sheet.ppf
    w,h=4*ppf,(4 if d.kind=='gond4' else 3)*ppf
    if d.rotation%2:w,h=h,w
    x0=round(d.x-w/2);y0=round(d.y-h/2);x1=round(d.x+w/2);y1=round(d.y+h/2)
    patch=sheet.ink[y0-3:y1+4,x0-3:x1+4].copy()/sheet.black
    if d.rotation%2:patch=patch.T
    H,W=patch.shape
    if H<20 or W<20:return False
    scores=[]
    for axis,c in [(0,3),(0,H-4),(1,3),(1,W-4),(1,(W-1)/2)]:
        p=patch if axis==0 else patch.T;c=round(c)
        scores.append(float(max(np.mean(p[c-2:c+3,9:-9],axis=1))))
    halves=[]
    for x in [10,round(W/2)+6]:
        halves.append(patch[10:-10,x:x+round(W/2)-16].ravel())
    v=np.concatenate(halves)
    background=float(np.mean(v))
    # Require the two section ends, both run sides, and the double-sided spine.
    # Repeated floor grid lines fail the blank-body check.
    return min(scores)-background>.18 and background-float(np.percentile(v,35))<.032

def count_fixtures(sheet,walls,bounds):
    ds={}
    for kind,threshold in [('produce',.73),('sco',.73),('endcap',.73),('checkout',.75)]:
        ds[kind]=match_block(sheet,bounds,kind,threshold)
    gond=[]
    for kind in ['gond4','gond3']:
        gond.extend(d for d in match_block(sheet,bounds,kind,.69) if gondola_geometry(sheet,d))
    gond=suppress(gond)
    ds['gond4']=[d for d in gond if d.kind=='gond4']
    ds['gond3']=[d for d in gond if d.kind=='gond3']
    ds['cooler']=[d for d in match_block(sheet,bounds,'cooler',.70,1.) if not near_wall(d,walls,sheet.ppf)]
    totals={
        'gondola_sections_4ft':len(ds['gond4']),
        'gondola_sections_3ft':len(ds['gond3']),
        'endcaps':len(ds['endcap']),
        'checkouts':len(ds['checkout']),
        'self_checkouts':len(ds['sco']),
        'cooler_doors':len(ds['cooler']),
        'produce_tables':len(ds['produce'])
    }
    if sheet.debug:
        draw_detections(sheet,bounds,ds,'fixtures')
        import json
        from pathlib import Path
        with open(Path(sheet.debug)/'fixtures.json','w') as f:json.dump({k:[vars(d)for d in v]for k,v in ds.items()},f,indent=2)
    return totals,ds

def draw_detections(sheet,bounds,groups,name):
    left,top,right,bottom=bounds
    x0=max(int(left-120),0);y0=max(int(top-120),0)
    canvas=sheet.image[y0:int(bottom+120),x0:int(right+120)].copy()
    colors=[(0,0,220),(0,150,0),(200,0,0),(0,120,200),(200,0,150),(0,150,150),(150,80,0)]
    for i,(kind,ds) in enumerate(groups.items()):
        color=colors[i%len(colors)]
        for d in ds:
            x,y=d.x-x0,d.y-y0
            cv2.rectangle(canvas,(round(x-d.w/2),round(y-d.h/2)),(round(x+d.w/2),round(y+d.h/2)),color,2)
            cv2.putText(canvas,f'{kind} {d.score:.2f}',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    sheet.save(name,cv2.resize(canvas,None,fx=.4,fy=.4))
