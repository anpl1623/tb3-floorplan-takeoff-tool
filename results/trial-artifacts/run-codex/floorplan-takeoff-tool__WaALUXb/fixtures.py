"""Scale-normalized matching of the archive's CAD fixture blocks."""
from pathlib import Path
import cv2
import numpy as np


def block(kind,foot=20,pad=.15):
    dims={'g4':(4,4),'g3':(4,3),'cap':(4,1.5),'check':(3.75,12),
          'self':(3,3),'produce':(5,4)}
    w,h=dims[kind];s=foot*4
    image=np.full((round((h+2*pad)*s)+1,round((w+2*pad)*s)+1),248,np.uint8)
    def point(x,y):return round((x+pad)*s),round((y+pad)*s)
    def line(x1,y1,x2,y2):
        cv2.line(image,point(x1,y1),point(x2,y2),160,round(s*.055),cv2.LINE_AA)
    def rect(x1,y1,x2,y2):
        line(x1,y1,x2,y1);line(x2,y1,x2,y2);line(x2,y2,x1,y2);line(x1,y2,x1,y1)
    if kind in ['g4','g3']:
        rect(0,0,w,h);line(2,0,2,h)
    elif kind=='cap':
        rect(0,0,w,h);line(0,h,w,0)
    elif kind=='check':
        rect(0,0,2.5,12);line(1.25,.4,1.25,8.6);line(0,9,2.5,9)
        line(0,9,2.5,12);line(0,12,2.5,9);rect(2.5,4.75,3.75,6)
    elif kind=='self':
        rect(0,0,w,h)
        cv2.circle(image,point(1.5,1.5),round(s*5/6),160,round(s*.055),cv2.LINE_AA)
    elif kind=='produce':
        rect(0,0,w,h);line(0,0,w,h);line(0,h,w,0)
    return cv2.resize(image,(round(image.shape[1]/4),round(image.shape[0]/4)),interpolation=cv2.INTER_AREA)


def peaks(image,template,threshold,minsep=18):
    response=cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
    maximum=cv2.dilate(response,np.ones((minsep,minsep),np.uint8))
    yy,xx=np.where((response==maximum)&(response>threshold))
    return [(float(response[y,x]),int(x),int(y)) for x,y in zip(xx,yy)]


def overlap(a,b):
    x0=max(a['x'],b['x']);x1=min(a['x']+a['w'],b['x']+b['w'])
    y0=max(a['y'],b['y']);y1=min(a['y']+a['h'],b['y']+b['h'])
    area=max(0,x1-x0)*max(0,y1-y0)
    return area/(a['w']*a['h']+b['w']*b['h']-area)


def suppress(hits,iou=.3):
    kept=[]
    for hit in sorted(hits,key=lambda h:-h['score']):
        if all(overlap(hit,other)<iou for other in kept):kept.append(hit)
    return kept


def count(plan,scale,assets):
    p=cv2.resize(plan,None,fx=20/scale,fy=20/scale,interpolation=cv2.INTER_AREA)
    result={};detections=[]
    configs=[('check',4,.70),('self',1,.70),('produce',2,.70),('cap',2,.70),
             ('g4',2,.75),('g3',2,.75),('cooler',4,.81)]
    gondolas=[]
    for kind,rotations,threshold in configs:
        template=(cv2.imread(str(Path(assets)/'cooler_front.png'),cv2.IMREAD_GRAYSCALE)
                  if kind=='cooler' else block(kind))
        if template is None:raise ValueError('Missing cooler symbol reference')
        hits=[]
        for rotation in range(rotations):
            temp=np.rot90(template,rotation).copy()
            for score,x,y in peaks(p,temp,threshold):
                hits.append(dict(kind=kind,score=score,x=x,y=y,w=temp.shape[1],h=temp.shape[0],rotation=rotation))
        if kind in ['g4','g3']:
            gondolas.extend(hits)
        else:
            kept=suppress(hits)
            detections.extend(kept);result[kind]=len(kept)
    gondolas=suppress(gondolas,.30)
    detections.extend(gondolas)
    result['g4']=sum(h['kind']=='g4' for h in gondolas)
    result['g3']=sum(h['kind']=='g3' for h in gondolas)
    output={'gondola_sections_4ft':result['g4'],'gondola_sections_3ft':result['g3'],
            'endcaps':result['cap'],'checkouts':result['check'],'self_checkouts':result['self'],
            'cooler_doors':result['cooler'],'produce_tables':result['produce']}
    return output,detections,p
