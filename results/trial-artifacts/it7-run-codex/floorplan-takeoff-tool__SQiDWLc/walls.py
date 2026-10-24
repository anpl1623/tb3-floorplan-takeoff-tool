"""Find parallel wall faces, join the centerline graph, and classify the poche."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Wall:
    axis: int # 0: horizontal, 1: vertical
    c: float
    a: float
    b: float
    width: float
    kind: str = ''
    def ends(self):
        return np.array([[self.a,self.c],[self.b,self.c]]) if self.axis==0 else np.array([[self.c,self.a],[self.c,self.b]])

def merge_intervals(intervals,gap=0):
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]+gap:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def wall_features(patch,black):
    p=np.float32(patch)/black
    if min(p.shape)<3:return np.zeros(7)
    # A constant poche, parallel diagonal strokes, and a crosshatch have distinct
    # along-wall covariance. Keep both diagonal signs to support rotated walls.
    gx=cv2.Sobel(p,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(p,cv2.CV_32F,0,1,ksize=3)/8
    gm=gx+gy;gp=gx-gy
    es=sorted([float(np.mean(gm[1:-1,1:-1]**2)),float(np.mean(gp[1:-1,1:-1]**2))])
    return np.array([np.mean(p),np.std(p),np.std(np.mean(p,axis=0)),*np.sqrt(es),np.mean(np.abs(gx)),np.mean(np.abs(gy))])

def sample_wall(sheet,wall,margin=4):
    g=sheet.ink.T if wall.axis else sheet.ink
    y0=int(round(wall.c-wall.width/2+margin));y1=int(round(wall.c+wall.width/2-margin))+1
    x0=int(round(wall.a+10));x1=int(round(wall.b-10))
    return g[y0:y1,x0:x1]

def wall_references(sheet):
    refs={}
    for kind,(x,y,w,h) in zip(['EXT','P3','P2','P1'],sheet.wall_swatches):
        prof=np.mean(sheet.gray[y-3:y+h+4,x+8:x+w-8],axis=1)
        lo=int(np.argmin(prof[:len(prof)//2]))+y-3
        hi=int(np.argmin(prof[len(prof)//2:]))+len(prof)//2+y-3
        refs[kind]=wall_features(sheet.ink[lo+4:hi-3,x+10:x+w-10],sheet.black)
    return refs

def extract_faces(sheet):
    xend=int(sheet.swatches[0][0]-180);yend=int(sheet.gray.shape[0]*.87)
    g=sheet.gray[:yend,:xend]
    high=cv2.GaussianBlur(g,(0,0),3).astype(np.float32)-cv2.GaussianBlur(g,(0,0),.55).astype(np.float32)
    bw=np.uint8(high>sheet.black*.215)
    allfaces=[]
    ppf=sheet.ppf
    for axis in [0,1]:
        work=bw.T.copy() if axis else bw
        line=cv2.morphologyEx(work,cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))
        line=cv2.morphologyEx(line,cv2.MORPH_OPEN,np.ones((1,int(ppf*2.2)),np.uint8))
        n,lab,stats,_=cv2.connectedComponentsWithStats(line)
        faces=[]
        for x,y,w,h,p in stats[1:]:
            if w<ppf*2.2 or h>9:continue
            if axis==0 and (x<100 or y<200):continue
            if axis==1 and (y<100 or x<200):continue
            crop=high.T if axis else high
            # Contrast-weighted ridge location is less dependent on threshold
            # than the edge of a binarized line.
            weight=np.maximum(crop[y:y+h,x:x+w],0)*line[y:y+h,x:x+w]
            sums=np.sum(weight,axis=1)
            cy=float(np.sum(np.arange(y,y+h)*sums)/np.sum(sums))
            faces.append((cy,float(x),float(x+w)))
        faces.sort()
        allfaces.append(faces)
    return allfaces

def detect_bands(sheet,faces):
    bands=[]
    for axis,rs in enumerate(faces):
        for i,(y,x,X) in enumerate(rs):
            for y2,x2,X2 in rs[i+1:]:
                sep=y2-y
                if sep>sheet.ppf*.85:break
                if sep<sheet.ppf*.29:continue
                a=max(x,x2);b=min(X,X2)
                if b-a>sheet.ppf*2.2:
                    bands.append(Wall(axis,(y+y2)/2,a,b,sep))
    return bands

def group_bands(bands,tol=2.5):
    groups=[]
    for q in sorted(bands,key=lambda q:(q.axis,q.c,q.a)):
        candidates=[g for g in groups if g[0].axis==q.axis and abs(np.median([r.c for r in g])-q.c)<tol]
        if candidates:candidates[-1].append(q)
        else:groups.append([q])
    return groups

def find_exterior(sheet,bands):
    groups=group_bands(bands)
    h=[];v=[]
    for gg in groups:
        if np.median([q.width for q in gg])<sheet.ppf*.5:continue
        intervals=merge_intervals([(q.a,q.b) for q in gg],sheet.ppf*.1)
        length=sum(b-a for a,b in intervals)
        coord=np.median([q.c for q in gg])
        (v if gg[0].axis else h).append((coord,length))
    hs=[c for c,l in h if l>sheet.ppf*65]
    if len(hs)<2:raise ValueError('Exterior horizontal walls not found')
    top=min(hs);bottom=max(hs)
    vs=[c for c,l in v if l>(bottom-top)*.5]
    if len(vs)<2:raise ValueError('Exterior vertical walls not found')
    left=min(vs);right=max(vs)
    return np.array([left,top,right,bottom])

def measure_walls(sheet):
    faces=extract_faces(sheet)
    raw=detect_bands(sheet,faces)
    bounds=find_exterior(sheet,raw)
    left,top,right,bottom=bounds
    refs=wall_references(sheet)
    candidates=[]
    for q in raw:
        if q.axis==0:
            if q.c<top+sheet.ppf or q.c>bottom-sheet.ppf:continue
            q.a=max(q.a,left);q.b=min(q.b,right)
        else:
            if q.c<left+sheet.ppf or q.c>right-sheet.ppf:continue
            q.a=max(q.a,top);q.b=min(q.b,bottom)
        if q.b-q.a<sheet.ppf*2.2:continue
        patch=sample_wall(sheet,q)
        chunks=[patch[:,j:j+36] for j in range(0,patch.shape[1]-30,24)]
        ff=[wall_features(p,sheet.black) for p in chunks]
        f=np.median(ff,axis=0) if ff else wall_features(patch,sheet.black)
        # The thin partition is distinguishable by thickness. The other two
        # interior walls use the current architect's samples.
        if q.width<sheet.ppf*.49:
            q.kind='P1'
        else:
            weights=np.array([1.5,1.,.1,7.,7.,2.,2.])
            q.kind=min(['P2','P3'],key=lambda k:np.linalg.norm((f-refs[k])*weights))
        candidates.append(q)
    # Join pieces across openings and at wall intersections. Type changes stay
    # separate; a T junction extends each branch to the receiving centerline.
    merged=[]
    for group in group_bands(candidates):
        c=float(np.average([q.c for q in group],weights=[q.b-q.a for q in group]))
        for kind in sorted(set(q.kind for q in group)):
            qs=[q for q in group if q.kind==kind]
            width=float(np.median([q.width for q in qs]))
            for a,b in merge_intervals([(q.a,q.b) for q in qs],sheet.ppf*6.2):
                merged.append(Wall(group[0].axis,c,a,b,width,kind))
    # Remove isolated look-alikes (e.g. fronts of cooler banks). A wall must
    # participate in the wall graph, at one end at least.
    exterior=[Wall(0,top,left,right,sheet.ppf*2/3,'EXT'),Wall(0,bottom,left,right,sheet.ppf*2/3,'EXT'),Wall(1,left,top,bottom,sheet.ppf*2/3,'EXT'),Wall(1,right,top,bottom,sheet.ppf*2/3,'EXT')]
    kept=[]
    for q in merged:
        contacts=[]
        for r in merged+exterior:
            if q.axis==r.axis:continue
            if r.a-sheet.ppf*.6<=q.c<=r.b+sheet.ppf*.6:
                for isend,end in enumerate([q.a,q.b]):
                    if abs(end-r.c)<sheet.ppf*.65:contacts.append((isend,r.c))
        if not contacts:continue
        for isend,c in contacts:
            if isend:q.b=c
            else:q.a=c
        kept.append(q)
    walls=exterior+kept
    totals={k:0. for k in ['EXT','P1','P2','P3']}
    for q in walls:totals[q.kind]+=(q.b-q.a)/sheet.ppf
    canvas=cv2.cvtColor(sheet.gray[:int(bottom+sheet.ppf*4),:int(right+sheet.ppf*4)],cv2.COLOR_GRAY2BGR)
    colors={'EXT':(0,0,255),'P1':(0,180,0),'P2':(255,0,0),'P3':(255,0,255)}
    for q in walls:
        a,b=np.int32(q.ends());cv2.line(canvas,tuple(a),tuple(b),colors[q.kind],5)
    sheet.save('walls',cv2.resize(canvas,(int(canvas.shape[1]*.35),int(canvas.shape[0]*.35))))
    return totals,walls,bounds,faces
