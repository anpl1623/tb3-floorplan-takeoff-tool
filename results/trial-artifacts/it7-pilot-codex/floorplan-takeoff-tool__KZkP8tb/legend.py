import cv2,numpy as np,json,sys
from engine import components,CODES

def extract(ink,ppf,box,debug=None):
    h,w=ink.shape
    x0=max(int(box[2]+ppf*2),int(w*.60));x1=int(w*.88)
    a=ink[:,x0:x1]
    # Connected rectangle boundaries remain intact when their hatch touches the
    # perimeter, unlike a contour selected from the white interior.
    b=np.uint8(a>.3)
    contours,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for c in contours:
        x,y,r,t=cv2.boundingRect(c)
        if 5*ppf<r<9*ppf and 2.5*ppf<t<5*ppf:
            rects.append((x+x0,y,r,t))
    unique=[]
    for r in sorted(rects,key=lambda r:-r[2]*r[3]):
        if any(abs(r[0]-q[0])<ppf*.3 and abs(r[1]-q[1])<ppf*.3 for q in unique):continue
        unique.append(r)
    rects=unique
    best=[]
    for r in rects:
        seq=sorted([q for q in rects if abs(q[0]-r[0])<ppf*.25 and abs(q[2]-r[2])<ppf*.3 and abs(q[3]-r[3])<ppf*.3],key=lambda q:q[1])
        if len(seq)>len(best):best=seq
    if len(best)<8:raise ValueError('Could not read the eight floor swatches from the legend')
    floors=best[:8]
    xmin=int(np.median([q[0] for q in floors]));ww=int(np.median([q[2] for q in floors]));last=floors[-1][1]+floors[-1][3]
    # Read long horizontal strokes in the narrow swatch column. Group both
    # faces of a strip, including solid and hatched poche.
    col=ink[:,xmin:xmin+ww]
    persist=cv2.morphologyEx(col,cv2.MORPH_OPEN,np.ones((1,round(ww*.8)),np.uint8))
    pro=np.mean(persist,axis=1)
    rows=np.flatnonzero((pro>.25)&(np.arange(h)>last+ppf))
    groups=np.split(rows,np.flatnonzero(np.diff(rows)>ppf*.85)+1)
    walls=[]
    for g in groups:
        if len(g) and .15*ppf<g[-1]-g[0]+1<1.2*ppf:
            walls.append((xmin,int(g[0]),ww,int(g[-1]-g[0]+1)))
        if len(walls)==4:break
    if len(walls)!=4:raise ValueError('Could not read the partition swatches')
    out={'floor_boxes':floors,'wall_boxes':walls,'floors':[],'walls':[]}
    for kind,rs in [('floors',floors),('walls',walls)]:
        for i,(x,y,r,t) in enumerate(rs):
            # Reconstruct on the same physical pixel grid as the plan.
            patch=ink[y:y+t,x:x+r]
            patch=cv2.resize(patch,(round(r/ppf*16),round(t/ppf*16)),interpolation=cv2.INTER_AREA)
            out[kind].append(patch)
            if debug:cv2.imwrite(debug+'/legend-'+kind+str(i)+'.png',np.uint8((1-patch)*255))
    if debug:json.dump(dict(floor_boxes=floors,wall_boxes=walls),open(debug+'/legend.json','w'),indent=2)
    return out

