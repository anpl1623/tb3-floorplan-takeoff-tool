"""Small offline digit reader for graphic-scale captions."""
from pathlib import Path
import numpy as np
import cv2


def digit_image(dark):
    dark=np.maximum(dark.astype(np.float32),0)
    if dark.size==0:return np.zeros((28,22),np.float32)
    peak=np.percentile(dark,98)
    if peak<=0:return np.zeros((28,22),np.float32)
    dark=np.clip(dark/peak,0,1)
    ys,xs=np.where(dark>.22)
    if not len(xs):return np.zeros((28,22),np.float32)
    dark=dark[ys.min():ys.max()+1,xs.min():xs.max()+1]
    h,w=dark.shape;f=min(24/h,19/w)
    rh,rw=max(1,round(h*f)),max(1,round(w*f))
    resized=cv2.resize(dark,(rw,rh),interpolation=cv2.INTER_AREA if f<1 else cv2.INTER_LINEAR)
    canvas=np.zeros((28,22),np.float32);y=(28-rh)//2;x=(22-rw)//2
    canvas[y:y+rh,x:x+rw]=resized
    return canvas


def split_number(g,contrast):
    # Caption paper is local, so coloured paper and uneven scan exposure cancel.
    white=np.percentile(g,90)
    dark=np.maximum(0.,white-g.astype(np.float32))
    b=np.uint8(dark>max(5,contrast*.18))
    n,l,st,ce=cv2.connectedComponentsWithStats(b)
    chars=[]
    for i,(x,y,w,h,area) in enumerate(st):
        if i and area>=7 and h>=7 and h<=g.shape[0]*.65 and w<=g.shape[1]*.65:
            chars.append((int(x),int(y),int(w),int(h)))
    if not chars:return []
    # Only components on the numeric baseline, excluding stray paper specks.
    maxh=max(q[3] for q in chars)
    chars=[q for q in chars if q[3]>.6*maxh]
    def split(q):
        x,y,w,h=q
        if w<=h*1.04:return [q]
        col=np.sum(dark[y:y+h,x:x+w],axis=0)
        lo=max(2,round(w*.25));hi=min(w-2,round(w*.75))
        cut=lo+int(np.argmin(col[lo:hi]))
        return split((x,y,cut,h))+split((x+cut,y,w-cut,h))
    chars=[part for q in chars for part in split(q)]
    chars.sort()
    return [(q,digit_image(dark[q[1]:q[1]+q[3],q[0]:q[0]+q[2]])) for q in chars]


_BANK=None

def read_number(g,contrast):
    global _BANK
    if _BANK is None:
        z=np.load(Path(__file__).with_name('digits.npz'))
        _BANK=(z['images'].astype(np.float32),z['labels'])
    templates,labels=_BANK
    chars=split_number(g,contrast)
    if not chars or len(chars)>3:return None,1.
    value='';errors=[]
    for box,char in chars:
        # Small translations allow for asymmetric antialiasing on faded scans.
        distances=[]
        for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
            moved=np.roll(char,(dy,dx),(0,1))
            # Compare ink masks rather than the paper/background brightness.
            dist=np.mean((templates-moved)**2,axis=(1,2))
            distances.append(dist)
        ds=np.min(distances,axis=0);j=int(np.argmin(ds));value+=str(int(labels[j]));errors.append(float(ds[j]))
    return int(value),float(np.mean(errors))


def scale_caption(g,contrast,bar):
    xa,ya,xb,yb=bar
    candidates=[]
    for fraction,multiple in [(1.,1),(.5,2),(.25,4),(.125,8)]:
        x=round(xa+fraction*(xb-xa))
        crop=g[yb+10:min(g.shape[0],yb+80),x-38:x+38]
        value,error=read_number(crop,contrast)
        if value is not None and value>0 and error<.16:
            candidates.append((value*multiple,error))
    if not candidates:return 32.
    # Several labels vote on the same total span (e.g. 4,8,16,32).
    totals={q[0] for q in candidates}
    best=min(totals,key=lambda v:sum(min(.2,e+.11*(t!=v)) for t,e in candidates))
    return float(best)
