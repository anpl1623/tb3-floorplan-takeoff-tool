"""Conservative recognition of boxed finish-code callouts; texture matching is the fallback."""
from pathlib import Path
import cv2
import numpy as np


def word_image(patch):
    # Discard the callout border; room numbers also have boxes, but different text.
    g=patch[4:-4,4:-4]
    if min(g.shape)<3:return np.zeros((28,128),np.float32)
    dark=np.maximum(0.,np.percentile(g,92)-g.astype(np.float32))
    peak=np.percentile(dark,98)
    if peak<5:return np.zeros((28,128),np.float32)
    dark=np.clip(dark/peak,0,1)
    yy,xx=np.where(dark>.23)
    if not len(xx):return np.zeros((28,128),np.float32)
    dark=dark[yy.min():yy.max()+1,xx.min():xx.max()+1]
    h,w=dark.shape;f=min(22/h,122/w)
    rw,rh=max(1,round(w*f)),max(1,round(h*f))
    im=cv2.resize(dark,(rw,rh),interpolation=cv2.INTER_AREA if f<1 else cv2.INTER_LINEAR)
    out=np.zeros((28,128),np.float32);y=(28-rh)//2;x=(128-rw)//2
    out[y:y+rh,x:x+rw]=im
    return out


def tag_boxes(ink,rect,contrast):
    xl,yt,xr,yb=np.round(rect).astype(int)
    crop=ink[yt:yb+1,xl:xr+1]
    b=np.uint8(crop>contrast*.25)*255
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    cs,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cs:
        x,y,w,h=cv2.boundingRect(c)
        if 18<h<36 and 30<w<100 and 1.3<w/h<4 and cv2.contourArea(c)>.68*w*h:
            q=(x+xl,y+yt,w,h)
            if not any(abs(q[0]-a[0])<5 and abs(q[1]-a[1])<5 for a in boxes):boxes.append(q)
    return boxes


_BANK=None

def recognize_tags(g,ink,rect,contrast):
    global _BANK
    if _BANK is None:
        z=np.load(Path(__file__).with_name('finish_tags.npz'))
        ims=z['images'].astype(np.float32).reshape(len(z['labels']),-1)
        ims/=np.maximum(1e-6,np.linalg.norm(ims,axis=1,keepdims=True))
        _BANK=(ims,z['labels'])
    bank,labels=_BANK
    found=[]
    for x,y,w,h in tag_boxes(ink,rect,contrast):
        if h<24:continue
        im=word_image(g[y:y+h,x:x+w])
        variants=np.array([np.roll(im,(dy,dx),(0,1)).ravel() for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]])
        variants/=np.maximum(1e-6,np.linalg.norm(variants,axis=1,keepdims=True))
        scores=np.max(bank@variants.T,axis=1)
        by_label=np.array([np.max(scores[labels==k]) if np.any(labels==k) else 0 for k in range(8)])
        winner=int(np.argmax(by_label));best=float(by_label[winner])
        second=float(np.partition(by_label,-2)[-2]);negative=float(np.max(scores[labels<0]))
        if best>.82 and best-second>.055 and best-negative>.07:
            found.append(dict(code=winner,confidence=best,margin=best-max(second,negative),center=(x+w/2,y+h/2),box=(x,y,w,h)))
    return found
