"""Raster sheet registration and measurements. All working coordinates are pixels."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import cv2
import fitz
import numpy as np
from dataclasses import dataclass
cv2.setNumThreads(2)

@dataclass
class Sheet:
    image: np.ndarray
    gray: np.ndarray
    ink: np.ndarray
    swatches: list
    wall_swatches: list
    ppf: float
    black: float
    debug: str | None = None
    def save(self, name, image):
        if self.debug:
            os.makedirs(self.debug,exist_ok=True)
            cv2.imwrite(os.path.join(self.debug,name+'.png'),image)

def intersection(a,b):
    aa=np.cross(np.r_[a[:2],1.],np.r_[a[2:],1.])
    bb=np.cross(np.r_[b[:2],1.],np.r_[b[2:],1.])
    q=np.cross(aa,bb)
    return q[:2]/q[2]

def register(image):
    h,w=image.shape[:2]
    if h>w:
        image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE);h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sm=cv2.resize(gray,(w//2,h//2),interpolation=cv2.INTER_AREA)
    ls=cv2.createLineSegmentDetector().detect(sm)[0]
    if ls is None: raise ValueError('No sheet frame detected')
    ls=ls[:,0]*2
    dl=ls[:,2:]-ls[:,:2];length=np.linalg.norm(dl,axis=1)
    hs=ls[(np.abs(dl[:,1])<.20*np.abs(dl[:,0]))&(length>w*.68)]
    vs=ls[(np.abs(dl[:,0])<.20*np.abs(dl[:,1]))&(length>h*.68)]
    if len(hs)<2 or len(vs)<2:
        raise ValueError('Unable to register the drawing frame')
    top=hs[np.argmin((hs[:,1]+hs[:,3])/2)]
    bot=hs[np.argmax((hs[:,1]+hs[:,3])/2)]
    left=vs[np.argmin((vs[:,0]+vs[:,2])/2)]
    right=vs[np.argmax((vs[:,0]+vs[:,2])/2)]
    corners=np.float32([intersection(top,left),intersection(top,right),intersection(bot,right),intersection(bot,left)])
    # A fixed working frame avoids changing the feature sizes with the PDF DPI.
    m=cv2.getPerspectiveTransform(corners,np.float32([[0,0],[6840,0],[6840,4500],[0,4500]]))
    return cv2.warpPerspective(image,m,(6841,4501),flags=cv2.INTER_LINEAR,borderValue=(255,255,255))

def rectangles(gray):
    ed=cv2.Canny(gray,30,75)
    cs,_=cv2.findContours(ed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return [(cv2.boundingRect(c),cv2.contourArea(c)) for c in cs]

def find_swatches(gray):
    h,w=gray.shape
    rs=[]
    x0=int(w*.60);x1=int(w*.89);y0=int(h*.08);y1=int(h*.40)
    for (x,y,ww,hh),area in rectangles(gray[y0:y1,x0:x1]):
        if 115<ww<270 and 50<hh<140 and 1.65<ww/hh<2.35 and area>.70*ww*hh:
            r=(x+x0,y+y0,ww,hh)
            
            near=[j for j,q in enumerate(rs) if abs(r[0]-q[0])<5 and abs(r[1]-q[1])<5]
            if not near:rs.append(r)
            elif r[2]*r[3]>rs[near[0]][2]*rs[near[0]][3]:rs[near[0]]=r
    best=None
    for r in rs:
        x,y,ww,hh=r
        step=hh*4/3
        chain=[];err=0
        for j in range(8):
            yy=y+j*step
            cand=[q for q in rs if abs(q[0]-x)<8 and abs(q[2]-ww)<10 and abs(q[1]-yy)<12 and abs(q[3]-hh)<10]
            if not cand:break
            q=min(cand,key=lambda q:abs(q[1]-yy));chain.append(q);err+=abs(q[1]-yy)
        if len(chain)==8 and (best is None or err<best[0]):best=(err,chain)
    if best is None:raise ValueError('Unable to locate the eight finish swatches')
    return best[1]

def find_wall_swatches(gray,swatches):
    x,y,w,h=swatches[-1]
    # Locate the four narrow wall samples below the floor finish samples.
    crop=gray[y+h+100:y+h+650,x-12:x+w+12]
    ls=cv2.createLineSegmentDetector().detect(crop)[0]
    candidates=[]
    if ls is not None:
        for xx,yy,XX,YY in ls[:,0]:
            if abs(xx-XX)>.7*w and abs(yy-YY)<3:
                candidates.append(float((yy+YY)/2+y+h+100))
    candidates.sort()
    groups=[]
    for q in candidates:
        if groups and q-groups[-1][-1]<25:groups[-1].append(q)
        else:groups.append([q])
    bands=[(x,int(round(min(g))),w,int(round(max(g)-min(g)))) for g in groups if 4<max(g)-min(g)<26]
    if len(bands)!=4:
        # The sheet's legend has a regular row grid even if a faded edge was lost.
        bands=[(x,int(y+h+186+j*97),w,16 if j<3 else 10) for j in range(4)]
    return bands

def find_scale(gray,ink):
    h,w=gray.shape;y0=int(h*.86);x0=int(w*.18);x1=int(w*.65)
    bw=np.uint8(ink[y0:,x0:x1]>80)*255
    bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    cs,_=cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in cs:
        x,y,ww,hh=cv2.boundingRect(c)
        if 450<ww<1400 and 15<hh<45 and 20<ww/hh<45:
            roi=bw[y:y+hh,x:x+ww]
            dens=[np.mean(roi[3:-3,int(ww*f):int(ww*(f+.05))]>0) for f in [.03,.17,.30,.42,.65,.85]]
            score=dens[0]+dens[2]+dens[3]-dens[1]-dens[4]-dens[5]
            candidates.append((score,x+x0,y+y0,ww,hh))
    if not candidates: raise ValueError('Graphic scale bar not found')
    _,x,y,ww,hh=max(candidates)
    # Stroke centers, rather than the outer edges of the bar, define its length.
    ppf=(ww-3)/32.
    return ppf,(x,y,ww,hh)

def read_sheet(path,debug=None):
    with fitz.open(path) as doc:
        if len(doc)!=1:raise ValueError('Expected a one-sheet PDF')
        page=doc[0]
        pix=page.get_pixmap(matrix=fitz.Matrix(7200/max(page.rect.width,page.rect.height),7200/max(page.rect.width,page.rect.height)),alpha=False,colorspace=fitz.csRGB)
        rgb=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
        image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    image=register(image)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    white=cv2.dilate(gray,cv2.getStructuringElement(cv2.MORPH_RECT,(91,91)))
    ink=white.astype(np.float32)-gray.astype(np.float32)
    swatches=find_swatches(gray)
    walls=find_wall_swatches(gray,swatches)
    # Contrast is learned from the heavy outlines in the current sheet's legend.
    x,y,w,h=walls[0]
    black=float(np.percentile(ink[max(0,y-4):y+h+5,x+5:x+w-5],97))
    ppf,scale=find_scale(gray,ink)
    sheet=Sheet(image,gray,ink,swatches,walls,ppf,black,debug)
    sheet.save('registered',cv2.resize(image,(1710,1125)))
    return sheet
