"""Offline raster drawing measurements."""
from pathlib import Path
import sys,os,math,json
import cv2
import numpy as np
from scipy import ndimage as ndi
import fitz
cv2.setNumThreads(2)

CODES=['LVT-1','PC-1','SC-1','EP-1','VCT-1','CT-1','CPT-1','WOM-1']

def load_scan(path):
    doc=fitz.open(path)
    if not len(doc): raise ValueError('Empty PDF')
    p=doc[0]
    ims=p.get_images(full=True)
    if ims and max(i[2]*i[3] for i in ims)>1e6:
        q=max(ims,key=lambda i:i[2]*i[3]); raw=doc.extract_image(q[0])
        a=cv2.imdecode(np.frombuffer(raw['image'],np.uint8),cv2.IMREAD_GRAYSCALE)
        # Embedded scans normally occupy the entire page. Render when it is a
        # composite or has a PDF rotation, so the page's transforms are applied.
        if len(ims)>1 or p.rotation:
            pix=p.get_pixmap(matrix=fitz.Matrix(2.5,2.5),colorspace=fitz.csGRAY)
            a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width).copy()
    else:
        pix=p.get_pixmap(matrix=fitz.Matrix(2.5,2.5),colorspace=fitz.csGRAY)
        a=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width).copy()
    if max(a.shape)>8500:
        a=cv2.resize(a,None,fx=8500/max(a.shape),fy=8500/max(a.shape),interpolation=cv2.INTER_AREA)
    return a

def normalize(a):
    h,w=a.shape
    small=cv2.resize(a,(max(1,w//8),max(1,h//8)),interpolation=cv2.INTER_AREA)
    bg=cv2.dilate(small,np.ones((91,91),np.uint8))
    bg=cv2.GaussianBlur(bg,(0,0),15)
    bg=cv2.resize(bg,(w,h),interpolation=cv2.INTER_CUBIC)
    ink=np.clip(1-a.astype(np.float32)/np.maximum(bg,1),0,1)
    # Set the darkest architectural strokes to a consistent strength. Do not
    # threshold: faded hatch marks carry useful information.
    black=np.percentile(ink,99.9)
    ink=np.clip((ink-.012)/(max(black,.2)-.012),0,1)
    return ink

def deskew(ink):
    h,w=ink.shape
    sm=cv2.resize(ink,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(np.uint8(255*(1-sm)))[0]
    if lines is None: return ink,0
    lines=lines[:,0].astype(float)
    ang=np.rad2deg(np.arctan2(lines[:,3]-lines[:,1],lines[:,2]-lines[:,0]));ang=(ang+45)%90-45
    weights=np.linalg.norm(lines[:,2:]-lines[:,:2],axis=1)
    sel=(np.abs(ang)<4)&(weights>250)
    if not sel.any():return ink,0
    order=np.argsort(ang[sel]);sang=ang[sel][order];sw=weights[sel][order]
    theta=sang[np.searchsorted(np.cumsum(sw),sum(sw)/2)]
    close=(np.abs(ang-theta)<.04)&(weights>250)
    if close.any(): theta=np.average(ang[close],weights=weights[close]**2)
    if abs(theta)>.002:
        mat=cv2.getRotationMatrix2D((w/2,h/2),theta,1)
        ink=cv2.warpAffine(ink,mat,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)
    return ink,float(theta)

def components(b):
    n,lab,st,cen=cv2.connectedComponentsWithStats(np.uint8(b),8)
    return st[1:],cen[1:]

def scale_bar(ink,return_reading=False):
    h,w=ink.shape
    b=np.uint8(ink>.55)
    b[:int(.65*h)]=0;b[:,int(.67*w):]=0
    b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((max(3,h//1000),max(9,w//650)),np.uint8))
    st,cen=components(b)
    rects=[s for s in st if s[2]>w*.0025 and h*.0015<s[3]<h*.012 and .8<s[2]/s[3]<35 and s[4]>s[2]*s[3]*.7]
    pairs=[]
    for a in rects:
        x,y,r,t,_=a
        for b in rects:
            u,v,s,z,_=b
            if abs(y-v)<max(t,z)*.4 and abs(t-z)<max(t,z)*.3 and 1.7<s/r<2.3 and abs((u-x)/r-2)<.2:
                pairs.append((abs(s/r-2)+abs((u-x)/r-2),a,b))
    if not pairs: raise ValueError('Could not locate the graphic scale')
    _,a,b=min(pairs,key=lambda p:p[0]);x,y,r,t,_=a;u,v,s,z,_=b
    # Start with the reference positions of a 32-unit bar. The numeric
    # division labels below supply the actual units after the geometric fit.
    ppf=((u+(s-1)/2)-(x+(r-1)/2))/10
    # Include the far end of the 32 ft bar to reduce subpixel quantization
    # error. The black-interval centroids are at 2 and 12 ft.
    pro=ink[int(y+max(3,t*.2)):int(y+t-max(3,t*.2))].mean(axis=0)
    centers=[]
    for lo,hi in [(x-3,x+ppf*4+4),(x+ppf*8-4,x+ppf*16+4)]:
        ix=np.arange(max(0,round(lo)),min(w,round(hi)))
        weight=np.maximum(pro[ix]-.15,0)
        centers.append(float(np.average(ix,weights=weight)))
    ix=np.arange(max(0,round(x+ppf*31)),min(w,round(x+ppf*33)))
    j=ix[np.argmax(pro[ix])]
    if pro[j]>.25:
        ix=np.arange(max(0,j-3),min(w,j+4));weight=np.maximum(pro[ix]-.02,0)
        centers.append(float(np.average(ix,weights=weight)))
        refined=float(np.polyfit([2,12,32],centers,1)[0])
        if abs(refined/ppf-1)<.02:ppf=refined
    from scale_ocr import endpoint_units
    units,divisions=endpoint_units(ink,float(x),float(y+t),float(t),float(ppf))
    ppf*=32/units
    result=(float(ppf),[int(x),int(y),int(u+s),int(y+t)])
    return result+({"end_feet":units,"read_divisions":divisions},) if return_reading else result

def find_plan(ink,ppf):
    h,w=ink.shape
    b=np.uint8(ink>.43)
    b[:int(.12*h)]=0;b[int(.9*h):]=0;b[:,int(.65*w):]=0
    horizontal=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,max(100,int(15*ppf))),np.uint8))
    st,_=components(horizontal)
    ss=[]
    for s in st:
        if s[2]<=w*.27:continue
        x,y,length,thickness,_=s
        # Dimension lines can be as dark as walls on a photocopy. An exterior
        # wall has another face eight inches away; a dimension line does not.
        lo=max(0,int(y-ppf));hi=min(h,int(y+ppf+thickness+2))
        pro=ink[lo:hi,int(x+ppf):int(x+length-ppf)].mean(axis=1)
        a0=max(0,y-lo-2);a1=min(len(pro),y-lo+max(5,int(ppf*.2)))
        peak=int(np.argmax(pro[a0:a1])+a0)
        offset=ppf*2/3
        support=0.
        for direction in [-1,1]:
            k=peak+direction*offset
            j0=max(0,round(k-ppf*.12));j1=min(len(pro),round(k+ppf*.12)+1)
            if j0<j1:support=max(support,float(pro[j0:j1].max()))
        if support>max(.25,float(pro[peak])*.45):ss.append(s)
    ss.sort(key=lambda s:s[1])
    pairs=[]
    for a in ss:
        for b in ss:
            if b[1]-a[1]<h*.25: continue
            overlap=min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])
            if overlap/min(a[2],b[2])<.75:continue
            if abs(a[0]-b[0])>ppf*50 or abs(a[0]+a[2]-b[0]-b[2])>ppf*50:continue
            area=(max(a[0]+a[2],b[0]+b[2])-min(a[0],b[0]))*(b[1]-a[1])
            pairs.append((area,a,b))
    if not pairs:raise ValueError('Could not locate the plan perimeter')
    _,a,b=max(pairs,key=lambda x:x[0])
    x0=min(a[0],b[0]);x1=max(a[0]+a[2],b[0]+b[2])-1
    y0=a[1];y1=b[1]+b[3]-1
    # Estimate the outer face at each edge using its long straight stroke.
    for axis,side in [(0,0),(0,1),(1,0),(1,1)]:
        if axis==0:
            c=x0 if side==0 else x1
            lo=max(0,int(c-ppf));hi=min(w,int(c+ppf+1))
            pro=np.mean(ink[int(y0+ppf):int(y1-ppf),lo:hi],axis=0)
        else:
            c=y0 if side==0 else y1
            lo=max(0,int(c-ppf));hi=min(h,int(c+ppf+1))
            pro=np.mean(ink[lo:hi,int(x0+ppf):int(x1-ppf)],axis=1)
        peaks=np.flatnonzero(pro>max(.28,.6*pro.max()))
        groups=np.split(peaks,np.flatnonzero(np.diff(peaks)>1)+1)
        groups=[g for g in groups if len(g)]
        if not groups:continue
        # The outermost strong edge near the initial wall estimate.
        g=groups[0] if side==0 else groups[-1]
        center=np.average(g+lo,weights=pro[g])
        if axis==0:
            if side==0:x0=center
            else:x1=center
        else:
            if side==0:y0=center
            else:y1=center
    return [float(x0),float(y0),float(x1),float(y1)]

def canonical(ink,box,ppf,res=16,pad=5):
    x0,y0,x1,y1=box
    w=int(round((x1-x0)/ppf*res))+1;h=int(round((y1-y0)/ppf*res))+1
    dst=(w+2*pad*res,h+2*pad*res)
    mat=np.array([[res/ppf,0,pad*res-x0*res/ppf],[0,res/ppf,pad*res-y0*res/ppf]],float)
    a=cv2.warpAffine(ink,mat,dst,flags=cv2.INTER_AREA,borderValue=0)
    return a,(pad*res,pad*res,pad*res+(x1-x0)/ppf*res,pad*res+(y1-y0)/ppf*res)

def init(path,debug=None):
    a=load_scan(path)
    ink=normalize(a)
    del a
    ink,angle=deskew(ink)
    last_error=None
    for quarter_turn in [0,2,1,3]:
        work=np.rot90(ink,quarter_turn).copy() if quarter_turn else ink
        try:
            ppf,scale,reading=scale_bar(work,return_reading=True)
            box=find_plan(work,ppf)
            break
        except ValueError as error:
            last_error=error
    else:
        raise ValueError('Cannot locate an upright scaled plan: '+str(last_error))
    ink=work
    angle+=quarter_turn*90
    plan,rect=canonical(ink,box,ppf)
    if debug:
        Path(debug).mkdir(exist_ok=True,parents=True)
        cv2.imwrite(str(Path(debug)/'plan.png'),np.uint8(255*(1-plan)))
        cv2.imwrite(str(Path(debug)/'sheet.png'),cv2.resize(np.uint8(255*(1-ink)),None,fx=.3,fy=.3))
        np.save(str(Path(debug)/'ink.npy'),ink);np.save(str(Path(debug)/'plan.npy'),plan)
        json.dump(dict(ppf=ppf,angle=angle,box=box,rect=rect,scale=scale,scale_reading=reading),open(str(Path(debug)/'meta.json'),'w'),indent=2)
    return ink,ppf,box,plan,rect

