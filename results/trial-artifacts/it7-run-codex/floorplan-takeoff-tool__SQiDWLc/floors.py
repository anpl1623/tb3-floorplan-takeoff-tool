"""Finish-region reconstruction and recognition from the sheet legend."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

CODES=['LVT-1','PC-1','SC-1','EP-1','VCT-1','CT-1','CPT-1','WOM-1']

def norm_glyph(patch):
    p=np.float32(patch)
    p=np.maximum(np.percentile(p,95)-p,0)
    if p.size==0:return np.zeros((22,72),np.float32)
    # A label outline is lighter than the text and is excluded before matching.
    mask=p>max(float(np.max(p))*.48,15)
    ys,xs=np.where(mask)
    if len(xs)<10:return np.zeros((22,72),np.float32)
    p=p[min(ys):max(ys)+1,min(xs):max(xs)+1]
    p=cv2.resize(p,(72,22),interpolation=cv2.INTER_AREA)
    p-=np.mean(p);p/=max(float(np.linalg.norm(p)),1e-6)
    return p

def recognize_tags(sheet,bounds):
    templates=np.load(Path(__file__).parent/'assets/finish_tags.npz')
    tm={k:norm_glyph(templates[k]) for k in templates.files}
    left,top,right,bottom=np.int32(bounds)
    x0=max(0,left-10);y0=max(0,top-10)
    gray=sheet.gray[y0:bottom+12,x0:right+12]
    ed=cv2.Canny(gray,20,50)
    cs,_=cv2.findContours(ed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in cs:
        x,y,w,h=cv2.boundingRect(c)
        if not (sheet.ppf*.9<w<sheet.ppf*4.5 and sheet.ppf*.7<h<sheet.ppf*1.5):continue
        area=cv2.contourArea(c)
        if area<.68*w*h:continue
        roi=gray[y+3:y+h-3,x+3:x+w-3]
        f=norm_glyph(roi)
        scores=sorted([(float(np.sum(f*v)),k) for k,v in tm.items()],reverse=True)
        if scores[0][0]>.65 and scores[0][0]-scores[1][0]>.035:
            candidates.append((scores[0][0],scores[0][1],x+x0+w/2,y+y0+h/2,w,h))
    accepted=[]
    for q in sorted(candidates,reverse=True):
        if not any(np.hypot(q[2]-r[2],q[3]-r[3])<sheet.ppf*.8 for r in accepted):accepted.append(q)
    return accepted

def finish_features(image,mask,black):
    """Phase-independent, contrast-normalized local texture descriptor."""
    p=np.float32(image)/max(black,1)
    mask=np.uint8(mask)
    mask=cv2.erode(mask,np.ones((11,11),np.uint8))>0
    if not np.any(mask):return np.zeros(28)
    # Suppress annotation and wall ink; the hatch strokes are lighter.
    bad=cv2.dilate(np.uint8(p>.55),np.ones((11,11),np.uint8))>0
    valid=mask&~bad&(p<.65)
    if np.sum(valid)<100:valid=mask
    means=[np.mean(np.clip(p,0,.65)[valid]),np.std(p[valid])]
    fs=[]
    for sig in [0.7,1.4,2.8]:
        b=cv2.GaussianBlur(p,(0,0),sig)
        gx=cv2.Sobel(b,cv2.CV_32F,1,0,ksize=3)/8
        gy=cv2.Sobel(b,cv2.CV_32F,0,1,ksize=3)/8
        fs.extend([np.mean(np.abs(gx)[valid]),np.mean(np.abs(gy)[valid]),np.mean(np.abs(gx+gy)[valid])/1.414,np.mean(np.abs(gx-gy)[valid])/1.414])
    # High-frequency dots and a uniform solid poche differ after smoothing.
    low=cv2.GaussianBlur(p,(0,0),4)
    fs.extend([np.mean(np.abs(p-low)[valid]),np.std(low[valid])])
    ratios=[]
    for j in range(3):
        v=np.array(fs[4*j:4*j+4]);ratios.extend(v/max(np.sum(v),1e-6))
    return np.r_[means,fs,ratios]

def interior_mask(shape,walls,bounds):
    left,top,right,bottom=bounds
    # Work at the native raster resolution. Pixel-center inclusion defines areas
    # without adding the stroke width to the wall thickness.
    yy,xx=np.ogrid[:shape[0],:shape[1]]
    extwidth=np.median([q.width for q in walls if q.kind=='EXT'])
    inside=(xx>=left+extwidth/2)&(xx<=right-extwidth/2)&(yy>=top+extwidth/2)&(yy<=bottom-extwidth/2)
    for q in walls:
        if q.kind=='EXT':continue
        if q.axis:
            band=(np.abs(xx-q.c)<=q.width/2)&(yy>=q.a)&(yy<=q.b)
        else:
            band=(np.abs(yy-q.c)<=q.width/2)&(xx>=q.a)&(xx<=q.b)
        inside&=~band
    return inside

def measure_floors(sheet,walls,bounds):
    left,top,right,bottom=bounds
    h=int(np.ceil(bottom+3));w=int(np.ceil(right+3))
    inside=interior_mask((h,w),walls,bounds)
    ink=sheet.ink[:h,:w]
    bw=np.uint8(ink>sheet.black*.52)
    bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    free=np.uint8(inside&~(bw>0))
    n,labels,stats,cents=cv2.connectedComponentsWithStats(free)
    minarea=sheet.ppf**2*18
    regions=[i for i in range(1,n) if stats[i,4]>minarea]
    seeds=np.zeros(labels.shape,np.int32)
    for i in regions:seeds[labels==i]=i
    # Only contours of finish regions are barriers. Furniture, columns, text and
    # door swing arcs are restored to the surrounding finish by nearest region.
    _,nearest=distance_transform_edt(seeds==0,return_indices=True)
    filled=seeds[tuple(nearest)];filled[~inside]=0
    del nearest,seeds
    # A broad paper estimate retains solid finish swatches and solid-filled rooms.
    paper=cv2.dilate(sheet.gray,cv2.getStructuringElement(cv2.MORPH_RECT,(801,801)))
    texture=paper.astype(np.float32)-sheet.gray.astype(np.float32)
    refs={}
    for k,(x,y,ww,hh) in zip(CODES,sheet.swatches):
        patch=texture[y+7:y+hh-7,x+7:x+ww-7]
        refs[k]=finish_features(patch,np.ones(patch.shape,np.uint8),sheet.black)
    tags=recognize_tags(sheet,bounds)
    known={}
    for score,code,x,y,ww,hh in tags:
        i=int(filled[int(y),int(x)])
        if i and i not in known:known[i]=code
    # Textured unlabelled rooms can also be compared with larger, labelled areas
    # on the plan; this covers differences between a swatch's and plan's pitch.
    feats={}
    for i in regions:
        x,y,ww,hh,_=stats[i]
        feats[i]=finish_features(texture[y:y+hh,x:x+ww],labels[y:y+hh,x:x+ww]==i,sheet.black)
    planrefs={}
    for i,k in known.items():planrefs.setdefault(k,[]).append(feats[i])
    weights=np.r_[1.,.5,np.full(12,8.),1.,.1,np.full(12,1.5)]
    kinds={}
    for i in regions:
        if i in known:kinds[i]=known[i];continue
        f=feats[i]
        distances={k:min(np.linalg.norm((f-r)*weights) for r in [refs[k]]+planrefs.get(k,[])) for k in CODES}
        kinds[i]=min(distances,key=distances.get)
    totals={k:0. for k in CODES}
    sizes=np.bincount(filled.ravel(),minlength=n)
    for i,k in kinds.items():totals[k]+=float(sizes[i])/sheet.ppf**2
    if sheet.debug:
        colors=np.array([[235,220,160],[210,235,245],[170,200,170],[200,180,240],[235,190,150],[200,240,190],[210,170,190],[170,170,170]],np.uint8)
        lookup=np.full((n,3),255,np.uint8)
        for i,k in kinds.items():lookup[i]=colors[CODES.index(k)]
        canvas=lookup[filled]
        for i,k in kinds.items():
            x,y=cents[i];cv2.putText(canvas,f'{i} {k}',(int(x)-25,int(y)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
        for score,k,x,y,_,_ in tags:cv2.putText(canvas,f'{k} {score:.2f}',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        sheet.save('finishes',cv2.resize(canvas,(int(w*.35),int(h*.35))))
        import json
        with open(Path(sheet.debug)/'regions.json','w') as f:
            json.dump({'tags':tags,'regions':[{ 'id':i,'sf':sizes[i]/sheet.ppf**2,'code':kinds[i],'known':i in known,'features':feats[i].tolist()}for i in regions],'refs':{k:v.tolist()for k,v in refs.items()}},f,indent=2)
    return totals,filled,kinds
