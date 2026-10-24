"""Face-of-wall floor geometry and per-sheet material recognition."""
import cv2
import numpy as np
from scipy import ndimage
from vision import components, save_debug


def wall_mask(shape,walls):
    mask=np.zeros(shape,np.uint8)
    for w in walls:
        half=w['observed']/2
        if w['axis']=='h':
            x0,x1=int(np.ceil(w['lo'])),int(np.floor(w['hi']))
            y0,y1=int(np.ceil(w['pos']-half)),int(np.floor(w['pos']+half))
        else:
            x0,x1=int(np.ceil(w['pos']-half)),int(np.floor(w['pos']+half))
            y0,y1=int(np.ceil(w['lo'])),int(np.floor(w['hi']))
        mask[max(0,y0):y1+1,max(0,x0):x1+1]=1
    return mask


def floor_regions(a,p,walls,threshold,bounds):
    mask=wall_mask(a.shape,walls)
    b=(a>threshold).astype(np.uint8)
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    b|=mask
    cs,lab=components(b)
    keep=np.zeros(lab.max()+1,np.uint8)
    for i,x,y,w,h,ar,cx,cy in cs:
        if w>p*15 and h>p*5 or h>p*15 and w>p*5:keep[i]=1
    b=keep[lab]
    x0,y0,x1,y1=bounds
    b[:round(y0),:]=1;b[round(y1):,:]=1;b[:,:round(x0)]=1;b[:,round(x1):]=1
    cs,lab=components(1-b)
    # Door arcs and outline ink may divide tiny pockets. Use the surrounding
    # room to absorb those pockets, then give each boundary half its lineweight.
    big=np.zeros(lab.max()+1,np.uint8)
    for i,x,y,w,h,ar,cx,cy in cs:
        if ar > p*p*12:big[i]=1
    lab[~big[lab].astype(bool)]=0
    _,ix=ndimage.distance_transform_edt(lab==0,return_indices=True)
    labels=lab[tuple(ix)]
    labels[mask.astype(bool)]=0
    labels[:int(np.ceil(y0+p*.667)),:]=0;labels[int(np.ceil(y1-p*.667)):,:]=0
    labels[:,:int(np.ceil(x0+p*.667))]=0;labels[:,int(np.ceil(x1-p*.667)):]=0
    return labels,lab,mask


def texture_descriptor(patch):
    patch=cv2.resize(patch,(72,72),interpolation=cv2.INTER_AREA)
    patch=cv2.GaussianBlur(patch.astype(np.float32),(0,0),.6)
    mean=float(patch.mean());sd=float(patch.std())
    q=np.quantile(patch,[.1,.5,.9])
    desc=[mean,sd,*q]
    for shift in [1,2,4,8,12,18,24,36]:
        desc.append(np.sqrt(np.mean((patch[:,shift:]-patch[:,:-shift])**2)))
        desc.append(np.sqrt(np.mean((patch[shift:]-patch[:-shift])**2)))
        if shift<20:
            desc.append(np.sqrt(np.mean((patch[shift:,shift:]-patch[:-shift,:-shift])**2)))
            desc.append(np.sqrt(np.mean((patch[shift:,:-shift]-patch[:-shift,shift:])**2)))
    window=np.hanning(72).astype(np.float32)
    ft=np.fft.fftshift(np.fft.fft2((patch-mean)*window[:,None]*window[None,:]))
    power=(abs(ft)**2).astype(np.float32)
    fy,fx=np.mgrid[-36:36,-36:36];rr=np.hypot(fx,fy);angle=(np.arctan2(fy,fx)+np.pi)%(np.pi)
    # Frequency and orientation together distinguish dots, dashed planks,
    # diagonal hatches, cross hatches, grids, and dense walk-off mats.
    for rlo,rhi in [(1,3),(3,5),(5,8),(8,13),(13,22),(22,37)]:
        for ang in range(8):
            theta=ang*np.pi/8
            diff=abs((angle-theta+np.pi/2)%np.pi-np.pi/2)
            sel=(rr>=rlo)&(rr<rhi)&(diff<np.pi/16)
            desc.append(np.sqrt(power[sel].sum())/72**2*3)
    return np.array(desc,np.float32)


def legend_models(full,p,swatches):
    side=round(p*2.8)
    models=[]
    for k,(x,y,w,h) in enumerate(swatches):
        patch=full[y+5:y+h-5,x+5:x+w-5]
        hh,ww=patch.shape
        features=[]
        for yy in np.linspace(0,max(0,hh-side),4).astype(int):
            for xx in np.linspace(0,max(0,ww-side),12).astype(int):
                features.append(texture_descriptor(patch[yy:yy+side,xx:xx+side]))
        models.append(np.array(features))
    # A control-joint finish has much larger empty panels than its legend box.
    models[1]=np.vstack([models[1],texture_descriptor(np.zeros((side,side),np.float32)+.02)])
    return models


def classify_regions(a,p,labels,seed,models,debug=None):
    side=round(p*2.8);half=side//2
    areas=np.bincount(labels.ravel())
    mapping={};scores={}
    # Normalize feature groups to balance ink strength and texture direction.
    weights=np.ones(models[0].shape[1],np.float32)
    weights[0:5]=.7;weights[37:]=2
    for lab in np.unique(labels):
        if lab==0:continue
        ys,xs=np.where(seed==lab)
        if not len(xs):continue
        x0,x1=xs.min(),xs.max();y0,y1=ys.min(),ys.max()
        points=[]
        for y in range(y0+half,y1-half+1,max(half,1)):
            for x in range(x0+half,x1-half+1,max(half,1)):
                if np.mean(seed[y-half:y-half+side,x-half:x-half+side]==lab)>.97:points.append((x,y))
        if len(points)>160:
            points=[points[i] for i in np.linspace(0,len(points)-1,160).astype(int)]
        features=[]
        for x,y in points:features.append(texture_descriptor(a[y-half:y-half+side,x-half:x-half+side]))
        if not features:
            # Rooms narrower than a normal sample are represented by their
            # largest inscribed patch.
            sel=(seed[y0:y1+1,x0:x1+1]==lab)
            dist=ndimage.distance_transform_edt(sel)
            yy,xx=np.unravel_index(dist.argmax(),dist.shape);sz=max(8,round(dist[yy,xx]*1.3))
            x=x0+xx;y=y0+yy
            features=[texture_descriptor(a[max(0,y-sz):y+sz,max(0,x-sz):x+sz])]
        feats=np.array(features)
        distances=[]
        for model in models:
            d=np.sum(((feats[:,None,:]-model[None,:,:])*weights)**2,axis=2)
            distances.append(np.sqrt(d.min(1)))
        distances=np.array(distances).T
        # Use the majority of informative patches. White fixture interiors
        # form a minority and are not allowed to erase the floor underneath.
        best=distances.min(1)
        selected=np.argsort(best)[:max(3,round(len(best)*.4))]
        wins=distances[selected].argmin(1)
        counts=np.bincount(wins,minlength=8)
        choice=int(counts.argmax())
        if choice==1 and counts[1] < len(wins)*.8:
            other=counts.copy();other[1]=0
            if other.max()>len(wins)*.2:choice=int(other.argmax())
        mapping[int(lab)]=choice
        scores[int(lab)]=(counts.tolist(),np.round(np.median(distances,axis=0),3).tolist())
    if debug:
        colors=np.array([[180,210,230],[230,230,230],[180,230,180],[180,180,230],[230,200,160],[200,160,200],[160,200,230],[160,160,160]],np.uint8)
        lut=np.zeros((labels.max()+1,3),np.uint8)+255
        for lab,k in mapping.items():lut[lab]=colors[k]
        out=(lut[labels].astype(np.float32)*(1-a*.5)[:,:,None]).astype(np.uint8)
        cv2.imwrite(str(debug)+'-floor-class.png',out)
    return mapping,scores
