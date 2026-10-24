"""Sheet-specific finish texture recognition and wall-face floor measurement."""
import numpy as np
import cv2
from scipy.ndimage import find_objects
from geometry import wall_mask

CODES=('LVT-1','PC-1','SC-1','EP-1','VCT-1','CT-1','CPT-1','WOM-1')


from vision import paper_model


def spectral_bins(n):
    yy,xx=np.meshgrid(np.fft.fftfreq(n),np.fft.fftfreq(n),indexing='ij')
    rad=np.sqrt(xx**2+yy**2)
    ang=(np.arctan2(yy,xx)+np.pi/16)%np.pi
    angle=np.floor(ang/np.pi*8).astype(int)
    bands=np.digitize(rad,[.015,.03,.045,.065,.095,.14,.21,.32,.5])-1
    ix=bands*8+angle
    good=(bands>=0)&(bands<8)
    return ix,good


_BINS={}
def signature(patches,contrast,whites):
    n=patches[0].shape[0]
    if n not in _BINS:_BINS[n]=spectral_bins(n)
    ix,good=_BINS[n];window=np.outer(np.hanning(n),np.hanning(n))
    spectra=[];means=[];stds=[];features=[]
    for p,white in zip(patches,whites):
        d=(float(white)-p.astype(np.float32))/contrast
        means.append(np.mean(d));stds.append(np.std(d))
        f=np.fft.fft2((d-np.mean(d))*window)
        power=(np.abs(f)/n**2)**2
        v=np.bincount(ix[good].ravel(),weights=power[good].ravel(),minlength=64)
        spectra.append(v)
    ps=np.median(spectra,axis=0)
    return np.r_[np.median(means),np.median(stds),np.sqrt(ps)*5]


def get_patches(g,mask,n=64,max_samples=100,mask_origin=(0,0)):
    # All samples lie wholly within the finish region.
    padded=cv2.copyMakeBorder(np.uint8(mask),1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    d=cv2.distanceTransform(padded,cv2.DIST_L2,5)[1:-1,1:-1]
    ox,oy=mask_origin
    sy=(-oy)%16;sx=(-ox)%16
    ys,xs=np.where(d[sy::16,sx::16]>n*.72)
    ys=ys*16+sy+oy;xs=xs*16+sx+ox
    if len(xs)==0:
        y,x=np.unravel_index(np.argmax(d),d.shape)
        ys=np.array([y+oy]);xs=np.array([x+ox])
    if len(xs)>max_samples:
        pick=np.linspace(0,len(xs)-1,max_samples).astype(int);xs=xs[pick];ys=ys[pick]
    patches=[];centers=[]
    h,w=g.shape
    for y,x in zip(ys,xs):
        if x<n//2 or y<n//2 or x+n//2>w or y+n//2>h:continue
        patches.append(g[y-n//2:y+n//2,x-n//2:x+n//2]);centers.append((x,y))
    return patches,np.asarray(centers)


def legend_signatures(g,boxes,contrast,paper,n=64):
    refs=[]
    for x,y,w,h in boxes:
        patches=[];whites=[]
        for yy in np.linspace(y+5+n//2,y+h-5-n//2,2).astype(int):
            for xx in np.linspace(x+5+n//2,x+w-5-n//2,5).astype(int):
                patches.append(g[yy-n//2:yy+n//2,xx-n//2:xx+n//2]);whites.append(paper(xx,yy))
        refs.append(signature(patches,contrast,whites))
    return np.asarray(refs)


def floor_regions(g,ink,walls,rect,scale,contrast,threshold=.59,morph=False,small=False,occlusions=None,barriers=None,tags=()):
    xl,yt,xr,yb=np.round(rect).astype(int)
    box=(slice(yt,yb+1),slice(xl,xr+1))
    wm=wall_mask(walls,g.shape)[box]
    # Medium weight boundary lines survive; most hatch lines do not.
    b=np.uint8(ink[box]>contrast*threshold)
    if morph:b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    b|=wm
    # Ensure that the exterior is sealed even where a single/pair/rolling door is drawn.
    b[0]=b[-1]=1;b[:,0]=b[:,-1]=1
    n,labels,stats,centers=cv2.connectedComponentsWithStats(1-b)
    eligible=(stats[:,4]>scale**2*(.09 if small else 25))&(np.min(stats[:,2:4],axis=1)>scale*(.18 if small else 4.1))
    eligible[0]=False
    lookup=np.where(eligible,np.arange(n),0).astype(np.int32)
    seeds=lookup[labels]
    if small and occlusions is not None:
        blocked=occlusions[box].copy()
        from finish_tags import tag_boxes
        for x,y,w,h in tag_boxes(ink,rect,contrast):blocked[y-yt:y-yt+h,x-xl:x-xl+w]=1
        radius=max(1,round(scale*1.35))
        blocked=cv2.dilate(blocked,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
        seeds[blocked!=0]=0
    if not np.any(eligible):raise ValueError('Could not recover floor finish regions')
    # Fill occluded floor within its wall-bounded room. A case against a stockroom
    # partition must inherit the sales-floor finish, never the finish across the wall.
    partition=wm.copy()
    if barriers is not None:partition|=barriers[box]
    nr,rooms,roomstats,_=cv2.connectedComponentsWithStats(1-partition)
    filled=np.zeros_like(labels,np.int32)
    next_label=int(labels.max())+1
    hints={}
    nearby=None if barriers is None else cv2.dilate(barriers[box],np.ones((7,7),np.uint8))
    for room in range(1,nr):
        x,y,w,h,area=roomstats[room]
        if area<scale**2*(.10 if barriers is not None else 1):continue
        sl=(slice(y,y+h),slice(x,x+w));inside=rooms[sl]==room
        if nearby is not None and np.sum(nearby[sl][inside])>scale*5:
            votes=[]
            for tag in tags:
                tx,ty=np.round(tag['center']).astype(int);tx-=xl+x;ty-=yt+y
                if 0<=tx<w and 0<=ty<h and inside[ty,tx]:votes.append(tag['code'])
            if votes and len(set(votes))==1:
                view=filled[sl];view[inside]=next_label;hints[next_label]=votes[0];next_label+=1
                continue
        local=np.where(inside,seeds[sl],0).astype(np.int32)
        ids=np.unique(local[local!=0])
        if not len(ids):
            if area<scale**2*(.10 if barriers is not None else 25):continue
            local[inside]=next_label
            view=seeds[sl];view[inside]=next_label
            ids=np.array([next_label]);next_label+=1
        _,nearest=cv2.distanceTransformWithLabels(np.uint8(local==0),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_CCOMP)
        mapping=np.zeros(int(nearest.max())+1,np.int32)
        positions=np.flatnonzero(local)
        mapping[nearest.ravel()[positions]]=local.ravel()[positions]
        view=filled[sl]
        view[inside]=mapping[nearest[inside]]
    if barriers is not None:
        # Give each side half of the boundary stroke's pixels. Wall faces stay
        # excluded, while a finish boundary itself has no deductible area.
        missing=(filled==0)&(wm==0)
        _,nearest=cv2.distanceTransformWithLabels(np.uint8(filled==0),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        lookup=np.zeros(int(nearest.max())+1,np.int32)
        positions=np.flatnonzero(filled)
        lookup[nearest.ravel()[positions]]=filled.ravel()[positions]
        filled[missing]=lookup[nearest[missing]]
    return filled,seeds,stats,(xl,yt),wm,hints


def measure_floors(g,ink,walls,rect,scale,contrast,boxes,_threshold=.59,_morph=False,_search=True,occlusions=None,_small=False,_barriers=None):
    paper=paper_model(g)
    refs=legend_signatures(g,boxes,contrast,paper)
    from finish_tags import recognize_tags
    tags=recognize_tags(g,ink,rect,contrast)
    labels,seeds,stats,(xl,yt),wm,hints=floor_regions(g,ink,walls,rect,scale,contrast,_threshold,_morph,_small,occlusions,_barriers,tags)
    h,w=labels.shape
    gc=g[yt:yt+h,xl:xl+w]
    # A large-joint concrete swatch contains both joint strokes and plain paper.
    # Include a plain-paper prototype so a region sampled between joints is valid.
    blank_patches=[];blank_whites=[]
    for xx in np.linspace(g.shape[1]*.18,g.shape[1]*.52,8).astype(int):
        yy=round(g.shape[0]*.10)
        blank_patches.append(g[yy-32:yy+32,xx-32:xx+32]);blank_whites.append(paper(xx,yy))
    blank=signature(blank_patches,contrast,blank_whites)
    assignments={};distances={};features={}
    objects=find_objects(labels)
    for i in np.unique(labels):
        if i==0:continue
        sl=objects[i-1]
        sample_mask=labels[sl]==i
        if _barriers is not None and occlusions is not None:
            visible=occlusions[yt:yt+h,xl:xl+w][sl]==0
            if np.any(sample_mask&visible):sample_mask&=visible
        patches,centers=get_patches(gc,sample_mask,max_samples=120,mask_origin=(sl[1].start,sl[0].start))
        if not len(patches):continue
        whites=paper(centers[:,0]+xl,centers[:,1]+yt)
        feat=signature(patches,contrast,whites)
        # Mean and spread retain the distinction between solid, sparse and white finishes.
        weights=np.ones(refs.shape[1]);weights[0]=.45;weights[1]=.6
        ds=np.linalg.norm((refs-feat)*weights,axis=1)
        if refs[1,0]<.15 and refs[1,1]<.18:
            ds[1]=min(ds[1],np.linalg.norm((blank-feat)*weights))
        k=int(np.argmin(ds));assignments[int(i)]=k;distances[int(i)]=ds;features[int(i)]=feat
    votes={}
    for tag in tags:
        x,y=tag['center'];xx=round(x)-xl;yy=round(y)-yt
        if 0<=yy<h and 0<=xx<w:
            region=int(labels[yy,xx])
            if region in assignments:votes.setdefault(region,[]).append(tag['code'])
    conflicts=sum(len(set(v))>1 for v in votes.values())
    for region,codes in votes.items():
        if len(set(codes))==1:assignments[region]=codes[0]
    assignments.update(hints)
    areas=np.bincount(labels.ravel())
    result={code:round(sum(areas[i] for i,k in assignments.items() if k==j)/scale**2,1) for j,code in enumerate(CODES)}
    fit_cost=float(sum(areas[i]*distances[i][k] for i,k in assignments.items())/max(1,sum(areas[i] for i in assignments)))
    info=dict(labels=labels,seeds=seeds,assignments=assignments,distances=distances,refs=refs,features=features,
              origin=(xl,yt),tags=tags,conflicts=conflicts,fit_cost=fit_cost,threshold=_threshold,morph=_morph,small=_small,geometry_hints=hints)
    if _search:
        best=(result,info)
        variants=[(.47,False,False,None),(.59,True,False,None),(.74,False,False,None)]
        from boundaries import rounded_boundaries,straight_boundaries
        barriers,boundary_info=rounded_boundaries(ink,rect,scale,contrast)
        barriers|=straight_boundaries(ink,rect,walls,scale,contrast,tags,occlusions,barriers)
        if np.any(barriers):variants.append((.59,False,False,barriers))
        for threshold,morph,small,barrier in variants:
            try:
                alternate,detail=measure_floors(g,ink,walls,rect,scale,contrast,boxes,threshold,morph,False,occlusions,small,barrier)
            except ValueError:
                continue
            # Code callouts are independent of hatch texture and line weight.
            def selection_key(detail):
                reliable_geometry=len(detail['geometry_hints'])>=3 and len(set(detail['geometry_hints'].values()))>=2
                return (detail['conflicts'],detail['fit_cost']+(-.015 if reliable_geometry else .006 if detail is not info else 0))
            candidate_key=selection_key(detail)
            current_key=selection_key(best[1])
            if candidate_key<current_key:best=(alternate,detail)
        return best
    return result,info
