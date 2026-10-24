"""Recover wall centerlines and wall-face polygons from linework."""
from vision import Edge,line_edges,merge_edges
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class Wall:
    axis:int
    p:float
    a:float
    b:float
    width:float
    kind:str=''


def wall_candidates(ink,contrast,scale,relaxed=False):
    es=line_edges(ink,contrast*(.24 if relaxed else .39),max(30,round(scale*2.5)),gap=3,split=True)
    if relaxed:
        es=[e for e in es if e.strength>contrast*.50 or (e.thickness>=1.8 and e.strength>contrast*.28)]
    else:
        es=[e for e in es if e.strength>contrast*.67 or (e.thickness>=2 and e.strength>contrast*.50) or (e.thickness>=3 and e.strength>contrast*.40) or
            (scale*.30<e.thickness-3<scale*.95 and e.strength>contrast*.30)]
    thin=merge_edges([e for e in es if e.thickness<scale*.29],tolerance=max(1.5,scale*.075),maxgap=scale*16)
    thick=merge_edges([e for e in es if e.thickness>=scale*.29],tolerance=max(1.5,scale*.075),maxgap=scale*16)
    es=thin+thick
    # A dark solid poche can join its two faces into one connected strip.
    # Retain those strips as well as the separately visible face pairs.
    walls=[Wall(e.axis,e.p,e.a,e.b,max(1.,e.thickness-3.)) for e in es
           if scale*.30<e.thickness-3<scale*.95 and e.b-e.a>scale*3]
    for axis in (0,1):
        a=sorted([e for e in thin if e.axis==axis],key=lambda e:e.p)
        for i,e in enumerate(a):
            for f in a[i+1:]:
                d=f.p-e.p
                if d>scale*.95:break
                if d<scale*.30:continue
                if e.thickness+f.thickness<4.:continue
                if d>scale*.48 and e.thickness+f.thickness<5.:continue
                lo=max(e.a,f.a);hi=min(e.b,f.b)
                if hi-lo<scale*3:continue
                walls.append(Wall(axis,(e.p+f.p)/2,lo,hi,d))
    unique=[]
    for t in sorted(walls,key=lambda q:-(q.b-q.a)):
        duplicate=any(t.axis==u.axis and abs(t.p-u.p)<scale*.13 and abs(t.width-u.width)<scale*.15 and
                      t.a>=u.a-scale*.2 and t.b<=u.b+scale*.2 for u in unique)
        if not duplicate:unique.append(t)
    return unique,es


def recover_weak_walls(ink,contrast,scale,primary,exterior,rect):
    """A faint face pair must close between established perpendicular walls."""
    weak,_=wall_candidates(ink,contrast,scale,relaxed=True)
    out=list(primary)
    anchors=primary+exterior
    for t in interior_candidates(weak,rect,scale):
        if t.width<scale*.50:continue
        if any(t.axis==u.axis and abs(t.p-u.p)<scale*.18 and
               min(t.b,u.b)-max(t.a,u.a)>.75*min(t.b-t.a,u.b-u.a) for u in primary):continue
        ends=[]
        for end in (t.a,t.b):
            choices=[u.p for u in anchors if u.axis!=t.axis and abs(u.p-end)<scale*.9 and
                     u.a-scale*.6<t.p<u.b+scale*.6]
            if choices:ends.append(min(choices,key=lambda p:abs(p-end)))
        if len(ends)==2:
            out.append(Wall(t.axis,t.p,ends[0],ends[1],t.width))
    return out


def find_exterior(walls,shape,scale,edges=None):
    h,w=shape
    vs=[t for t in walls if t.axis==1 and t.b-t.a>h*.30 and t.p<w*.965 and t.p>w*.035]
    vs=sorted(vs,key=lambda t:t.p)
    if len(vs)<2:
        if edges is not None:return exterior_from_faces(edges,shape,scale)
        raise ValueError('Could not identify both exterior walls')
    # Choose the broadest pair whose vertical ranges coincide.
    pairs=[]
    for i,v in enumerate(vs):
        for u in vs[i+1:]:
            overlap=min(v.b,u.b)-max(v.a,u.a)
            if u.p-v.p>w*.25 and overlap>h*.3:
                pairs.append((overlap*(u.p-v.p),v,u))
    if not pairs:
        if edges is not None:return exterior_from_faces(edges,shape,scale)
        raise ValueError('Could not identify the exterior footprint')
    _,left,right=max(pairs,key=lambda t:t[0])
    hs=[t for t in walls if t.axis==0 and t.b-t.a>(right.p-left.p)*.8 and
        t.a<left.p+scale*1.5 and t.b>right.p-scale*1.5]
    hs=sorted(hs,key=lambda t:t.p)
    if len(hs)<2:
        if edges is not None:return exterior_from_faces(edges,shape,scale)
        raise ValueError('Could not identify the exterior perimeter')
    top,bottom=hs[0],hs[-1]
    if edges is not None and (abs(top.p-np.median([left.a,right.a]))>scale*1.5 or
                              abs(bottom.p-np.median([left.b,right.b]))>scale*1.5):
        return exterior_from_faces(edges,shape,scale)
    rect=(left.p,top.p,right.p,bottom.p)
    ext=[Wall(0,top.p,left.p,right.p,top.width,'EXT'),Wall(0,bottom.p,left.p,right.p,bottom.width,'EXT'),
         Wall(1,left.p,top.p,bottom.p,left.width,'EXT'),Wall(1,right.p,top.p,bottom.p,right.width,'EXT')]
    return rect,ext


def interior_candidates(walls,rect,scale):
    xl,yt,xr,yb=rect
    out=[]
    for t in walls:
        if t.axis==0:
            if t.p<yt+scale*.9 or t.p>yb-scale*.9 or t.b<xl or t.a>xr:continue
            a=max(xl,t.a);b=min(xr,t.b)
        else:
            if t.p<xl+scale*.9 or t.p>xr-scale*.9 or t.b<yt or t.a>yb:continue
            a=max(yt,t.a);b=min(yb,t.b)
        if b-a>scale*3:
            out.append(Wall(t.axis,t.p,a,b,t.width))
    connected=[]
    for t in out:
        if t.width<scale*.50:
            connected.append(t);continue
        joins=0
        for end in (t.a,t.b):
            if t.axis==0 and min(abs(end-xl),abs(end-xr))<scale*1.2:joins+=1;continue
            if t.axis==1 and min(abs(end-yt),abs(end-yb))<scale*1.2:joins+=1;continue
            if any(u.axis!=t.axis and abs(u.p-end)<scale*1.2 and u.a-scale*.7<t.p<u.b+scale*.7 for u in out):joins+=1
        if joins:connected.append(t)
    return connected


def strip_feature(g,axis,p,a,b,width,contrast):
    """Poche appearance inside a wall, normalized by local paper tone."""
    a=round(a);b=round(b);p=round(p)
    if b-a<10:return None
    pad=max(30,round(width*3));inner=max(1,int(width*.25))
    if axis==0:
        strip=g[max(0,p-pad):p+pad+1,a:b]
        body=g[p-inner:p+inner+1,a:b]
    else:
        strip=g[a:b,max(0,p-pad):p+pad+1].T
        body=g[a:b,p-inner:p+inner+1].T
    if body.size==0:return None
    white=250.
    # Statistics along the wall, avoiding notes and intersections with robust aggregation.
    v=(white-body.astype(float))/contrast
    # Door openings have no two dark wall faces; omit them from statistics.
    pp=strip.shape[0]//2; half=width/2
    f1=strip[max(0,round(pp-half)-2):round(pp-half)+3]
    f2=strip[max(0,round(pp+half)-2):round(pp+half)+3]
    valid=np.ones(v.shape[1],bool)
    if f1.size and f2.size:
        test=((white-np.min(f1,axis=0))>contrast*.67)&((white-np.min(f2,axis=0))>contrast*.67)
        if np.sum(test)>max(12,.15*len(test)):valid=test
    m=np.mean(v,axis=0)
    keep=valid&(m>=np.percentile(m[valid],45))&(m<=np.percentile(m[valid],96))
    if np.sum(keep)<10:keep=valid
    vv=v[:,keep]
    # Compute derivatives before selecting columns to preserve hatch orientation.
    dx=cv2.Sobel(v.astype(np.float32),cv2.CV_32F,1,0,ksize=3)/8
    dy=cv2.Sobel(v.astype(np.float32),cv2.CV_32F,0,1,ksize=3)/8
    dx=dx[1:-1,keep];dy=dy[1:-1,keep]
    ex=np.mean(dx*dx);ey=np.mean(dy*dy)
    corr=float(np.mean(dx*dy))/max(1e-7,np.sqrt(ex*ey))
    x2=dx*dx;y2=dy*dy
    energy=x2+y2
    # Energy weighting keeps a note or a stray wall-face pixel from outweighing
    # hundreds of faint hatch strokes in the fourth angular moment.
    orientation4=float(np.sum((x2*x2+y2*y2-6*x2*y2)/np.maximum(energy,1e-8))/max(1e-7,float(np.sum(energy))))
    return np.array([np.mean(vv),np.std(vv),np.percentile(vv,25),np.percentile(vv,75),
                     np.mean(np.abs(dx)),np.mean(np.abs(dy)),corr,orientation4])



def finish_walls(g,ink,candidates,ext,scale,contrast,swatches):
    allwalls=ext+candidates
    # Snap every junction to the centerline of the wall it meets.
    snapped=[]
    for t in candidates:
        ends=[t.a,t.b]
        for j,end in enumerate(ends):
            choices=[]
            for u in allwalls:
                if u.axis==t.axis:continue
                if abs(u.p-end)<scale*1.1 and u.a-scale*.8<=t.p<=u.b+scale*.8:
                    choices.append(u.p)
            if choices:ends[j]=min(choices,key=lambda q:abs(q-end))
        snapped.append(Wall(t.axis,t.p,ends[0],ends[1],t.width))
    refs=[]
    for x,y,w,h in swatches:
        refs.append(strip_feature(g,0,y+(h-1)/2,x+8,x+w-8,h-3,contrast))
    out=[]
    expected_width={'P1':.40625,'P2':.604167,'P3':2/3}
    def appearance_distance(feature,reference,axis=0):
        if axis:
            reference=reference.copy()
            reference[4],reference[5]=reference[5],reference[4]
        patterned=max(reference[4:6])>.08 or min(reference[4:6])>.035
        weights=np.array([1.5,1.5,.6,.6,3.,3.,1.,.8]) if patterned else np.array([2.,2.,1.,1.,4.,4.,0.,0.])
        return float(np.linalg.norm((feature-reference)*weights))
    prototypes={'P2':refs[2],'P3':refs[1]}
    for t in snapped:
        whole=strip_feature(g,t.axis,t.p,t.a+scale*.25,t.b-scale*.25,t.width,contrast)
        prior={kind:3*max(0.,abs(t.width/scale-width)-.06) for kind,width in expected_width.items()}
        if t.width<scale*.50:
            fallback='P1'
            joins=any(u.axis!=t.axis and min(abs(u.p-t.a),abs(u.p-t.b))<scale*.8 and
                      u.a-scale*.7<t.p<u.b+scale*.7 for u in allwalls)
            if not joins and whole is not None and appearance_distance(whole,refs[3],t.axis)>.85:continue
        elif whole is not None:
            scores={kind:appearance_distance(whole,reference,t.axis)+prior[kind] for kind,reference in prototypes.items()}
            fallback=min(scores,key=scores.get)
            if min(scores.values())>.70 and t.b-t.a>scale*6:continue
        else:fallback=min(expected_width,key=lambda kind:abs(expected_width[kind]-t.width/scale))
        cuts=[t.a,t.b]
        for u in allwalls:
            if u.axis!=t.axis and t.a+scale*.25<u.p<t.b-scale*.25 and u.a-scale*.7<=t.p<=u.b+scale*.7:
                cuts.append(u.p)
        cuts=sorted(cuts)
        for a,b in zip(cuts[:-1],cuts[1:]):
            if b-a<scale*.4:continue
            # A corridor between two collinear partitions is not a continuation.
            aa=round(a+scale*.4);bb=round(b-scale*.4)
            if bb>aa:
                p1=round(t.p-t.width/2);p2=round(t.p+t.width/2)
                if t.axis==0:
                    s1=np.max(ink[p1-2:p1+3,aa:bb],axis=0);s2=np.max(ink[p2-2:p2+3,aa:bb],axis=0)
                else:
                    s1=np.max(ink[aa:bb,p1-2:p1+3],axis=1);s2=np.max(ink[aa:bb,p2-2:p2+3],axis=1)
                cover=np.mean((s1>contrast*.56)&(s2>contrast*.56))
                if cover<.17:continue
            feature=strip_feature(g,t.axis,t.p,a+scale*.25,b-scale*.25,t.width,contrast)
            if t.width<scale*.50:kind='P1'
            elif feature is None:kind=fallback
            else:
                scores={kind:appearance_distance(feature,reference,t.axis)+prior[kind] for kind,reference in prototypes.items()}
                kind=min(scores,key=scores.get)
                if scores[kind]>.70:kind=fallback
            out.append(Wall(t.axis,t.p,a,b,t.width,kind))
    return ext+out,refs


def wall_lengths(walls,scale):
    return {kind:round(sum(t.b-t.a for t in walls if t.kind==kind)/scale,2) for kind in ('EXT','P1','P2','P3')}


def wall_mask(walls,shape):
    mask=np.zeros(shape,np.uint8)
    for t in walls:
        a=round(t.a);b=round(t.b);lo=round(t.p-t.width/2);hi=round(t.p+t.width/2)
        if t.axis==0:mask[lo:hi+1,a:b+1]=1
        else:mask[a:b+1,lo:hi+1]=1
    return mask



def exterior_from_faces(edges,shape,scale):
    """Use the CMU's visible outside faces when a dark scan hides an inside face."""
    h,w=shape
    vs=sorted([e for e in edges if e.axis==1 and e.b-e.a>h*.30 and w*.035<e.p<w*.965 and e.thickness>=3.4],key=lambda e:e.p)
    hs=[e for e in edges if e.axis==0 and e.b-e.a>w*.25 and e.thickness>=3.4]
    possibilities=[]
    for i,left in enumerate(vs):
        for right in vs[i+1:]:
            width=right.p-left.p
            if not w*.25<width<w*.90:continue
            for top in hs:
                if top.a>left.p+scale or top.b<right.p-scale:continue
                if abs(top.p-left.a)>scale or abs(top.p-right.a)>scale:continue
                for bottom in hs:
                    if bottom.p-top.p<h*.25:continue
                    if bottom.a>left.p+scale or bottom.b<right.p-scale:continue
                    if abs(bottom.p-left.b)>scale or abs(bottom.p-right.b)>scale:continue
                    possibilities.append((width*(bottom.p-top.p),left.p,top.p,right.p,bottom.p))
    if not possibilities:raise ValueError('Could not recover the exterior wall faces')
    _,xl,yt,xr,yb=max(possibilities)
    width=scale*2/3;inset=width/2
    xl+=inset;xr-=inset;yt+=inset;yb-=inset
    ext=[Wall(0,yt,xl,xr,width,'EXT'),Wall(0,yb,xl,xr,width,'EXT'),
         Wall(1,xl,yt,yb,width,'EXT'),Wall(1,xr,yt,yb,width,'EXT')]
    return (xl,yt,xr,yb),ext
