"""Swing, paired, rolling and sliding door symbol recognition along recovered walls."""
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from floors import paper_model


def merged_door_walls(walls,scale):
    groups=[]
    for t in sorted(walls,key=lambda t:(t.axis,t.p,t.a)):
        found=None
        for group in groups:
            u=group[0]
            if u.axis==t.axis and u.kind==t.kind and abs(u.p-t.p)<scale*.12 and min(abs(t.a-max(v.b for v in group)),abs(t.b-min(v.a for v in group)))<scale*.3:
                found=group;break
        if found is None:groups.append([t])
        else:found.append(t)
    from geometry import Wall
    return [Wall(q[0].axis,float(np.average([t.p for t in q],weights=[t.b-t.a for t in q])),min(t.a for t in q),max(t.b for t in q),float(np.mean([t.width for t in q])),q[0].kind) for q in groups]


def void_darkness(g,paper,wall,start,end):
    p=round(wall.p);half=max(1,round(wall.width/2-2.8));a=round(min(start,end));b=round(max(start,end))
    if b-a<3:return 1.
    if wall.axis==0:
        body=g[p-half:p+half+1,a:b]
        white=paper((a+b)/2,p)
    else:
        body=g[a:b,p-half:p+half+1]
        white=paper(p,(a+b)/2)
    return float(white-np.mean(body))


def swing_doors(g,ink,walls,scale,contrast,paper=None):
    if paper is None:paper=paper_model(g)
    ws=merged_door_walls(walls,scale)
    radius=max(1,round(scale*.075));dil=cv2.dilate(ink,np.ones((2*radius+1,2*radius+1),np.uint8))
    found=[];candidates=[]
    rr=scale*3
    theta=np.linspace(np.pi*.065,np.pi*.435,50)
    for wi,t in enumerate(ws):
        arr=dil if t.axis==0 else dil.T
        h,w=arr.shape
        xx=np.arange(max(2,round(t.a)),min(w-2,round(t.b)+1))
        if len(xx)<scale*2:continue
        for normal in (-1,1):
            cy=t.p+normal*t.width/2
            leafrows=np.round(cy+normal*np.linspace(scale*.15,rr-scale*.1,36)).astype(int)
            if min(leafrows)<0 or max(leafrows)>=h:continue
            leafscore=np.mean(arr[leafrows[:,None],xx[None,:]],axis=0)/contrast
            for along in (-1,1):
                rows=np.round(cy+normal*rr*np.sin(theta)).astype(int)
                off=np.round(along*rr*np.cos(theta)).astype(int)
                if min(rows)<0 or max(rows)>=h:continue
                cols=xx[None,:]+off[:,None]
                valid=(cols>=0)&(cols<w)
                cols=cols.clip(0,w-1)
                arcvalues=arr[rows[:,None],cols]*valid/contrast
                arc=np.mean(arcvalues,axis=0)
                coverage=np.mean(arcvalues>.28,axis=0)
                arc20=np.percentile(arcvalues,20,axis=0)
                quality=np.minimum(arc,leafscore)
                peaks,_=find_peaks(quality,height=.28,distance=scale*.65,prominence=.06)
                for ix in peaks:
                    s=float(xx[ix]);opening=(s,s+along*rr)
                    if min(opening)<t.a-scale*.6 or max(opening)>t.b+scale*.6:continue
                    void=void_darkness(g,paper,t,s+along*scale*.25,s+along*scale*2.75)/contrast
                    q=float(quality[ix]);ar=float(arc[ix]);le=float(leafscore[ix])
                    olo,ohi=sorted(opening)
                    neighbor=max(void_darkness(g,paper,t,max(t.a,olo-scale*1.5),olo-scale*.25),
                                 void_darkness(g,paper,t,ohi+scale*.25,min(t.b,ohi+scale*1.5)))/contrast
                    # A dense floor hatch can cover every sample of an arc.
                    # A door has an actual curved ridge, stronger than the two
                    # concentric curves immediately beside its swing stroke.
                    adjacent=[]
                    for radius_ft in (2.65,3.35):
                        ay=np.round(cy+normal*scale*radius_ft*np.sin(theta)).astype(int).clip(0,h-1)
                        ax=np.round(s+along*scale*radius_ft*np.cos(theta)).astype(int).clip(0,w-1)
                        adjacent.append(arr[ay,ax]/contrast)
                    ridge_delta=arcvalues[:,ix]-(adjacent[0]+adjacent[1])/2
                    ridge=float(np.mean(ridge_delta));ridge_fraction=float(np.mean(ridge_delta>.12))
                    cand=dict(wall=wi,axis=t.axis,p=t.p,hinge=s,along=along,normal=normal,quality=q,arc=ar,leaf=le,void=void,opening=opening,coverage=float(coverage[ix]),arc20=float(arc20[ix]),neighbor=neighbor,ridge=ridge,ridge_fraction=ridge_fraction)
                    candidates.append(cand)
                    if void>.33 or q<.38 or coverage[ix]<.92 or ridge<.18 or ridge_fraction<.55:continue
                    if neighbor-void<.06 and not (ridge>.30 and coverage[ix]>.96 and le>.40):continue
                    found.append(cand)
    # A hinge may be found on both adjoining fragments of one centerline.
    out=[]
    for d in sorted(found,key=lambda d:-d['quality']):
        if any(d['axis']==e['axis'] and abs(d['p']-e['p'])<scale*.5 and abs(d['hinge']-e['hinge'])<scale*.6 for e in out):continue
        out.append(d)
    pairs=[];singles=[];used=set()
    for i,d in enumerate(out):
        if i in used:continue
        match=None
        for j,e in enumerate(out):
            if i==j or j in used or d['axis']!=e['axis'] or d['normal']!=e['normal']:continue
            if abs(d['p']-e['p'])>scale*.3 or d['along']==e['along']:continue
            gap=e['hinge']-d['hinge']
            if abs(abs(gap)/scale-6)<.3 and np.sign(gap)==d['along']:
                match=j;break
        if match is not None:
            pairs.append((d,out[match]));used.add(i);used.add(match)
        else:singles.append(d);used.add(i)
    return singles,pairs,candidates,ws


def sliding_doors(g,ink,walls,scale,contrast):
    ws=merged_door_walls(walls,scale)
    radius=max(1,round(scale*.055));dil=cv2.dilate(ink,np.ones((2*radius+1,2*radius+1),np.uint8))
    found=[];candidates=[]
    # Two outward arrowheads and two offset, overlapping sliding panels.
    for wi,t in enumerate(ws):
        arr=dil if t.axis==0 else dil.T
        gray=g if t.axis==0 else g.T
        h,w=arr.shape
        xx=np.arange(max(round(3.3*scale),round(t.a+scale*3)),min(w-round(3.3*scale),round(t.b-scale*3))+1)
        if not len(xx):continue
        for normal in (-1,1):
            cy=t.p+normal*scale*.75
            samples=[]
            for side in (-1,1):
                for up in (-1,1):
                    tt=np.linspace(.02,.95,12)
                    cols=xx[None,:]+np.round(side*(3.25-.5*tt)*scale).astype(int)[:,None]
                    rows=np.round(cy+up*.25*tt*scale).astype(int)
                    if rows.min()<0 or rows.max()>=h:continue
                    samples.append(arr[rows[:,None],cols]/contrast)
            if len(samples)!=4:continue
            samples=np.concatenate(samples,axis=0)
            arrow=np.mean(samples,axis=0);coverage=np.mean(samples>.24,axis=0)
            peaks,_=find_peaks(arrow,height=.3,distance=scale*3,prominence=.1)
            for ix in peaks:
                c=float(xx[ix]);p=round(t.p)
                # Staggered leaves: left is on one track, right is on the other.
                left=np.arange(round(c-2.7*scale),round(c-.7*scale))
                right=np.arange(round(c+.7*scale),round(c+2.7*scale))
                off=np.unique(np.round(np.linspace(.05,.13,3)*scale).astype(int))
                glt=np.mean(gray[(p-off)[:,None],left]);glb=np.mean(gray[(p+off)[:,None],left])
                grt=np.mean(gray[(p-off)[:,None],right]);grb=np.mean(gray[(p+off)[:,None],right])
                stagger=abs((grt-glt)+(glb-grb))/(2*contrast)
                cand=dict(axis=t.axis,p=t.p,center=c,normal=normal,arrow=float(arrow[ix]),coverage=float(coverage[ix]),stagger=float(stagger),wall=wi)
                candidates.append(cand)
                if stagger<.35:continue
                if not (arrow[ix]>=.34 and coverage[ix]>=.80) and not (stagger>.65 and arrow[ix]>.25 and coverage[ix]>.60):continue
                found.append(cand)
    out=[]
    for d in sorted(found,key=lambda q:-(q['arrow']+q['stagger'])):
        if any(d['axis']==e['axis'] and abs(d['p']-e['p'])<scale*.5 and abs(d['center']-e['center'])<scale*2 for e in out):continue
        out.append(d)
    return out,candidates


def coiling_doors(g,ink,walls,scale,contrast,paper=None):
    if paper is None:paper=paper_model(g)
    ws=merged_door_walls(walls,scale)
    found=[];profiles=[]
    for wi,t in enumerate(ws):
        if t.width<scale*.52 or t.b-t.a<scale*14:continue
        gray=g if t.axis==0 else g.T
        p=round(t.p);a=round(t.a);b=round(t.b)
        inner=max(3,round(t.width/2-3))
        off=np.r_[np.arange(-inner,-2),np.arange(3,inner+1)]
        if len(off)<2:continue
        x=np.arange(a,b+1)
        white=paper(x,p) if t.axis==0 else paper(p,x)
        profile=(white-np.mean(gray[(p+off)[:,None],x],axis=0))/contrast
        sm=uniform_filter1d(profile,size=max(3,round(scale*.7)),mode='nearest')
        baseline=np.percentile(sm,65)
        if baseline<.04:continue
        binary=sm<baseline*.53
        # Contiguous low-poche openings, with small scan defects closed.
        binary=cv2.morphologyEx(binary.astype(np.uint8)[None,:],cv2.MORPH_CLOSE,np.ones((1,max(1,round(scale*.25))),np.uint8))[0]
        transitions=np.diff(np.r_[0,binary,0].astype(int));starts=np.where(transitions==1)[0];ends=np.where(transitions==-1)[0]
        for lo,hi in zip(starts,ends):
            width=(hi-lo)/scale
            profiles.append(dict(wall=wi,axis=t.axis,p=t.p,a=a+lo,b=a+hi,width=width,baseline=baseline))
            if not (6.5<width<30):continue
            center=a+(lo+hi)/2
            found.append(dict(axis=t.axis,p=t.p,center=center,width=width,wall=wi))
    return found,profiles
