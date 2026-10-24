"""Recognize the chain's dimensioned CAD fixture symbols."""
import cv2
import numpy as np
from scipy.signal import find_peaks
from vision import line_edges,merge_edges


def gondolas(g,ink,rect,scale,contrast):
    xl,yt,xr,yb=rect
    es=line_edges(ink,contrast*.25,max(30,round(scale*2.5)),gap=3)
    es=[e for e in es if e.strength>contrast*.32 and e.b-e.a>scale*2.7 and
        ((e.axis==0 and yt<e.p<yb and e.a>=xl-3 and e.b<=xr+3) or
         (e.axis==1 and xl<e.p<xr and e.a>=yt-3 and e.b<=yb+3))]
    es=merge_edges(es,tolerance=max(1.5,scale*.065),maxgap=scale*.5)
    runs=[]
    for mid in es:
        sides=[]
        for sign in (-1,1):
            cand=[]
            for e in es:
                if e.axis!=mid.axis:continue
                if abs(e.p-(mid.p+sign*2*scale))>scale*.14:continue
                overlap=min(e.b,mid.b)-max(e.a,mid.a)
                if overlap>scale*2.7:cand.append((overlap,e))
            sides.append(max(cand,key=lambda q:q[0])[1] if cand else None)
        if any(e is None for e in sides):continue
        left,right=sides
        if abs((right.p-left.p)/scale-4)>.15:continue
        a=max(left.a,right.a)-scale*.12;b=min(left.b,right.b)+scale*.12
        if b-a<scale*2.8:continue
        # Repeated transverse shelf dividers are 3 or 4 feet apart.
        p0=round(left.p+scale*.18);p1=round(right.p-scale*.18)
        aa=max(0,round(a));bb=round(b)
        if mid.axis==1:strip=ink[aa:bb+1,p0:p1];profile=np.mean(strip,axis=1)
        else:strip=ink[p0:p1,aa:bb+1];profile=np.mean(strip,axis=0)
        peaks,props=find_peaks(profile,height=contrast*.16,prominence=contrast*.10,distance=scale*.65)
        peaks=peaks.astype(float)+aa
        intervals=np.diff(peaks)/scale
        n3=int(np.sum(np.abs(intervals-3)<.22));n4=int(np.sum(np.abs(intervals-4)<.22))
        caps=int(np.sum(np.abs(intervals-1.5)<.2))
        valid=(np.abs(intervals-3)<.22)|(np.abs(intervals-4)<.22)|(np.abs(intervals-1.5)<.2)
        if n3+n4<1 or np.mean(valid)<.85:continue
        # Shelf interiors erase the underlying finish hatch. Sample clear fields
        # on either side of the spine, away from transverse shelf dividers.
        blank=[];half=max(3,round(scale*.36))
        for k,distance in enumerate(intervals):
            if min(abs(distance-3),abs(distance-4))>.22:continue
            along=(peaks[k]+peaks[k+1])/2
            for cross in (mid.p-scale,mid.p+scale):
                x,y=(cross,along) if mid.axis==1 else (along,cross)
                x=round(x);y=round(y)
                patch=ink[y-half:y+half+1,x-half:x+half+1]
                if patch.size:blank.append(float(np.mean(patch)))
        hh,ww=ink.shape
        paper_noise=float(np.mean(ink[round(hh*.07):round(hh*.1),round(ww*.2):round(ww*.5)]))
        if blank and np.median(blank)>paper_noise+contrast*.018:continue
        # Reject duplicate centerlines arising from pixel-width fragments.
        if any(r['axis']==mid.axis and abs(r['p']-mid.p)<scale*.3 and min(r['b'],b)-max(r['a'],a)>scale*4 for r in runs):continue
        runs.append(dict(axis=mid.axis,p=mid.p,a=a,b=b,width=right.p-left.p,
                         sections_3ft=n3,sections_4ft=n4,endcaps=caps,dividers=peaks.tolist(),
                         quality=min(mid.strength,left.strength,right.strength)/contrast-
                         2*abs((right.p-left.p)/scale-4)-abs((left.p+right.p)/2-mid.p)/scale))
    # A shelf edge and a parallel floor hatch can form an offset copy of a
    # genuine run. Two four-foot cabinets cannot occupy the same floor space.
    distinct=[]
    for run in sorted(runs,key=lambda r:-r['quality']):
        if any(run['axis']==r['axis'] and abs(run['p']-r['p'])<scale*3.75 and
               min(run['b'],r['b'])-max(run['a'],r['a'])>scale*2.7 for r in distinct):continue
        distinct.append(run)
    return distinct


def rectangle_candidates(edges,axis,length,width,scale):
    es=sorted([e for e in edges if e.axis==axis and e.b-e.a>=scale*(length-.4)],key=lambda e:e.p)
    out=[]
    for i,e in enumerate(es):
        for f in es[i+1:]:
            dist=(f.p-e.p)/scale
            if dist>width+.16:break
            if dist<width-.16:continue
            a=max(e.a,f.a);b=min(e.b,f.b)
            if abs((b-a)/scale-length)>.35:continue
            if axis==0:box=(a,e.p,b,f.p)
            else:box=(e.p,a,f.p,b)
            out.append((box,axis,a,b,e.p,f.p))
    return out


def path_values(ink,pts,scale):
    # A small radius allows for the antialiasing and subpixel edge positions of scans.
    radius=max(1,round(scale*.07))
    h,w=ink.shape
    vals=[]
    for x,y in pts:
        x=round(x);y=round(y)
        if radius<=x<w-radius and radius<=y<h-radius:
            vals.append(np.max(ink[y-radius:y+radius+1,x-radius:x+radius+1]))
    return np.asarray(vals,dtype=np.float32)


def sample_path(ink,pts,scale):
    values=path_values(ink,pts,scale)
    return float(np.mean(values)) if len(values) else 0.


def x_values(ink,box,scale):
    x0,y0,x1,y1=box
    t=np.linspace(.10,.90,28)
    pts=np.r_[np.c_[x0+t*(x1-x0),y0+t*(y1-y0)],np.c_[x0+t*(x1-x0),y1-t*(y1-y0)]]
    return path_values(ink,pts,scale)


def x_score(ink,box,scale):
    x0,y0,x1,y1=box
    t=np.linspace(.10,.90,28)
    pts=np.r_[np.c_[x0+t*(x1-x0),y0+t*(y1-y0)],np.c_[x0+t*(x1-x0),y1-t*(y1-y0)]]
    return sample_path(ink,pts,scale)


def basic_fixtures(g,ink,rect,scale,contrast,obstacles=None):
    xl,yt,xr,yb=rect
    es=line_edges(ink,contrast*.26,max(30,round(scale*2.3)),gap=2)
    es=[e for e in es if e.strength>contrast*.31 and
        ((e.axis==0 and yt<e.p<yb and xl-3<=e.a and e.b<=xr+3) or
         (e.axis==1 and xl<e.p<xr and yt-3<=e.a and e.b<=yb+3))]
    es=merge_edges(es,tolerance=max(1.,scale*.055),maxgap=scale*.15)
    hh,ww=ink.shape
    paper_noise=float(np.mean(ink[round(hh*.07):round(hh*.1),round(ww*.2):round(ww*.5)]))
    detections={}
    specs={'checkouts':[(0,12.,2.5),(1,12.,2.5)],
           'produce_tables':[(0,5.,4.),(1,4.,5.),(0,4.,5.),(1,5.,4.)],
           'self_checkouts':[(0,3.,3.),(1,3.,3.)]}
    for kind,geoms in specs.items():
        found=[]
        for axis,length,width in geoms:
            for box,_,a,b,p0,p1 in rectangle_candidates(es,axis,length,width,scale):
                x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2
                if kind=='produce_tables':
                    values=x_values(ink,box,scale)
                    fields=[(x0+u*(x1-x0),y0+v*(y1-y0)) for u,v in [(.5,.25),(.75,.5),(.5,.75),(.25,.5)]]
                    field_radius=scale*.22
                elif kind=='self_checkouts':
                    theta=np.linspace(0,2*np.pi,64,endpoint=False)
                    pts=np.c_[cx+scale*.8*np.cos(theta),cy+scale*.8*np.sin(theta)]
                    values=path_values(ink,pts,scale)
                    fields=[(x0+u*(x1-x0),y0+v*(y1-y0)) for u,v in [(.18,.18),(.82,.18),(.82,.82),(.18,.82),(.5,.5)]]
                    field_radius=scale*.12
                else:
                    # X-marked bagging well at one end of the twelve-foot stand.
                    if axis==1:
                        boxes=[(x0,y0,x1,y0+3*scale),(x0,y1-3*scale,x1,y1)]
                        fields=[(x0+u*(x1-x0),y0+v*scale) for u in (.25,.75) for v in (4.5,7.5)]
                    else:
                        boxes=[(x0,y0,x0+3*scale,y1),(x1-3*scale,y0,x1,y1)]
                        fields=[(x0+v*scale,y0+u*(y1-y0)) for u in (.25,.75) for v in (4.5,7.5)]
                    alternatives=[x_values(ink,q,scale) for q in boxes]
                    values=max(alternatives,key=lambda q:float(np.mean(q)))
                    field_radius=scale*.20
                quality=float(np.mean(values))/contrast
                coverage=float(np.mean(values>contrast*.25))
                if quality<.35 or coverage<.84:continue
                blank=[];radius=max(2,round(field_radius))
                for fx,fy in fields:
                    fx=round(fx);fy=round(fy)
                    patch=ink[fy-radius:fy+radius+1,fx-radius:fx+radius+1]
                    if patch.size:blank.append(float(np.mean(patch)))
                if blank and np.median(blank)>paper_noise+contrast*.025:continue
                if obstacles is not None:
                    xa,ya,xb,yb=np.round(box).astype(int)
                    if np.mean(obstacles[ya:yb+1,xa:xb+1])>.025:continue
                same=[j for j,d in enumerate(found) if np.hypot(cx-d['center'][0],cy-d['center'][1])<scale*.65]
                item=dict(box=box,center=(cx,cy),quality=quality,axis=axis)
                if same:
                    j=same[0]
                    if quality>found[j]['quality']:found[j]=item
                else:found.append(item)
        detections[kind]=found
    return detections


def cooler_doors(g,ink,rect,scale,contrast,obstacles=None):
    xl,yt,xr,yb=rect
    es=line_edges(ink,contrast*.19,max(20,round(scale*1.5)),gap=0)
    es=[e for e in es if e.strength>contrast*.25 and
        ((e.axis==0 and yt<e.p<yb and xl-3<=e.a and e.b<=xr+3) or
         (e.axis==1 and xl<e.p<xr and yt-3<=e.a and e.b<=yb+3))]
    es=merge_edges(es,tolerance=max(1.,scale*.04),maxgap=0)
    found=[]
    for axis in (0,1):
        for box,_,a,b,p0,p1 in rectangle_candidates(es,axis,2.3,.6,scale):
            x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2
            t=np.linspace(.12,.88,28)
            scores=[]
            for rev in (False,True):
                yy=y0+t*(y1-y0) if not rev else y1-t*(y1-y0)
                pts=np.c_[x0+t*(x1-x0),yy]
                scores.append(sample_path(ink,pts,scale)/contrast)
            quality=max(scores)
            if quality<.39:continue
            hh,ww=ink.shape
            noise=float(np.mean(ink[round(hh*.07):round(hh*.10),round(ww*.2):round(ww*.5)]))
            fields=[(cx,cy+scale*s) if axis==0 else (cx+scale*s,cy) for s in (-1.2,1.2)]
            blank=[];rad=max(3,round(scale*.3))
            for fx,fy in fields:
                fx=round(fx);fy=round(fy)
                patch=ink[fy-rad:fy+rad+1,fx-rad:fx+rad+1]
                if patch.size:blank.append(float(np.mean(patch)))
            if blank and min(blank)>noise+contrast*.025:continue
            if obstacles is not None:
                xa,ya,xb,yb=np.round(box).astype(int)
                if np.mean(obstacles[ya:yb+1,xa:xb+1])>.025:continue
            if any(np.hypot(cx-d['center'][0],cy-d['center'][1])<scale*.8 for d in found):continue
            found.append(dict(box=box,center=(cx,cy),quality=quality,axis=axis))
    return complete_cooler_rows(found,ink,scale,contrast,obstacles)


def endcaps(g,ink,rect,scale,contrast,runs,obstacles=None):
    """Count attached caps from their dividers, plus cap blocks at run ends whose dividers were interrupted."""
    found=[]
    for r in runs:
        pp=r['dividers']
        for a,b in zip(pp[:-1],pp[1:]):
            if abs((b-a)/scale-1.5)>.20:continue
            s=(a+b)/2
            cx,cy=(r['p'],s) if r['axis']==1 else (s,r['p'])
            found.append(dict(center=(cx,cy),axis=r['axis']))
    xl,yt,xr,yb=rect
    es=line_edges(ink,contrast*.24,max(20,round(scale)),gap=1)
    es=[e for e in es if e.strength>contrast*.30 and
        ((e.axis==0 and yt<e.p<yb and xl-3<=e.a and e.b<=xr+3) or
         (e.axis==1 and xl<e.p<xr and yt-3<=e.a and e.b<=yb+3))]
    es=merge_edges(es,tolerance=max(1.,scale*.045),maxgap=0)
    for axis,length,width in [(0,4.,1.5),(1,1.5,4.),(0,1.5,4.),(1,4.,1.5)]:
        for box,_,a,b,p0,p1 in rectangle_candidates(es,axis,length,width,scale):
            x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2
            if any(np.hypot(cx-d['center'][0],cy-d['center'][1])<scale*.8 for d in found):continue
            run_axis=1 if x1-x0>y1-y0 else 0
            cross,along=(cx,cy) if run_axis==1 else (cy,cx)
            attached=any(r['axis']==run_axis and abs(r['p']-cross)<scale*.3 and
                         min(abs(along-r['dividers'][0]),abs(along-r['dividers'][-1]))<scale*2
                         for r in runs)
            if not attached:continue
            if obstacles is not None:
                xa,ya,xb,yb=np.round(box).astype(int)
                if np.mean(obstacles[ya:yb+1,xa:xb+1])>.025:continue
            t=np.linspace(.12,.88,40)
            qs=[]
            for reverse in (False,True):
                yy=y0+t*(y1-y0) if not reverse else y1-t*(y1-y0)
                qs.append(sample_path(ink,np.c_[x0+t*(x1-x0),yy],scale)/contrast)
            if max(qs)<.38:continue
            found.append(dict(center=(cx,cy),axis=axis,box=box,quality=max(qs)))
    return found



def complete_cooler_rows(found,ink,scale,contrast,obstacles):
    """Register damaged faces against the repeated 2.5-foot cabinet-door pitch."""
    if len(found)<2:return found
    hh,ww=ink.shape
    noise=float(np.mean(ink[round(hh*.07):round(hh*.10),round(ww*.2):round(ww*.5)]))
    def register(cx,cy,axis,strict=False):
        best=None
        for dx in (-2,0,2):
            for dy in (-2,0,2):
                x=cx+dx;y=cy+dy
                if axis==0:box=(x-scale*1.15,y-scale*.3,x+scale*1.15,y+scale*.3)
                else:box=(x-scale*.3,y-scale*1.15,x+scale*.3,y+scale*1.15)
                x0,y0,x1,y1=box
                if x0<3 or y0<3 or x1>ww-3 or y1>hh-3:continue
                if obstacles is not None:
                    xa,ya,xb,yb=np.round(box).astype(int)
                    if np.mean(obstacles[ya:yb+1,xa:xb+1])>.025:continue
                t=np.linspace(.12,.88,28)
                diag=max(sample_path(ink,np.c_[x0+t*(x1-x0),y0+t*(y1-y0)],scale),
                         sample_path(ink,np.c_[x0+t*(x1-x0),y1-t*(y1-y0)],scale))/contrast
                if diag<(.43 if strict else .34):continue
                if axis==0:
                    faces=[np.c_[x0+t*(x1-x0),np.full(len(t),yy)] for yy in (y0,y1)]
                    fields=[(x,y+scale*s) for s in (-1.2,1.2)]
                else:
                    faces=[np.c_[np.full(len(t),xx),y0+t*(y1-y0)] for xx in (x0,x1)]
                    fields=[(x+scale*s,y) for s in (-1.2,1.2)]
                border=np.array([sample_path(ink,p,scale)/contrast for p in faces])
                if border.max()<.40 or border.mean()<(.38 if strict else .27):continue
                blank=[];radius=max(3,round(scale*.33))
                for fx,fy in fields:
                    fx=round(fx);fy=round(fy)
                    patch=ink[fy-radius:fy+radius+1,fx-radius:fx+radius+1]
                    if patch.size:blank.append(float(np.mean(patch)))
                if blank and min(blank)>noise+contrast*.025:continue
                quality=float(diag+.3*border.mean())
                if best is None or quality>best['quality']:
                    best=dict(box=box,center=(x,y),quality=quality,axis=axis,recovered=True)
        return best
    groups=[]
    for d in found:
        axis=d['axis'];normal=d['center'][1-axis]
        group=next((q for q in groups if q[0]['axis']==axis and abs(np.mean([t['center'][1-axis] for t in q])-normal)<scale*.18),None)
        if group is None:groups.append([d])
        else:group.append(d)
    original=list(found)
    for group in groups:
        if len(group)<2:continue
        axis=group[0]['axis'];normal=float(np.median([t['center'][1-axis] for t in group]))
        positions=sorted(t['center'][axis] for t in group)
        pitch=scale*2.5
        proposed=[]
        for left,right in zip(positions[:-1],positions[1:]):
            n=round((right-left)/pitch)
            if 2<=n<=6 and abs((right-left)-n*pitch)<scale*.2:
                proposed.extend((left+(right-left)*j/n,False) for j in range(1,n))
        # A missing end face can otherwise look like the end of the case.
        breaks=[0]+[i+1 for i,(a,b) in enumerate(zip(positions[:-1],positions[1:])) if b-a>pitch*1.6]+[len(positions)]
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            if hi-lo>=2:
                proposed.extend([(positions[lo]-pitch,True),(positions[hi-1]+pitch,True)])
        for along,strict in proposed:
            cx,cy=(along,normal) if axis==0 else (normal,along)
            if any(np.hypot(cx-d['center'][0],cy-d['center'][1])<scale*.8 for d in found):continue
            item=register(cx,cy,axis,strict)
            if item is not None:found.append(item)
    return found


def fixture_mask(shape,runs,basic,coolers,scale):
    """White interiors of CAD blocks are occlusions, not bare floor samples."""
    mask=np.zeros(shape,np.uint8)
    def rectangle(box):
        x0,y0,x1,y1=np.round(box).astype(int)
        x0=max(0,x0-2);y0=max(0,y0-2)
        x1=min(shape[1],x1+3);y1=min(shape[0],y1+3)
        mask[y0:y1,x0:x1]=1
    for run in runs:
        p,a,b=run['p'],run['a'],run['b'];half=run['width']/2
        if run['axis']==0:rectangle((a-1.6*scale,p-half,b+1.6*scale,p+half))
        else:rectangle((p-half,a-1.6*scale,p+half,b+1.6*scale))
    for group in basic.values():
        for item in group:rectangle(item['box'])
    for item in coolers:
        x0,y0,x1,y1=item['box']
        if item['axis']==0:rectangle((x0,y0-scale*3,x1,y1+scale*3))
        else:rectangle((x0-scale*3,y0,x1+scale*3,y1))
    return mask


def cooler_case_segments(t,ink,scale,contrast):
    """Distinguish the shallow repeated door-face block from a partition strip."""
    if not scale*.46<t.width<scale*.82 or t.b-t.a<scale*4: return []
    arr=ink if t.axis==0 else ink.T
    normal=round(t.p);a=max(2,round(t.a));b=min(arr.shape[1]-2,round(t.b))
    if b-a<scale*4:return []
    p0=round(t.p-t.width/2);p1=round(t.p+t.width/2)
    if p0<3 or p1+3>=arr.shape[0]:return []
    face1=np.max(arr[p0-2:p0+3,a:b],axis=0).astype(float)
    face2=np.max(arr[p1-2:p1+3,a:b],axis=0).astype(float)
    profile=np.minimum(face1,face2)
    # A case has a white cabinet behind the front faces. Wall and floor hatches
    # do not supply both that field and the repeated shallow diagonal face.
    rad=max(3,round(scale*.3));fields=[]
    for sign in (-1,1):
        p=round(normal+sign*scale*1.2)
        if p-rad<0 or p+rad>=arr.shape[0]:continue
        fields.append(np.mean(arr[p-rad:p+rad+1,a:b],axis=0))
    if not fields:return []
    from scipy.ndimage import uniform_filter1d,binary_closing
    white=np.min(fields,axis=0)
    white=uniform_filter1d(white,size=max(3,round(scale*.6)))
    smooth=uniform_filter1d(profile,size=max(3,round(scale*.8)))
    hh,ww=ink.shape
    noise=float(np.mean(ink[round(hh*.07):round(hh*.1),round(ww*.2):round(ww*.5)]))
    present=(smooth>contrast*.28)&(white<noise+contrast*.025)
    present=binary_closing(present,structure=np.ones(max(2,round(scale*.35))))
    transitions=np.diff(np.r_[0,present,0].astype(int));starts=np.where(transitions==1)[0];ends=np.where(transitions==-1)[0]
    banks=[]
    inner=max(2,round(t.width/2-4));pitch=scale*2.5
    for lo,hi in zip(starts,ends):
        if hi-lo<scale*3.8:continue
        body=arr[normal-inner:normal+inner+1,a+lo:a+hi].astype(np.float32)
        dx=cv2.Sobel(body,cv2.CV_32F,1,0,ksize=3)[1:-1,2:-2]/8
        dy=cv2.Sobel(body,cv2.CV_32F,0,1,ksize=3)[1:-1,2:-2]/8
        if not dx.size:continue
        ex=float(np.mean(dx*dx));ey=float(np.mean(dy*dy))
        corr=abs(float(np.mean(dx*dy)))/max(1e-6,np.sqrt(ex*ey))
        ratio=float(np.mean(np.abs(dy)))/max(.001,float(np.mean(np.abs(dx))))
        local=arr[p0-2:p1+3,a+lo:a+hi]
        outline=cv2.Canny(local,max(8,contrast*.15),max(20,contrast*.35))
        lines=cv2.HoughLinesP(outline,1,np.pi/180,8,minLineLength=scale*.8,maxLineGap=scale*.3)
        diagonals=[]
        if lines is not None:
            for x0,y0,x1,y1 in lines[:,0]:
                angle=abs((np.degrees(np.arctan2(y1-y0,x1-x0))+90)%180-90)
                if 8<angle<23:diagonals.append((x0+x1)/2)
        distinct=[]
        for x in sorted(diagonals):
            if not distinct or x-distinct[-1]>scale*.7:distinct.append(x)
        if len(distinct)<2:continue
        prof=profile[lo:hi];body_level=float(np.percentile(prof,65))
        if body_level<contrast*.28:continue
        coords=np.arange(len(prof),dtype=float)
        gap_ratio=1.
        for phase in np.arange(0,pitch,1.):
            distance=np.abs((coords-phase+pitch/2)%pitch-pitch/2)
            chosen=distance<max(1.,scale*.045)
            if np.sum(chosen)>=3:
                gap_ratio=min(gap_ratio,float(np.mean(prof[chosen]))/body_level)
        if gap_ratio>.76:continue
        n=round((hi-lo)/pitch)
        if n<2:continue
        mid=a+(lo+hi)/2
        centers=[mid+(j-(n-1)/2)*pitch for j in range(n)]
        banks.append(dict(axis=t.axis,p=t.p,a=a+lo,b=a+hi,count=n,centers=centers,
                          gradient_ratio=ratio,correlation=corr,gap_ratio=gap_ratio))
    return banks
