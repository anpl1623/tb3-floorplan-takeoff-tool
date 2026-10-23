"""Openings and door symbols on the recovered wall centerlines."""
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from vision import groups

ASSETS=Path(__file__).resolve().parent/'assets'


def automatic_doors(im,s,walls):
    t0=cv2.imread(str(ASSETS/'arrow.png'),cv2.IMREAD_GRAYSCALE)
    a=cv2.resize(im,None,fx=24/s,fy=24/s,interpolation=cv2.INTER_AREA if s>24 else cv2.INTER_LINEAR)
    # Only dark arrow ink; the walk-off mat's hatch otherwise obscures it.
    a=np.maximum(0,190-a.astype(np.float32))
    t0=np.maximum(0,190-t0.astype(np.float32))
    result=[]
    for rotation in [0,1]:
        # Work in the wall's tangent / normal coordinate system.
        sheet=a if rotation==0 else a.T.copy()
        arrows=[[],[]]
        for direction in range(2):
            t=t0 if direction==0 else np.fliplr(t0).copy()
            corr=cv2.matchTemplate(sheet,t,cv2.TM_CCOEFF_NORMED)
            peaks=(corr==cv2.dilate(corr,np.ones((13,13),np.uint8)))&(corr>.78)
            yy,xx=np.where(peaks)
            for y,x in zip(yy,xx):arrows[direction].append((x+t.shape[1]/2,y+t.shape[0]/2,float(corr[y,x])))
        for left in arrows[0]:
            for right in arrows[1]:
                if abs(left[1]-right[1])>5 or not 4*24<right[0]-left[0]<12*24:continue
                x=(left[0]+right[0])/2*s/24;y=(left[1]+right[1])/2*s/24
                ori='h' if rotation==0 else 'v'
                possible=[w for w in walls if w['ori']==ori and w['lo']<left[0]*s/24 and w['hi']>right[0]*s/24
                          and s*.25<abs(w['c']-y)<s*1.15]
                if not possible:continue
                wall=min(possible,key=lambda w:abs(abs(w['c']-y)-s*.7))
                center=(x,wall['c']) if ori=='h' else (wall['c'],x)
                if not any(np.hypot(q[0]-center[0],q[1]-center[1])<s for q in result):
                    result.append((float(center[0]),float(center[1])))
    return result



def _swing_arcs(darker,s,w,l,r):
    """Verify circular swing ink, independently of the surrounding floor hatch.

    Compare each quarter-circle to two concentric background arcs. This rejects
    straight cooler fronts and works for light, antialiased door symbols.
    """
    a=darker if w['ori']=='h' else darker.T
    c=w['c'];t=w['t'];mid=(l+r)/2
    width=(r-l)/s
    leaf=3.0 if width<6.7 else 4.0
    if 3.4<width<4.3:leaf=4.0
    theta=np.linspace(np.pi/12,np.pi*5/12,36)
    cosine=np.cos(theta);sine=np.sin(theta)
    result=[]
    for side in [-1,1]:
        for normal in [-1,1]:
            best=0.0
            hinge=mid+side*leaf*s*(.5 if width<4.3 else 1)
            for shift in [-.22,0,.22]:
                for dy in [-.1,0,.1]:
                    for dr in [-.12,0,.12]:
                        tx=hinge+shift*s;ny=c+normal*t/2+dy*s
                        xy=[]
                        for offset in [0,-.25,.25]:
                            x=np.rint(tx-side*(leaf+dr+offset)*s*cosine).astype(int)
                            y=np.rint(ny+normal*(leaf+dr+offset)*s*sine).astype(int)
                            xy.append((x,y))
                        valid=np.ones(len(theta),bool)
                        for x,y in xy:valid &=(x>=0)&(x<a.shape[1])&(y>=0)&(y<a.shape[0])
                        if valid.sum()<25:continue
                        values=[a[y[valid],x[valid]].astype(float) for x,y in xy]
                        background=np.maximum(values[1],values[2])
                        score=float(np.mean((background-values[0]>20)&(values[0]<230)))
                        best=max(best,score)
            result.append(best)
    if width<4.3:return max(result)
    return max(min(result[0],result[2]),min(result[1],result[3]))

def count_doors(im,s,walls):
    auto=automatic_doors(im,s,walls)
    counts=dict(single=0,pair=0,overhead_coiling=0,auto_sliding=len(auto))
    openings=[]
    darker=cv2.erode(im,np.ones((3,3),np.uint8))
    for wi,w in enumerate(walls):
        a=im if w['ori']=='h' else im.T
        c=w['c'];t=w['t'];lo=round(w['lo']);hi=round(w['hi'])
        # Door thresholds have lighter, thinner strokes than wall faces, even
        # though they are drawn continuously across the door opening.
        metric=np.mean(a[round(c-t/2)-2:round(c+t/2)+3,:],axis=0)
        metric=uniform_filter1d(metric,size=max(3,round(s*.55)))
        reference=metric[lo+10:hi-10]
        valid=np.ones(len(reference),bool)
        for point in auto:
            normal=point[1] if w['ori']=='h' else point[0]
            tangent=point[0] if w['ori']=='h' else point[1]
            if abs(normal-c)<s*.5:
                coords=np.arange(lo+10,hi-10)
                valid &= abs(coords-tangent)>s*4.25
        if np.count_nonzero(valid)>s:reference=reference[valid]
        base=np.percentile(reference,25)
        gap=median_filter(metric>base+12,size=5)
        gap=cv2.morphologyEx(gap.astype(np.uint8)[None,:],cv2.MORPH_CLOSE,np.ones((1,round(s)),np.uint8))[0]
        for l,r in groups(gap[lo:hi]):
            l+=lo;r+=lo;width=(r-l)/s
            if width<1.9:continue
            pos=((l+r)/2,c) if w['ori']=='h' else (c,(l+r)/2)
            if any(np.hypot(pos[0]-q[0],pos[1]-q[1])<s*2 for q in auto):continue
            kind='single' if width<4.3 else 'pair' if width<8 else 'overhead_coiling'
            if kind!='overhead_coiling' and _swing_arcs(darker,s,w,l,r)<.78:continue
            counts[kind]+=1
            openings.append(dict(kind=kind,wall=wi,lo=int(l),hi=int(r),c=float(c),ori=w['ori']))
    return counts,dict(openings=openings,automatic=auto)
