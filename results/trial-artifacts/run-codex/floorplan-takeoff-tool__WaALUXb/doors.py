"""Locate numbered door references and measure the corresponding openings."""
import cv2
import numpy as np
from scipy import ndimage


def references(image,scale):
    p=cv2.resize(image,None,fx=20/scale,fy=20/scale,interpolation=cv2.INTER_AREA)
    ink=np.clip(cv2.GaussianBlur(p,(0,0),7).astype(np.float32)-p.astype(np.float32),0,130)
    yy,xx=np.indices((35,35));rr=np.hypot(xx-17,yy-17)
    mask=((rr>=9)&(rr<=16)).astype(np.uint8)
    responses=[]
    for radius in [11.5,12,12.5]:
        template=(np.exp(-.5*((rr-radius)/.85)**2)*100).astype(np.float32)
        response=cv2.matchTemplate(ink,template,cv2.TM_CCORR_NORMED,mask=mask)
        response[~np.isfinite(response)]=0
        responses.append(response)
    score=np.maximum.reduce(responses)
    maximum=cv2.dilate(score,np.ones((21,21),np.uint8))
    ys,xs=np.where((score==maximum)&(score>.78))
    peaks=sorted([(float(score[y,x]),x+17,y+17) for x,y in zip(xs,ys)],reverse=True)
    keep=[]
    for value,x,y in peaks:
        if all(np.hypot(x-xx,y-yy)>20 for _,xx,yy in keep):keep.append((value,x,y))
    return keep


def opening_width(plan,scale,wall,along):
    p=plan if wall['axis']=='H' else plan.T
    position=wall['p'];half=(5/24 if wall['kind']=='P1' else 1/3)-.08
    a=max(0,round((position-half)*scale));b=min(p.shape[0],round((position+half)*scale)+1)
    profile=p[a:b].astype(float).mean(0)
    profile=ndimage.gaussian_filter1d(profile,scale*.12)
    lo=max(0,round((along-10)*scale));hi=min(len(profile),round((along+10)*scale))
    white=np.percentile(profile[lo:hi],95);dark=np.percentile(profile[lo:hi],10)
    gap=profile > (dark*.4+white*.6)
    gap=ndimage.binary_closing(gap,structure=np.ones(max(1,round(.35*scale))))
    labels,n=ndimage.label(gap)
    center=min(len(profile)-1,max(0,round(along*scale)))
    if labels[center]==0:
        near=[k for k in range(max(0,center-round(scale*.8)),min(len(profile),center+round(scale*.8))) if labels[k]>0]
        if not near:return 0.
        center=min(near,key=lambda v:abs(v-center))
    indices=np.flatnonzero(labels==labels[center])
    return (indices[-1]-indices[0]+1)/scale


def pair_leaf_score(sheet,box,scale,wall,along,width):
    """Look for both open leaves, rather than inferring type from width alone."""
    p=sheet if wall['axis']=='H' else sheet.T
    along_origin=box[0] if wall['axis']=='H' else box[1]
    normal_origin=box[1] if wall['axis']=='H' else box[0]
    # The clear gap is slightly narrower than the nominal opening at its jambs.
    nominals=sorted({max(3,round((width+offset)*2)/2) for offset in [.25,.5,.75]})
    scores=[]
    for nominal in nominals:
        for sign in [-1,1]:
            positions=sorted([normal_origin+(wall['p']+sign*nominal*.2)*scale,
                              normal_origin+(wall['p']+sign*nominal*.4)*scale])
            ya=max(0,round(positions[0]));yb=min(p.shape[0],round(positions[1]))
            if yb<=ya:continue
            sides=[]
            for side in [-1,1]:
                x=along_origin+(along+side*nominal/2)*scale
                xa=max(0,round(x-.17*scale));xb=min(p.shape[1],round(x+.17*scale))
                crop=p[ya:yb,xa:xb]
                sides.append(float(np.mean(crop.min(axis=1)<170)) if crop.size else 0.)
            scores.append(min(sides))
    return max(scores) if scores else 0.


def count(sheet,box,scale,walls,regions):
    x1,y1,x2,y2=box
    margin=round(5*scale)
    cx1=max(0,x1-margin);cy1=max(0,y1-margin)
    cx2=min(sheet.shape[1],x2+margin);cy2=min(sheet.shape[0],y2+margin)
    circles=references(sheet[cy1:cy2,cx1:cx2],scale)
    plan=sheet[y1:y2,x1:x2]
    counts={'single':0,'pair':0,'overhead_coiling':0,'auto_sliding':0}
    detections=[]
    for score,x,y in circles:
        x=x/20+(cx1-x1)/scale;y=y/20+(cy1-y1)/scale
        candidates=[]
        for wall in walls:
            along=x if wall['axis']=='H' else y
            perpendicular=y if wall['axis']=='H' else x
            if wall['a']-.5 < along < wall['b']+.5:
                candidates.append((abs(wall['p']-perpendicular),wall))
        if not candidates:continue
        distance,wall=min(candidates,key=lambda pair:pair[0])
        if distance>4.5:continue
        along=x if wall['axis']=='H' else y
        width=opening_width(plan,scale,wall,along)
        # Sliding frames have parallel tracks across the opening. These are
        # distinct from the clear gaps below swing leaves and the coiling door.
        sliding_context=False
        for region in regions:
            if region['finish']!='WOM-1':continue
            bx,by,bxx,byy=region['outer'].bounds
            if wall['axis']=='H':
                sliding_context |= (bx-.5<along<bxx+.5 and min(abs(wall['p']-by),abs(wall['p']-byy))<.7)
            else:
                sliding_context |= (by-.5<along<byy+.5 and min(abs(wall['p']-bx),abs(wall['p']-bxx))<.7)
        leaf_score=pair_leaf_score(sheet,box,scale,wall,along,width) if width>2.8 else 0.
        if width<1.5 and sliding_context:
            kind='auto_sliding'
        elif width>4.2:
            kind='pair' if leaf_score>.5 else 'overhead_coiling'
        elif leaf_score>.6:
            kind='pair'
        else:
            kind='single'
        counts[kind]+=1
        detections.append(dict(x=float(x),y=float(y),score=float(score),kind=kind,
                               opening_width=float(width),pair_leaf_score=leaf_score,wall=wall.copy()))
    return counts,detections
