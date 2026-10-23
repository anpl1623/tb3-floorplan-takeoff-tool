"""Find orthogonal wall faces, classify their infill, and join centerlines."""
import cv2
import numpy as np
from vision import components,PPF


def runs(mask):
    delta=np.diff(np.r_[False,mask,False].astype('int8'))
    return list(zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1)))


def line_candidates(plan,axis=0):
    image=plan if axis==0 else plan.T
    binary=(image<155).astype('uint8')
    opened=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,48),np.uint8))
    _,stats,_=components(opened)
    groups=[]
    for x,y,width,height,area in sorted(stats,key=lambda a:a[1]):
        if height>8 or width<48:continue
        coordinate=y+(height-1)/2
        group=next((g for g in groups if abs(g['pos']-coordinate)<2.5),None)
        if group is None:groups.append({'pos':float(coordinate),'intervals':[(x,x+width-1)]})
        else:group['intervals'].append((x,x+width-1))
    for group in groups:
        intervals=[]
        for low,high in sorted(group['intervals']):
            if intervals and low-intervals[-1][1]<PPF*6.1:
                intervals[-1][1]=max(intervals[-1][1],high)
            else:intervals.append([low,high])
        group['intervals']=intervals
    candidates=[]
    for i,first in enumerate(groups):
        for second in groups[i+1:]:
            width=second['pos']-first['pos']
            if not 6<width<21:continue
            for a,b in first['intervals']:
                for c,d in second['intervals']:
                    low=max(a,c);high=min(b,d)
                    if high-low<PPF*3:continue
                    candidates.append({'axis':axis,'lo':float(low),'hi':float(high),'pos':(first['pos']+second['pos'])/2,'width':float(width),'faces':(first['pos'],second['pos'])})
    return candidates


def segment_style(plan,wall,low,high):
    image=plan if wall['axis']==0 else plan.T
    first,second=wall['faces']
    start=round(low)+3;end=round(high)-2
    if end<=start:return None
    # A clear corridor between collinear walls has no face strokes. Door
    # openings retain their jamb/threshold line and belong to the measured wall.
    presence=[]
    for face in (first,second):
        index=round(face)
        strip=image[max(0,index-2):index+3,start:end]
        presence.append(float(np.mean(np.min(strip,axis=0)<165)))
    if max(presence)<.25:return None
    center=round(wall['pos'])
    inset=min(12,max(3,(end-start)//8))
    profile=image[center-1:center+2,start+inset:end-inset].mean(axis=0)
    if not len(profile):return None
    fill=float(np.median(profile))
    if wall['width']>=10.5 and fill<207:return 'P2'
    return 'P1' if wall['width']<13.2 else 'P3'


def get_walls(plan,debug=None):
    height,width=plan.shape
    candidates=[]
    for axis in (0,1):
        limit=height if axis==0 else width
        candidates.extend(wall for wall in line_candidates(plan,axis) if 23<wall['pos']<limit-23)
    segments=[]
    for wall in candidates:
        cuts=[wall['lo'],wall['hi']]
        for other in candidates:
            if other['axis']==wall['axis']:continue
            if other['lo']-22<=wall['pos']<=other['hi']+22 and wall['lo']+3<other['pos']<wall['hi']-3:
                cuts.append(other['pos'])
        cuts=sorted(set(cuts))
        for low,high in zip(cuts,cuts[1:]):
            if high-low<4:continue
            style=segment_style(plan,wall,low,high)
            if style:segments.append({**wall,'lo':float(low),'hi':float(high),'typ':style})
    # Merge only touching, collinear segments of the same type. In particular,
    # do not bridge the clear corridor between an upper and a lower room wall.
    walls=[]
    for segment in sorted(segments,key=lambda a:(a['axis'],a['pos'],a['lo'])):
        same=[wall for wall in walls if wall['axis']==segment['axis'] and wall['typ']==segment['typ'] and abs(wall['pos']-segment['pos'])<3 and segment['lo']<=wall['hi']+12 and segment['hi']>=wall['lo']-12]
        if not same:walls.append(segment.copy());continue
        merged=same[0]
        merged['lo']=min(segment['lo'],*(wall['lo']for wall in same))
        merged['hi']=max(segment['hi'],*(wall['hi']for wall in same))
        for old in same[1:]:walls.remove(old)
    if debug:
        overlay=cv2.cvtColor(plan,cv2.COLOR_GRAY2BGR)
        for i,wall in enumerate(walls):
            low,high=round(wall['lo']),round(wall['hi']);coordinate=round(wall['pos'])
            a=(low,coordinate)if wall['axis']==0 else(coordinate,low)
            b=(high,coordinate)if wall['axis']==0 else(coordinate,high)
            cv2.line(overlay,a,b,{'P1':(0,180,0),'P2':(220,0,0),'P3':(0,0,220)}[wall['typ']],3)
            cv2.putText(overlay,str(i)+wall['typ'],((a[0]+b[0])//2,(a[1]+b[1])//2),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,0,220),2)
        cv2.imwrite(str(debug),overlay)
    return walls
