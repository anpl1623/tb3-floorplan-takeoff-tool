"""Door leaf/arc recognition and sliding/coiling assembly detection."""
import cv2
import numpy as np
from scipy import ndimage
from floors import wall_geometry
from walls import runs


def arc_measure(im,hinge,leaf,swing,radius=72):
    # leaf and swing are perpendicular unit vectors from the hinge.
    # A circular arc is invariant to the hatch behind it; compare its stroke with
    # narrow strips on either side, ignoring the tangencies at the wall and leaf.
    t=np.linspace(.10,np.pi/2-.10,100)
    unit=np.cos(t)[:,None]*np.array(leaf)+np.sin(t)[:,None]*np.array(swing)
    pts=np.array(hinge)+radius*unit
    def sample(off):
        p=pts+off*unit
        return ndimage.map_coordinates(im,[p[:,1],p[:,0]],order=1,mode='constant',cval=255)
    center=np.minimum.reduce([sample(o)for o in [-1,0,1]])
    surround=(sample(-4)+sample(4))/2
    contrast=surround-center
    return float(np.mean(contrast)),float(np.mean(contrast>12))

def detect_swings(plan,walls,padded,padding=144,debug=False):
    geo=wall_geometry(plan,walls)
    im=cv2.GaussianBlur(padded.astype('float32'),(0,0),.45)
    ink=255-im
    candidates=[]
    for wi,l in enumerate(geo):
        for sign in [-1,1]:
            # Door arcs attach to the wall face, and the leaf swings perpendicular to it.
            face=(l['pos']+sign*l['width']/2)*24+padding
            lo=int(l['lo']*24)+padding;hi=int(l['hi']*24)+padding
            ax=l['axis']
            mat=ink if ax==0 else ink.T
            # Average ink along each possible 3-foot leaf, away from its endpoints.
            y0=int(round(face+sign*5));y1=int(round(face+sign*69))
            strip=mat[min(y0,y1):max(y0,y1)+1,:]
            profile=cv2.blur(np.mean(strip,axis=0).astype('float32')[None,:],(3,1)).ravel()
            selected=(profile>45)
            for a,b in runs(selected[lo:hi]):
                a+=lo;b+=lo
                # Broad wall intersections can be rejected by the circular arc score too.
                xs=np.arange(max(lo,a-2),min(hi,b+2))
                if len(xs)>25:
                    xs=np.r_[xs[:5],xs[-5:]]
                for xx in xs:
                    # An open leaf consists of two thin parallel lines, so its hinge can be
                    # displaced by up to two pixels from the darkest column.
                    for tangent in [-1,1]:
                        leaf=(0,sign)if ax==0 else(sign,0)
                        swing=(tangent,0)if ax==0 else(0,tangent)
                        hinge=(xx,face)if ax==0 else(face,xx)
                        best=(-1,None,None,72)
                        for radius in (60,66,72,78,84,90,96):
                            for shift in [-1.5,0,1.5]:
                                hs=np.array(hinge)+shift*np.array(leaf)
                                score,support=arc_measure(im,hs,leaf,swing,radius)
                                if score>best[0]:best=(score,support,hs,radius)
                        score,support,hs,radius=best
                        if score>30 and support>.75:
                            candidates.append(dict(score=score,support=support,hinge=hs.tolist(),leaf=leaf,swing=swing,wall=wi,axis=ax,radius=radius))
    # Multiple pixel hypotheses of the same hinge are one leaf.
    result=[]
    for c in sorted(candidates,key=lambda d:d['score'],reverse=True):
        if any(np.linalg.norm(np.array(c['hinge'])-d['hinge'])<12 for d in result):continue
        result.append(c)
    if debug:
        col=cv2.cvtColor(padded,cv2.COLOR_GRAY2BGR)
        for i,d in enumerate(result):
            x,y=map(round,d['hinge']);cv2.circle(col,(x,y),10,(0,0,255),2)
            leaf=np.array(d['leaf']);swing=np.array(d['swing'])
            t=np.linspace(0,np.pi/2,100);pts=np.array([x,y])+d['radius']*(np.cos(t)[:,None]*leaf+np.sin(t)[:,None]*swing)
            cv2.polylines(col,[pts.astype('int32')],False,(0,0,255),2)
            cv2.putText(col,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
        return result,col
    return result

def pair_swings(swings):
    used=set();pairs=[]
    for i,a in enumerate(swings):
        if i in used:continue
        for j in range(i+1,len(swings)):
            b=swings[j]
            if j in used or a['axis']!=b['axis'] or a['wall']!=b['wall']:continue
            if a['leaf']!=b['leaf'] or a['swing']!=tuple(-v for v in b['swing']):continue
            ea=np.array(a['hinge'])+a['radius']*np.array(a['swing'])
            eb=np.array(b['hinge'])+b['radius']*np.array(b['swing'])
            if np.linalg.norm(ea-eb)<10:
                used.update([i,j]);pairs.append((i,j));break
    return len(swings)-len(used),len(pairs)

def detect_sliding(plan,walls,padded,padding=144):
    from fixtures import find_template,ASSETS
    template=cv2.imread(str(ASSETS/'sliding_arrow.png'),0)
    detections=find_template(padded,template,.8,4)
    geometry=wall_geometry(plan,walls)
    arrows=[]
    for score,x,y,width,height,rotation in detections:
        cx=(x+width/2-padding)/24;cy=(y+height/2-padding)/24
        axis=rotation%2
        near=[(i,wall)for i,wall in enumerate(geometry)if wall['axis']==axis and wall['lo']+.25<(cx if axis==0 else cy)<wall['hi']-.25 and abs(wall['pos']-(cy if axis==0 else cx))<1.5]
        if not near:continue
        index,wall=min(near,key=lambda item:abs(item[1]['pos']-(cy if axis==0 else cx)))
        position=cx if axis==0 else cy
        direction=(-1,1,1,-1)[rotation]
        normal=cy if axis==0 else cx
        if any(index==a[0] and abs(position-a[1])<.5 for a in arrows):continue
        arrows.append((index,position,direction,normal,score))
    used=set();result=[]
    for i,first in enumerate(arrows):
        if i in used or first[2]!=-1:continue
        choices=[(second[1]-first[1],j,second)for j,second in enumerate(arrows)if j not in used and second[0]==first[0] and second[2]==1 and 4<second[1]-first[1]<12 and abs(second[3]-first[3])<.6]
        if choices:
            gap,j,second=min(choices)
            result.append((first[0],(first[1]+second[1])/2,max(first[4],second[4])))
            used.update([i,j])
    # One arrow can be partially obscured by the walk-off-mat hatch. The remaining
    # arrow still identifies a sliding assembly in the adjacent wall.
    for i,arrow in enumerate(arrows):
        if i not in used:result.append((arrow[0],arrow[1]-arrow[2]*3,arrow[4]))
    return result

def detect_coiling(plan,walls,padded,padding=144):
    # In the coiling-door opening the CMU hatch is replaced by a dashed center
    # line. Measure that horizontal/vertical ridge inside the wall band. Automatic
    # sliding panels and swing openings are removed by their independent detections.
    geo=wall_geometry(plan,walls)
    im=padded.astype('float32')
    result=[]
    for wi,l in enumerate(geo):
        if l['width']<.55:continue
        mat=im if l['axis']==0 else im.T
        y=round(l['pos']*24)+padding
        x0=round(l['lo']*24)+padding;x1=round(l['hi']*24)+padding
        scores=[]
        for d in [-1,0,1]:
            center=mat[y+d-1:y+d+2].mean(axis=0)
            near=(mat[y+d-5:y+d-3].mean(axis=0)+mat[y+d+4:y+d+6].mean(axis=0))/2
            score=cv2.blur((near-center)[None,:],(33,1)).ravel()
            scores.append(score)
        score=np.max(scores,axis=0)
        selected=(score[x0:x1]>10).astype('uint8')[None,:]
        selected=cv2.morphologyEx(selected,cv2.MORPH_CLOSE,np.ones((1,9),np.uint8))
        for a,b in runs(selected[0]):
            if b-a>24*5:
                result.append((wi,(a+x0-padding)/24,(b+x0-padding)/24))
    return result

def door_quantities(plan,walls,padded,padding=144,debug=None):
    swings=detect_swings(plan,walls,padded,padding)
    single,pair=pair_swings(swings)
    sliding=detect_sliding(plan,walls,padded,padding)
    coiling=detect_coiling(plan,walls,padded,padding)
    coiling=[door for door in coiling if not any(door[0]==auto[0] and door[1]-1<auto[1]<door[2]+1 for auto in sliding)]
    coiling=[door for door in coiling if not any(door[0]==swing['wall'] and door[1]-1<(swing['hinge'][swing['axis']]-padding+swing['swing'][swing['axis']]*swing['radius']/2)/24<door[2]+1 for swing in swings)]
    if debug is not None:debug.update(swings=swings,sliding=sliding,coiling=coiling)
    return dict(single=single,pair=pair,overhead_coiling=len(coiling),auto_sliding=len(sliding))
