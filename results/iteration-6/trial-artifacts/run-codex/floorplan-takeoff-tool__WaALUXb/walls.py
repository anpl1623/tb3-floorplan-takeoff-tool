"""Wall face pairing and rectilinear centerline measurements."""
import cv2
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union


def face_pairs(image, scale):
    segments=[]
    height,width=image.shape
    for axis in ['H','V']:
        p=image if axis=='H' else image.T
        ink=(p<145).astype(np.uint8)
        rules=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,round(scale*2)),np.uint8))
        for thickness,kind in [(scale*5/12,'P1'),(scale*2/3,'thick')]:
            d=round(thickness)
            paired=rules[:-d]&rules[d:]
            n,labels,stats,centers=cv2.connectedComponentsWithStats(paired)
            for i,(x,y,w,h,area) in enumerate(stats[1:],1):
                if w < 2*scale or h > .4*scale:continue
                center=centers[i,1]+d/2
                lo=x;hi=x+w-1
                core=p[max(0,round(center)-2):round(center)+3,lo+3:hi-2]
                median=float(np.median(core));std=float(np.std(core))
                walltype=kind
                if kind=='thick':
                    if median>230 and std<17:continue
                    walltype='P3' if median>200 else 'P2'
                pos=center/scale
                transverse=(height-1)/scale if axis=='H' else (width-1)/scale
                # The exterior is measured as a closed centerline independently;
                # this also excludes automatic door frame lines within that wall.
                if pos<1 or pos>transverse-1:continue
                segments.append(dict(axis=axis,p=pos,a=lo/scale,b=hi/scale,kind=walltype,
                                     median=median,std=std))
    # An opaque cooler-case block can overwrite one wall face with a lighter
    # equipment stroke. Recover that face only on axes established by strong
    # wall pairs; this prevents fixture outlines from introducing new walls.
    established=list(segments)
    for axis in ['H','V']:
        p=image if axis=='H' else image.T
        weak=cv2.morphologyEx((p<180).astype(np.uint8),cv2.MORPH_OPEN,
                             np.ones((1,round(scale*2)),np.uint8))
        axes=[]
        for seg in established:
            if seg['axis']!=axis:continue
            if any(abs(pos-seg['p'])<.12 and kind==seg['kind'] for pos,kind in axes):continue
            axes.append((seg['p'],seg['kind']))
        for pos,kind in axes:
            half=(5/24 if kind=='P1' else 1/3)*scale
            c=pos*scale
            def face(row):
                lo=max(0,round(row)-2);hi=min(p.shape[0],round(row)+3)
                return weak[lo:hi].max(axis=0)
            profile=face(c-half)&face(c+half)
            coreline=p[round(c)-2:round(c)+3].astype(np.float32)
            mean=cv2.GaussianBlur(coreline.mean(axis=0)[None,:],(0,0),scale*.25)[0]
            moment=cv2.GaussianBlur((coreline**2).mean(axis=0)[None,:],(0,0),scale*.25)[0]
            spread=np.sqrt(np.maximum(0,moment-mean**2))
            body=((mean>195)&(mean<235) if kind=='P1' else mean<200 if kind=='P2'
                  else (mean>215)&(mean<244)&(spread>8))
            profile=profile&body.astype(np.uint8)
            profile=cv2.morphologyEx(profile[None,:],cv2.MORPH_CLOSE,
                                    np.ones((1,max(1,round(scale*.15))),np.uint8))[0]
            changes=np.diff(np.r_[0,profile,0].astype(np.int8))
            starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
            for a,b in zip(starts,ends):
                if b-a<2*scale:continue
                core=p[round(c)-2:round(c)+3,a+3:b-3]
                median=float(np.median(core));std=float(np.std(core))
                matches=(185<median<235 if kind=='P1' else median<200 if kind=='P2'
                         else median>215 and std>10)
                if matches:
                    segments.append(dict(axis=axis,p=pos,a=a/scale,b=(b-1)/scale,
                                         kind=kind,median=median,std=std))
    return segments


def group_segments(segments, scale):
    groups=[]
    for axis in ['H','V']:
        ss=sorted([s for s in segments if s['axis']==axis],key=lambda s:s['p'])
        positions=[]
        for seg in ss:
            if not positions or seg['p']-positions[-1][-1]['p']>.12:
                positions.append([seg])
            else:positions[-1].append(seg)
        for same in positions:
            pos=float(np.average([z['p'] for z in same],weights=[z['b']-z['a'] for z in same]))
            for kind in ['P1','P2','P3']:
                runs=sorted([z for z in same if z['kind']==kind],key=lambda z:z['a'])
                intervals=[]
                for z in runs:
                    if intervals and z['a']<=intervals[-1][1]:
                        intervals[-1][1]=max(intervals[-1][1],z['b'])
                        continue
                    if intervals and z['a']-intervals[-1][1]<6.5:
                        # Do not bridge a different wall type on the same axis.
                        gap0,gap1=intervals[-1][1],z['a']
                        occupied=any(o['kind']!=kind and o['a']<gap1 and o['b']>gap0 for o in same)
                        perp=[o for o in segments if o['axis']!=axis and o['a']-.6<=pos<=o['b']+.6]
                        bounded=(any(abs(o['p']-gap0)<.7 for o in perp) and
                                 any(abs(o['p']-gap1)<.7 for o in perp))
                        if not occupied and not bounded:
                            intervals[-1][1]=max(intervals[-1][1],z['b']);continue
                    intervals.append([z['a'],z['b']])
                for a,b in intervals:groups.append(dict(axis=axis,p=pos,a=a,b=b,kind=kind))
    return groups


def exterior_segments(regions,width_ft,height_ft):
    rectangular=[dict(axis='H',p=1/3,a=1/3,b=width_ft-1/3,kind='EXT'),
                 dict(axis='H',p=height_ft-1/3,a=1/3,b=width_ft-1/3,kind='EXT'),
                 dict(axis='V',p=1/3,a=1/3,b=height_ft-1/3,kind='EXT'),
                 dict(axis='V',p=width_ft-1/3,a=1/3,b=height_ft-1/3,kind='EXT')]
    if not regions:return rectangular,sum(z['b']-z['a'] for z in rectangular)
    # Close interior wall gaps between adjacent floor faces. The resulting
    # outside boundary describes the building even when its exterior has jogs.
    joined=unary_union([r['outer'].buffer(.45,join_style=2) for r in regions])
    joined=joined.buffer(-.45,join_style=2)
    if joined.geom_type=='MultiPolygon':joined=max(joined.geoms,key=lambda p:p.area)
    outline=Polygon(joined.exterior).buffer(1/3,join_style=2).simplify(.5,preserve_topology=True)
    rect=box(1/3,1/3,width_ft-1/3,height_ft-1/3)
    coords=np.array(outline.exterior.coords)
    if max(rect.exterior.distance(Point(point)) for point in coords)<.6:
        return rectangular,sum(z['b']-z['a'] for z in rectangular)
    segments=[]
    for a,b in zip(coords[:-1],coords[1:]):
        dx,dy=b-a
        if abs(dx)>5*abs(dy):
            segments.append(dict(axis='H',p=float((a[1]+b[1])/2),a=float(min(a[0],b[0])),
                                 b=float(max(a[0],b[0])),kind='EXT'))
        elif abs(dy)>5*abs(dx):
            segments.append(dict(axis='V',p=float((a[0]+b[0])/2),a=float(min(a[1],b[1])),
                                 b=float(max(a[1],b[1])),kind='EXT'))
    return segments,float(outline.length)


def measure(image, scale, regions=None):
    height,width=image.shape
    width_ft=(width-1)/scale;height_ft=(height-1)/scale
    exterior,perimeter=exterior_segments(regions,width_ft,height_ft)
    raw=group_segments(face_pairs(image,scale),scale)
    segments=[]
    # Exterior CMU has the same hatch as P3. Remove portions identified as
    # exterior before accumulating interior partition lengths.
    for segment in raw:
        intervals=[(segment['a'],segment['b'])]
        for ext in exterior:
            if ext['axis']!=segment['axis'] or abs(ext['p']-segment['p'])>.45:continue
            remaining=[]
            for a,b in intervals:
                lo=max(a,ext['a']-.6);hi=min(b,ext['b']+.6)
                if hi<=lo:remaining.append((a,b));continue
                if a<lo:remaining.append((a,lo))
                if hi<b:remaining.append((hi,b))
            intervals=remaining
        for a,b in intervals:
            if b-a>.5:segments.append(dict(segment,a=a,b=b))
    allwalls=segments+exterior
    # Extend each wall to the centerline of every wall it meets, including T's.
    for seg in segments:
        for end in ['a','b']:
            choices=[o['p'] for o in allwalls if o['axis']!=seg['axis'] and
                     o['a']-.75 <= seg['p'] <= o['b']+.75 and abs(o['p']-seg[end])<.8]
            if choices:seg[end]=min(choices,key=lambda v:abs(v-seg[end]))
    totals={kind:0. for kind in ['EXT','P1','P2','P3']}
    for seg in segments:totals[seg['kind']]+=seg['b']-seg['a']
    totals['EXT']=perimeter
    return totals,allwalls
