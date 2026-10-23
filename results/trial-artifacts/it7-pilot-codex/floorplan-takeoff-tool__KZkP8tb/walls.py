import cv2,numpy as np,json,sys,math
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from engine import components
R=16

def get_face_lines(src,vertical=False,threshold=.43):
    a=src.T.copy() if vertical else src
    raw=a
    # A gray opening keeps the persistent part of each stroke and removes text
    # and hatch marks. Faces are then joined across normal door openings.
    persistence=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((1,24),np.uint8))
    b=np.uint8(persistence>threshold)
    # Close short raster dropouts, preserving wall endpoints.
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))
    st,_=components(b)
    lines=[]
    for x,y,w,h,ar in st:
        if w<18 or h>20:continue
        sub=persistence[y:y+h,x:x+w]
        # Solid poche can connect both faces into one component.
        pro=np.mean(sub,axis=1)
        if h>4:
            yy=[float(y),float(y+h-1)]
        else:yy=[float(y+np.average(np.arange(h),weights=pro))]
        for v in yy:lines.append([v,float(x),float(x+w-1)])
    # Merge pieces from the same face. A long door gap must have a supporting
    # face on both sides; it is not an extra wall segment.
    lines.sort()
    groups=[]
    for l in lines:
        possible=[g for g in groups if abs(np.median([q[0] for q in g])-l[0])<1.7]
        if possible:possible[-1].append(l)
        else:groups.append([l])
    out=[]
    for g in groups:
        y=np.median([l[0] for l in g]);ranges=sorted([(l[1],l[2]) for l in g]);runs=[]
        for x0,x1 in ranges:
            if runs:
                gap=x0-runs[-1][1]
                lo,hi=int(runs[-1][1]+2),int(x0-1)
                band=raw[max(0,round(y)-1):round(y)+2,lo:hi]
                bridge=gap<R*.3 or (gap<R*16 and band.size and np.median(band.max(axis=0))>.16)
            else:bridge=False
            if bridge:runs[-1][1]=max(runs[-1][1],x1)
            else:runs.append([x0,x1])
        for x0,x1 in runs:
            if x1-x0>=R*2:out.append([float(y),float(x0),float(x1)])
    return out,persistence

def candidates(src,vertical=False,threshold=.43):
    ls,pers=get_face_lines(src,vertical,threshold)
    walls=[]
    for i,a in enumerate(ls):
        for b in ls[i+1:]:
            d=b[0]-a[0]
            if not R*.30<d<13.6:continue
            start=max(a[1],b[1]);end=min(a[2],b[2]);length=end-start
            if length<R*2:continue
            # Ensure these are persistent parallel faces, not neighboring ends
            # of two unrelated lines.
            j0,j1=int(start+5),int(end-5)
            qa=pers[max(0,round(a[0])-1):round(a[0])+2,j0:j1].max(axis=0)
            qb=pers[max(0,round(b[0])-1):round(b[0])+2,j0:j1].max(axis=0)
            strength=float(np.mean(np.minimum(qa,qb)))
            support=float(np.mean((qa>.4)&(qb>.4)))
            if support<.4 or strength<.4:continue
            walls.append(dict(axis='v' if vertical else 'h',c=(a[0]+b[0])/2,width=d,start=start,end=end,strength=strength,support=support))
    # Prefer a complete wall to two overlapping, slightly offset detections.
    out=[]
    for c in sorted(walls,key=lambda w:-(w['strength']*w['support']*(w['end']-w['start']))):
        if any(abs(c['c']-d['c'])<8 and min(c['end'],d['end'])-max(c['start'],d['start'])>min(c['end']-c['start'],d['end']-d['start'])*.7 for d in out):continue
        out.append(c)
    return out,ls

def align_network(walls):
    walls=[dict(w) for w in walls]
    for axis in ['h','v']:
        ls=sorted([w for w in walls if w['axis']==axis],key=lambda w:w['c']);groups=[]
        for w in ls:
            if groups and abs(w['c']-np.mean([q['c'] for q in groups[-1]]))<2:groups[-1].append(w)
            else:groups.append([w])
        for g in groups:
            c=np.average([w['c'] for w in g],weights=[w['end']-w['start'] for w in g])
            for w in g:w['c']=float(c)
    for w in walls:
        for endpoint in ['start','end']:
            cs=[q['c'] for q in walls if q['axis']!=w['axis'] and q['start']-R<=w['c']<=q['end']+R and abs(q['c']-w[endpoint])<R]
            if cs:w[endpoint]=min(cs,key=lambda c:abs(c-w[endpoint]))
    return walls


def rectangular_shell(src,rect):
    x0,y0,x1,y1=map(round,rect)
    checks=[src[max(0,y0-2):y0+3,x0:x1].max(axis=0),
            src[y1-2:y1+3,x0:x1].max(axis=0),
            src[y0:y1,max(0,x0-2):x0+3].max(axis=1),
            src[y0:y1,x1-2:x1+3].max(axis=1)]
    for p in checks:
        # Door frames retain a thin face stroke through the opening. A recess
        # or stepped perimeter leaves a genuinely empty stretch of this face.
        bad=p<.20
        ed=np.flatnonzero(np.diff(np.r_[False,bad,False]))
        if any(b-a>R*3.5 for a,b in zip(ed[::2],ed[1::2])):return False
    return True


def closed_outline(walls,rect,annotations=None):
    from shapely.geometry import LineString,Point
    from shapely.ops import unary_union,polygonize
    aligned=align_network(walls)
    lines=[]
    for w in aligned:
        if w['end']-w['start']<2:continue
        if w['axis']=='h':pts=[(w['start'],w['c']),(w['end'],w['c'])]
        else:pts=[(w['c'],w['start']),(w['c'],w['end'])]
        lines.append(LineString(pts))
    if not lines:return None
    pieces=list(polygonize(unary_union(lines)))
    if not pieces:return None
    outline=unary_union(pieces)
    if outline.geom_type=='MultiPolygon':outline=max(outline.geoms,key=lambda p:p.area)
    x0,y0,x1,y1=rect;expected=np.array([x0+R/3,y0+R/3,x1-R/3,y1-R/3])
    if np.max(np.abs(np.array(outline.bounds)-expected))>R*1.5:return None
    if outline.area<(x1-x0)*(y1-y0)*.35:return None
    # An accidentally unclosed room must not become a missing part of the
    # building. Its thin partitions and finish annotations expose that error.
    thin=0.
    boundary=outline.boundary.buffer(2)
    for w in aligned:
        if w['width']>=R*.5:continue
        pts=[(w['start'],w['c']),(w['end'],w['c'])] if w['axis']=='h' else [(w['c'],w['start']),(w['c'],w['end'])]
        thin+=LineString(pts).intersection(boundary).length
    if thin>max(R*2,outline.length*.02):return None
    if annotations:
        for t in annotations:
            if t['score']>.82 and not outline.buffer(R*.5).covers(Point(t['x'],t['y'])):return None
    return outline


def stroke_threshold(legend,margin=.13):
    """Choose line ink above any uniform gray fill on this sheet."""
    fills=[]
    for p in legend['walls']:
        h,w=p.shape;a=p[round(h*.33):round(h*.67),4:-4]
        if a.size and np.std(a)<.05:fills.append(float(np.median(a)))
    for p in legend['floors']:
        a=p[4:-4,4:-4]
        if a.size and np.std(a)<.05:fills.append(float(np.median(a)))
    return max(.43,min(.8,max(fills)+margin)) if fills else .43

def detect(src,rect,debug=None,legend=None,return_footprint=False,annotations=None):
    threshold=stroke_threshold(legend) if legend is not None else .43
    hor,hl=candidates(src,threshold=threshold);ver,vl=candidates(src,True,threshold);walls=hor+ver
    from shapely.geometry import box as geometry_box
    x0,y0,x1,y1=rect;offset=R/3
    footprint=None
    if not rectangular_shell(src,rect):footprint=closed_outline(walls,rect,annotations)
    if footprint is None:
        # Complete a rectangular perimeter continuously through door openings.
        hc=[y0+offset,y1-offset];vc=[x0+offset,x1-offset]
        walls=[w for w in walls if (min(abs(w['c']-c) for c in (hc if w['axis']=='h' else vc))>R*.6)
               and (y0+offset<w['c']<y1-offset if w['axis']=='h' else x0+offset<w['c']<x1-offset)]
        for axis,cs,s,e in [('h',hc,vc[0],vc[1]),('v',vc,hc[0],hc[1])]:
            for c in cs:walls.append(dict(axis=axis,c=c,width=R*2/3,start=s,end=e,strength=1.,support=1.))
        footprint=geometry_box(vc[0],hc[0],vc[1],hc[1])
    if debug:
        json.dump(dict(walls=walls,hl=hl,vl=vl),open(debug+'/walls.json','w'),indent=2)
        a=cv2.cvtColor(np.uint8(255*(1-src)),cv2.COLOR_GRAY2BGR)
        for i,d in enumerate(walls):
            c,s,e=d['c'],d['start'],d['end'];pt1=(round(s),round(c)) if d['axis']=='h' else (round(c),round(s));pt2=(round(e),round(c)) if d['axis']=='h' else (round(c),round(e));cv2.line(a,pt1,pt2,(0,0,255),2);cv2.putText(a,str(i)+':%.1f'%d['width'],pt1,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
        cv2.imwrite(debug+'/walls.png',a)
    return (walls,footprint) if return_footprint else walls


NOMINAL={'P1':4.875/12,'P2':7.25/12,'P3':8/12,'EXT':8/12}

def core(p):
    h,w=p.shape
    return p[max(1,round(h*.33)):min(h-1,round(h*.67)),3:-3]

def spectral_peaks(p):
    a=core(p);a=a-a.mean(axis=1,keepdims=True)
    power=(np.abs(np.fft.rfft(a,512,axis=1))**2).mean(axis=0)
    j=int(np.argmax(power[25:210])+25)
    return j/512 if power[j]/max(power.sum(),1e-8)>.05 else None

def type_features(p,freqs):
    a=core(p)
    if a.size==0:return np.zeros(2+len(freqs))
    mean=np.mean(a);std=np.std(a)
    a=a-a.mean(axis=1,keepdims=True)
    amps=[]
    for freq in freqs:
        m=0
        for f in [freq-.003,freq,freq+.003]:
            z=np.exp(-2j*np.pi*f*np.arange(a.shape[1]))
            m=max(m,float(np.sqrt(np.mean(np.abs(a@z)**2))*2/a.shape[1]))
        amps.append(m)
    return np.array([mean,std]+amps)

def classify_walls(src,walls,legend,rect,footprint=None):
    # Align fragments sharing a centerline before connecting junctions.
    walls=align_network(walls)
    # Combine collinear fragments only where they touch. This retains corridor
    # gaps between two otherwise aligned room partitions.
    merged=[]
    for w in sorted(walls,key=lambda w:(w['axis'],w['c'],w['start'])):
        if merged and w['axis']==merged[-1]['axis'] and abs(w['c']-merged[-1]['c'])<1 and w['start']-merged[-1]['end']<1.5:
            merged[-1]['end']=max(merged[-1]['end'],w['end'])
        else:merged.append(w)
    walls=merged
    # Classify separately on either side of each T-junction: two wall types
    # can share one centerline but have different poche and thickness.
    atoms=[]
    for w in walls:
        cuts=[w['start'],w['end']]+[q['c'] for q in walls if q['axis']!=w['axis'] and q['start']-R*.5<=w['c']<=q['end']+R*.5 and w['start']+2<q['c']<w['end']-2]
        cuts=sorted(set(cuts))
        for s,e in zip(cuts[:-1],cuts[1:]):
            if e-s<2:continue
            q=dict(w);q.update(start=s,end=e);atoms.append(q)
    refs=legend['walls'][1:]
    freqs=sorted(set(f for p in refs if (f:=spectral_peaks(p)) is not None))
    model=np.array([type_features(p,freqs) for p in refs])
    weights=np.array([5,3]+[10]*len(freqs))
    edges={'h':[rect[1]+R/3,rect[3]-R/3],'v':[rect[0]+R/3,rect[2]-R/3]}
    for w in atoms:
        if footprint is not None:
            from shapely.geometry import LineString
            c,s,e=w['c'],w['start'],w['end']
            line=LineString([(s,c),(e,c)] if w['axis']=='h' else [(c,s),(c,e)])
            exterior=line.intersection(footprint.boundary.buffer(2)).length>line.length*.9
        else:exterior=min(abs(w['c']-c) for c in edges[w['axis']])<R*.3
        if exterior:
            kind='EXT';dist=[0,0,0]
        else:
            a=src.T if w['axis']=='v' else src;c=w['c'];wid=w['width'];s,e=w['start'],w['end']
            # Refit the two faces locally; adjacent types can share a
            # centerline but have different thicknesses.
            y0=max(0,round(c)-12);y1=min(a.shape[0],round(c)+13)
            strip=a[y0:y1,round(s)+4:round(e)-3]
            if strip.size:
                pro=np.quantile(strip,.7,axis=1)
                peaks,_=find_peaks(pro,distance=3,height=.35)
                pairs=[]
                for u in peaks:
                    for v in peaks:
                        if 3.5<v-u<14 and abs((u+v)/2+y0-c)<3:
                            pairs.append((pro[u]+pro[v]-.08*abs((u+v)/2+y0-c),u,v))
                if pairs:
                    _,u,v=max(pairs)
                    def refine(j):
                        lo=max(0,j-1);hi=min(len(pro),j+2)
                        base=min(pro[lo],pro[hi-1])
                        ww=np.maximum(pro[lo:hi]-base,1e-4)
                        return float(np.average(np.arange(lo,hi),weights=ww))
                    u,v=refine(u),refine(v);wid=v-u;c=(u+v)/2+y0
                    w['width']=float(wid)
            p=a[round(c-wid/2):round(c+wid/2)+1,round(s)+3:round(e)-2]
            # Short clean samples reduce the effect of the door leaf and jambs.
            fs=[]
            for x in range(0,max(1,p.shape[1]-31),16):
                pp=p[:,x:x+48]
                if pp.shape[1]<20:continue
                fs.append(type_features(pp,freqs))
            feature=np.median(fs,axis=0) if fs else type_features(p,freqs)
            dist=np.sum(((model-feature)*weights)**2,axis=1)
            # Width is a secondary check, useful for nearly blank poches.
            dist+=.06*np.array([((wid/R-NOMINAL[k])/.1)**2 for k in ['P3','P2','P1']])
            kind=['P3','P2','P1'][int(np.argmin(dist))]
            w['features']=feature.tolist();w['dist']=dist.tolist()
        w['kind']=kind;w['nominal_width']=NOMINAL[kind]*R
    return atoms

def quantities(walls):
    return {kind:round(sum((w['end']-w['start'])/R for w in walls if w['kind']==kind),2) for kind in ['EXT','P1','P2','P3']}


def draw_typed_walls(src,walls,directory):
    a=cv2.cvtColor(np.uint8((1-src)*255),cv2.COLOR_GRAY2BGR)
    colors={'EXT':(0,0,220),'P1':(0,150,0),'P2':(220,0,0),'P3':(0,130,220)}
    for w in walls:
        c,s,e=w['c'],w['start'],w['end'];color=colors[w['kind']]
        p=(round(s),round(c)) if w['axis']=='h' else (round(c),round(s))
        q=(round(e),round(c)) if w['axis']=='h' else (round(c),round(e))
        cv2.line(a,p,q,color,3)
        cv2.putText(a,w['kind'],p,cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    cv2.imwrite(str(directory)+'/typed-walls.png',a)
