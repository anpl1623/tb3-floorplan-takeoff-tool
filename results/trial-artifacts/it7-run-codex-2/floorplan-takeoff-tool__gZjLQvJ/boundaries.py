"""Identify continuous rounded finish boundaries independently of hatch cells."""
import cv2
import numpy as np
from vision import line_edges,merge_edges

def straight_boundaries(ink,rect,walls,scale,contrast,tags,occlusions=None,curves=None):
    """Resolve straight finish zones in rooms carrying multiple finish codes."""
    from geometry import wall_mask
    xl,yt,xr,yb=np.round(rect).astype(int)
    wm=wall_mask(walls,ink.shape)
    _,rooms,_,_=cv2.connectedComponentsWithStats(1-wm[yt:yb+1,xl:xr+1])
    votes={}
    for tag in tags:
        x,y=np.round(tag['center']).astype(int);x-=xl;y-=yt
        if 0<=y<rooms.shape[0] and 0<=x<rooms.shape[1]:votes.setdefault(int(rooms[y,x]),set()).add(tag['code'])
    ambiguous={room for room,codes in votes.items() if room and len(codes)>1}
    if curves is not None:
        # The curved network already supplies the tangents in these rooms.
        # Additional straight hatch strokes would only subdivide those zones.
        touched=set(np.unique(rooms[(curves[yt:yb+1,xl:xr+1]!=0)&(wm[yt:yb+1,xl:xr+1]==0)]))
        ambiguous-=touched
    result=np.zeros(ink.shape,np.uint8)
    if not ambiguous:return result
    edges=line_edges(ink,contrast*.24,max(40,round(scale*8)),gap=3)
    edges=merge_edges([e for e in edges if e.thickness<scale*.18 and e.strength>contrast*.27],
                      tolerance=max(1.,scale*.055),maxgap=scale*.25)
    inside=[]
    for e in edges:
        x,y=((e.a+e.b)/2,e.p) if e.axis==0 else (e.p,(e.a+e.b)/2)
        x=round(x);y=round(y)
        if not xl<x<xr or not yt<y<yb or int(rooms[y-yt,x-xl]) not in ambiguous:continue
        aa=np.linspace(e.a,e.b,60).round().astype(int);pp=round(e.p)
        yy,xx=(np.full_like(aa,pp),aa) if e.axis==0 else (aa,np.full_like(aa,pp))
        valid=(xx>=xl)&(xx<=xr)&(yy>=yt)&(yy<=yb)
        if not np.all(valid) or np.mean(wm[yy,xx])>.1:continue
        if occlusions is not None and np.mean(occlusions[yy,xx])>.1:continue
        inside.append(e)
    for e in inside:
        ends=[]
        for end in (e.a,e.b):
            choices=[]
            for wall in walls:
                if wall.axis==e.axis:continue
                if abs(wall.p-end)<wall.width/2+scale*.4 and wall.a-scale*.3<e.p<wall.b+scale*.3:choices.append(wall.p)
            for f in inside:
                if f.axis==e.axis:continue
                if abs(f.p-end)<scale*.4 and f.a-scale*.4<e.p<f.b+scale*.4:choices.append(f.p)
            ends.append(min(choices,key=lambda v:abs(v-end)) if choices else None)
        if any(v is None for v in ends):continue
        points=((round(ends[0]),round(e.p)),(round(ends[1]),round(e.p))) if e.axis==0 else ((round(e.p),round(ends[0])),(round(e.p),round(ends[1])))
        cv2.line(result,*points,1,3)
    return result

def rounded_boundaries(ink,rect,scale,contrast):
    xl,yt,xr,yb=rect
    edges=line_edges(ink,contrast*.24,max(40,round(scale*10)),gap=3)
    edges=[e for e in edges if e.thickness<scale*.20 and e.strength>contrast*.28 and
           (yt+scale<e.p<yb-scale and e.a>=xl-scale and e.b<=xr+scale if e.axis==0 else
            xl+scale<e.p<xr-scale and e.a>=yt-scale and e.b<=yb+scale)]
    edges=merge_edges(edges,tolerance=max(1.,scale*.06),maxgap=scale*.35)
    hs=[e for e in edges if e.axis==0];vs=[e for e in edges if e.axis==1]
    dil=cv2.dilate(ink,np.ones((3,3),np.uint8)).astype(np.float32)/contrast
    theta=np.linspace(.12,np.pi/2-.12,40)
    corners=[]
    for hi,h in enumerate(hs):
        for vi,v in enumerate(vs):
            for sx,hend in ((1,h.a),(-1,h.b)):
                dx=sx*(hend-v.p)
                if not scale*.25<dx<scale*9:continue
                for sy,vend in ((1,v.a),(-1,v.b)):
                    dy=sy*(vend-h.p)
                    if not scale*.25<dy<scale*9 or abs(dx-dy)>scale*1.2:continue
                    radii=np.arange(max(scale*.8,(dx+dy)/2-scale*.5),min(scale*9,(dx+dy)/2+scale*1.4),max(1.,scale*.06))
                    if not len(radii):continue
                    best=None
                    for radius in radii:
                        cx=v.p+sx*radius;cy=h.p+sy*radius
                        values=[]
                        for offset in (0,-scale*.28,scale*.28):
                            xx=np.round(cx-sx*(radius+offset)*np.cos(theta)).astype(int)
                            yy=np.round(cy-sy*(radius+offset)*np.sin(theta)).astype(int)
                            values.append(dil[yy.clip(0,dil.shape[0]-1),xx.clip(0,dil.shape[1]-1)])
                        ridge=values[0]-(values[1]+values[2])/2
                        coverage=float(np.mean(values[0]>.22));strength=float(np.mean(values[0]));excess=float(np.mean(ridge));fraction=float(np.mean(ridge>.10))
                        score=excess+strength*.15
                        if best is None or score>best[0]:best=(score,radius,cx,cy,coverage,strength,excess,fraction)
                    score,radius,cx,cy,coverage,strength,excess,fraction=best
                    if coverage<.92 or strength<.28 or excess<.12 or fraction<.50:continue
                    corners.append(dict(h=hi,v=vi,sx=sx,sy=sy,r=radius,cx=cx,cy=cy,quality=score))
    distinct=[]
    for c in sorted(corners,key=lambda q:-q['quality']):
        if any(c['sx']==d['sx'] and c['sy']==d['sy'] and abs(c['r']-d['r'])<scale*2 and
               np.hypot(c['cx']-d['cx'],c['cy']-d['cy'])<scale*1.5 for d in distinct):continue
        distinct.append(c)
    corners=distinct
    # A long common tangent supported by both of its rounded ends establishes
    # a boundary. Isolated notes, door arcs and hatch coincidences do not.
    supported=[]
    for i,c in enumerate(corners):
        if any((c['h']==d['h'] and c['sx']!=d['sx'] and abs(c['r']-d['r'])<scale*.7) or
               (c['v']==d['v'] and c['sy']!=d['sy'] and abs(c['r']-d['r'])<scale*.7)
               for j,d in enumerate(corners) if i!=j):supported.append(c)
    mask=np.zeros(ink.shape,np.uint8)
    def draw_line(e,a=None,b=None):
        a=e.a if a is None else a;b=e.b if b is None else b
        pts=((round(a),round(e.p)),(round(b),round(e.p))) if e.axis==0 else ((round(e.p),round(a)),(round(e.p),round(b)))
        cv2.line(mask,*pts,1,3)
    for c in supported:
        h=hs[c['h']];v=vs[c['v']]
        draw_line(h);draw_line(v)
        theta_full=np.linspace(0,np.pi/2,100)
        points=np.c_[c['cx']-c['sx']*c['r']*np.cos(theta_full),c['cy']-c['sy']*c['r']*np.sin(theta_full)]
        cv2.polylines(mask,[np.round(points).astype(np.int32)],False,1,3)
        # At the open end of a U-shaped finish band, its tangent turns outward
        # to a wall. Follow that straight continuation to seal the finish zone.
        other_end=v.b if c['sy']==1 else v.a
        outward=-c['sx']
        for e in hs:
            if abs(e.p-other_end)>scale*.6:continue
            end=e.b if outward<0 else e.a
            far=e.a if outward<0 else e.b
            if abs(end-v.p)<scale*.5 and min(abs(far-xl),abs(far-xr))<scale*1:
                draw_line(e,min(far,v.p),max(far,v.p))
    return mask,dict(corners=supported,hs=hs,vs=vs)
