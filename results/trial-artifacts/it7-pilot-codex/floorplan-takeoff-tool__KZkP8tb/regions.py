import cv2,numpy as np,json,sys,math
from shapely.geometry import box,Polygon,Point,mapping
from shapely.ops import unary_union,polygonize
from walls import get_face_lines,R,stroke_threshold
from textures import TextureModel
from engine import CODES

def wall_rooms(walls,rect,footprint=None):
    strips=[]
    for w in walls:
        if w['kind']=='EXT':continue
        c,s,e=w['c'],w['start'],w['end'];d=w['nominal_width']/2
        p=box(s,c-d,e,c+d) if w['axis']=='h' else box(c-d,s,c+d,e)
        strips.append(p)
    inset=R*2/3
    interior=footprint.buffer(-R/3,join_style=2) if footprint is not None else box(rect[0]+inset,rect[1]+inset,rect[2]-inset,rect[3]-inset)
    free=interior.difference(unary_union(strips))
    return sorted([p for p in free.geoms if p.area>R*R*10] if hasattr(free,'geoms') else [free],key=lambda p:-p.area)

def inset_regions(rooms,hl,vl):
    out=[]
    for room in rooms:
        x0,y0,x1,y1=room.bounds
        cuts=[]
        if room.area>R*R*40:
            # Finish rectangles can share either, both, or neither of their
            # boundaries with walls. Check every corner in local coordinates.
            for cx,cy,sx,sy in [(x0,y0,1,1),(x1,y0,-1,1),(x1,y1,-1,-1),(x0,y1,1,-1)]:
                hs=[(sy*(y-cy),*sorted([sx*(a-cx),sx*(b-cx)])) for y,a,b in hl]
                vs=[(sx*(x-cx),*sorted([sy*(a-cy),sy*(b-cy)])) for x,a,b in vl]
                for hy,hx0,hx1 in hs:
                    if not R*2<hy<(y1-y0)-R:continue
                    if abs(hx0)>R*.5:continue
                    for vx,vy0,vy1 in vs:
                        if not R*2<vx<(x1-x0)-R:continue
                        if abs(vy0)>R*.5 or abs(hx1-vx)>R*.5 or abs(vy1-hy)>R*.5:continue
                        xx,yy=cx+sx*vx,cy+sy*hy
                        reg=box(min(cx,xx),min(cy,yy),max(cx,xx),max(cy,yy)).intersection(room)
                        if R*R*12<reg.area<room.area*.8:cuts.append(reg)
            hs=[q for q in hl if y0+R*.4<q[0]<y1-R*.4 and q[2]-q[1]>R*2]
            vs=[q for q in vl if x0+R*.4<q[0]<x1-R*.4 and q[2]-q[1]>R*2]
            for top in hs:
                for bottom in hs:
                    if bottom[0]-top[0]<R*2:continue
                    if abs(top[1]-bottom[1])>R*.5 or abs(top[2]-bottom[2])>R*.5:continue
                    lefts=[q for q in vs if abs(q[0]-top[1])<R*.5 and abs(q[1]-top[0])<R*.5 and abs(q[2]-bottom[0])<R*.5]
                    rights=[q for q in vs if abs(q[0]-top[2])<R*.5 and abs(q[1]-top[0])<R*.5 and abs(q[2]-bottom[0])<R*.5]
                    if not lefts or not rights:continue
                    left=min(lefts,key=lambda q:abs(q[0]-top[1]));right=min(rights,key=lambda q:abs(q[0]-top[2]))
                    reg=box(left[0],top[0],right[0],bottom[0]).intersection(room)
                    if R*R*12<reg.area<room.area*.8:cuts.append(reg)
        unique=[]
        for p in sorted(cuts,key=lambda p:-p.area):
            if any(p.symmetric_difference(q).area<R*R*2 for q in unique):continue
            unique.append(p)
        if unique:
            network=unary_union([room.boundary]+[p.boundary for p in unique])
            pieces=[p for p in polygonize(network) if p.area>R*R*2 and room.covers(p.representative_point())]
            out.extend(pieces)
        else:out.append(room)
    return out


def floor_quantities(src,walls,rect,legend,debug=None,annotations=None,boundary_src=None,footprint=None,occlusions=None):
    rooms=wall_rooms(walls,rect,footprint)
    boundary=src if boundary_src is None else boundary_src
    threshold=stroke_threshold(legend,margin=.04)
    hl,_=get_face_lines(boundary,threshold=threshold)
    vl,_=get_face_lines(boundary,True,threshold=threshold)
    rooms=inset_regions(rooms,hl,vl)
    expanded=[]
    for p in rooms:
        loop=loop_regions(src,p,hl,vl,debug)
        if loop:expanded.extend(loop)
        else:expanded.append(p)
    rooms=expanded
    model=TextureModel(legend)
    from tags import detect as detect_tags,in_polygon
    if annotations is None:annotations=detect_tags(src,debug)
    quantities={c:0. for c in CODES};output=[]
    col=[(100,180,240),(220,220,190),(150,190,120),(190,140,200),(200,180,100),(120,150,240),(150,150,180),(200,130,130)]
    im=cv2.cvtColor(np.uint8((1-src)*255),cv2.COLOR_GRAY2BGR)
    for i,p in enumerate(rooms):
        label,votes,coords,labels,dist=model.classify_polygon(src,p,occlusions)
        text_votes=in_polygon(annotations,p)
        if text_votes:
            label=CODES.index(max(text_votes,key=text_votes.get))
        quantities[CODES[label]]+=p.area/(R*R)
        if debug:print('room',i,round(p.area/(R*R),2),CODES[label],votes.tolist(),[round(a/R,2) for a in p.bounds])
        output.append(dict(area=p.area/(R*R),code=CODES[label],votes=votes.tolist(),bounds=list(p.bounds),geometry=mapping(p)))
        if debug:
            mask=np.zeros(src.shape,np.uint8);contours=[]
            for g in (p.geoms if hasattr(p,'geoms') else [p]):
                outer=np.round(np.array(g.exterior.coords)).astype(np.int32);contours.append(outer)
                cv2.fillPoly(mask,[outer],255)
                for ring in g.interiors:cv2.fillPoly(mask,[np.round(np.array(ring.coords)).astype(np.int32)],0)
            pixels=mask!=0;im[pixels]=np.uint8(im[pixels]*.75+np.array(col[label])*.25)
            cv2.polylines(im,contours,True,col[label],3)
            pt=p.representative_point();text='%s %.1f sf'%(CODES[label],p.area/(R*R))
            cv2.putText(im,text,(round(pt.x),round(pt.y)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,180),1)
    if debug:cv2.imwrite(debug+'/regions.png',im);json.dump(output,open(debug+'/regions.json','w'),indent=2)
    return {c:round(q,1) for c,q in quantities.items()}


def fit_corner(src,x,y,sx,sy,max_radius,guess=None):
    # x,y is the square corner; sx/sy point into the rectangle. Evaluate the
    # actual curved ink with a narrow positive band and neighboring negatives.
    rs=np.arange(.45*R,max_radius,.1*R)
    angle=np.linspace(0,np.pi/2,90)
    best=(-1,2*R)
    for r in rs:
        cx=x+sx*r;cy=y+sy*r
        xx=cx-sx*r*np.cos(angle);yy=cy-sy*r*np.sin(angle)
        mid=(angle>.15)&(angle<np.pi/2-.15);xx=xx[mid];yy=yy[mid];co=np.cos(angle[mid]);si=np.sin(angle[mid])
        vals=[]
        for off in [0,-2.8,2.8]:
            xs=(xx-sx*off*co).astype(np.float32)[None,:];ys=(yy-sy*off*si).astype(np.float32)[None,:]
            vals.append(cv2.remap(src,xs,ys,cv2.INTER_LINEAR).mean())
        score=vals[0]-(vals[1]+vals[2])*.5
        if score>best[0]:best=(float(score),float(r))
    return best[1],best[0]

def rounded_poly(l,t,r,b,radii):
    # TL, TR, BR, BL; a zero radius creates a square corner.
    pts=[]
    for cx,cy,rad,angles in [
        (l+radii[0],t+radii[0],radii[0],(180,270)),
        (r-radii[1],t+radii[1],radii[1],(270,360)),
        (r-radii[2],b-radii[2],radii[2],(0,90)),
        (l+radii[3],b-radii[3],radii[3],(90,180))]:
        if rad<.01:
            pts.append((cx,cy));continue
        aa=np.deg2rad(np.linspace(*angles,40))
        pts.extend(zip(cx+rad*np.cos(aa),cy+rad*np.sin(aa)))
    return Polygon(pts)

def loop_in_bottom(src,room,hl,vl,debug=None):
    x0,y0,x1,y1=room.bounds
    if room.area<R*R*2000:return None
    hs=[]
    for l in hl:
        y,s,e=l
        if e-s<R*25 or not y0+R*2<y<y1-R*3:continue
        if not room.contains(Point((s+e)/2,y)):continue
        if room.boundary.distance(Point((s+e)/2,y))<R*.5:continue
        hs.append(l)
    choices=[]
    for outer in hs:
        for inner in hs:
            d=inner[0]-outer[0]
            if not R*3<d<R*10:continue
            if abs(outer[1]-inner[1])>R*3 or abs(outer[2]-inner[2])>R*3:continue
            for bottom in hs:
                if bottom[0]-inner[0]<R*8:continue
                if abs(inner[1]-bottom[1])>R*3 or abs(inner[2]-bottom[2])>R*3:continue
                h=bottom[0]-inner[0]
                lefts=[v for v in vl if inner[1]-R*4<v[0]<inner[1]+R and min(v[2],bottom[0])-max(v[1],inner[0])>h*.7]
                rights=[v for v in vl if inner[2]-R<v[0]<inner[2]+R*4 and min(v[2],bottom[0])-max(v[1],inner[0])>h*.7]
                for li in lefts:
                    for ri in rights:
                        los=[v for v in vl if abs(v[0]-(li[0]-d))<R*.6 and v[2]-v[1]>h*.7]
                        ros=[v for v in vl if abs(v[0]-(ri[0]+d))<R*.6 and v[2]-v[1]>h*.7]
                        for lo in los:
                            for ro in ros:
                                front=(lo[2]+ro[2])/2
                                if not bottom[0]+R*1<front<bottom[0]+R*12:continue
                                if abs(lo[2]-ro[2])>R*1:continue
                                # Refine at the horizontal finish boundaries
                                # outside the loop, where the front floor joins.
                                frontlines=[q for q in hl if abs(q[0]-front)<R and (abs(q[2]-lo[0])<R*.4 or abs(q[1]-ro[0])<R*.4)]
                                if frontlines:front=np.median([q[0] for q in frontlines])
                                error=abs((li[0]-lo[0])-d)+abs((ro[0]-ri[0])-d)+abs(lo[2]-ro[2])
                                choices.append((error,lo[0],ro[0],outer[0],li[0],ri[0],inner[0],bottom[0],float(front)))
    if not choices:return None
    error,l,r,t,il,ir,it,ib,front=min(choices,key=lambda q:q[0])
    maxr=min((r-l)/3,(front-t)*.65,R*15)
    rtl,sl=fit_corner(src,l,t,1,1,maxr);rtr,sr=fit_corner(src,r,t,-1,1,maxr)
    if min(sl,sr)<.035:return None
    inner_rs=[];inner_scores=[]
    for x,y,sx,sy in [(il,it,1,1),(ir,it,-1,1),(ir,ib,-1,-1),(il,ib,1,-1)]:
        rad,score=fit_corner(src,x,y,sx,sy,min((ir-il)/3,(ib-it)/3,R*8));inner_rs.append(rad);inner_scores.append(score)
    outer=rounded_poly(l,t,r,front,[rtl,rtr,0,0])
    island=rounded_poly(il,it,ir,ib,inner_rs)
    frontpoly=box(x0,front,x1,y1)
    band=unary_union([outer,frontpoly]).difference(island).intersection(room)
    rest=room.difference(band)
    if debug:
        print('loop',dict(l=l/R,r=r/R,t=t/R,il=il/R,ir=ir/R,it=it/R,ib=ib/R,front=front/R,radii=[rtl/R,rtr/R]+[q/R for q in inner_rs],scores=[sl,sr]+inner_scores))
    return [band,rest]

def loop_regions(src,room,hl,vl,debug=None):
    """Recognize a circulation loop opening toward any side of the plan."""
    if room.area<R*R*2000:return None
    from shapely.affinity import affine_transform
    h,w=src.shape
    transforms=[(1,0,0,1,0,0),(0,1,-1,0,0,w-1),(-1,0,0,-1,w-1,h-1),(0,-1,1,0,h-1,0)]
    inverses=[transforms[0],(0,-1,1,0,w-1,0),transforms[2],(0,1,-1,0,0,h-1)]
    for k,m in enumerate(transforms):
        if k==0:
            result=loop_in_bottom(src,room,hl,vl,debug)
        else:
            a,b,d,e,ox,oy=m
            hs=[];vs=[]
            for axis,ls in [('h',hl),('v',vl)]:
                for c,s,t in ls:
                    pts=[(s,c),(t,c)] if axis=='h' else [(c,s),(c,t)]
                    pts=[(a*x+b*y+ox,d*x+e*y+oy) for x,y in pts]
                    (x,y),(u,v)=pts
                    if abs(y-v)<1e-5:hs.append([y,min(x,u),max(x,u)])
                    else:vs.append([x,min(y,v),max(y,v)])
            result=loop_in_bottom(np.rot90(src,k).copy(),affine_transform(room,m),hs,vs,debug)
        if result:
            if debug and k:print('loop orientation',k*90)
            return [affine_transform(p,inverses[k]) for p in result] if k else result
    return None


