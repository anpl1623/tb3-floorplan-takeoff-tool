"""Fixture symbol matching, bay classification, and repeated cooler-door counting."""
import cv2
import numpy as np
from pathlib import Path
cv2.setNumThreads(2)

ASSETS=Path(__file__).parent/'assets'

def find_template(plan,template,threshold=.65,rotations=1):
    candidates=[]
    im=plan.astype('float32')
    for rot in range(rotations):
        t=np.rot90(template,rot).copy().astype('float32')
        h,w=t.shape
        match=cv2.matchTemplate(im,t,cv2.TM_CCOEFF_NORMED)
        # Spatial maxima instead of the large cloud around every matching edge.
        maxima=cv2.dilate(match,np.ones((9,9),np.uint8))
        ys,xs=np.where((match>=threshold)&(match>=maxima-1e-6))
        for y,x in zip(ys,xs):candidates.append((float(match[y,x]),float(x),float(y),w,h,rot))
    detections=[]
    for c in sorted(candidates,reverse=True):
        score,x,y,w,h,rot=c
        center=(x+w/2,y+h/2)
        if any(abs(center[0]-(d[1]+d[3]/2))<min(w,d[3])*.45 and abs(center[1]-(d[2]+d[4]/2))<min(h,d[4])*.45 for d in detections):continue
        detections.append(c)
    return detections

def overlap_fraction(a,b):
    _,x,y,w,h,*_=a;_,xx,yy,ww,hh,*_=b
    area=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return area/min(w*h,ww*hh)

def count_coolers(plan,walls):
    from floors import wall_geometry
    template=cv2.imread(str(ASSETS/'cooler.png'),0)
    ds=find_template(plan,template,.79,4)
    geo=wall_geometry(plan,walls)
    candidates=[]
    for d in ds:
        score,x,y,w,h,rot=d
        if rot==0:cx=x+w/2;cy=y+h-9
        elif rot==1:cx=x+w-9;cy=y+h/2
        elif rot==2:cx=x+w/2;cy=y+9
        else:cx=x+9;cy=y+h/2
        axis=rot%2
        near=[l for l in geo if l['axis']==axis and l['lo']-1<(cx if axis==0 else cy)/24<l['hi']+1 and 1.5<abs(l['pos']-(cy if axis==0 else cx)/24)<4.5]
        candidates.append((score,cx,cy,rot,d,bool(near)))
    result=[]
    for a in candidates:
        score,cx,cy,rot,d,near=a
        # Freestanding banks have the same 30-inch door rhythm as wall banks.
        # Require a strong neighboring pane when there is no backing wall.
        if not near:
            along=cx if rot%2==0 else cy
            normal=cy if rot%2==0 else cx
            supported=any(b[3]%2==rot%2 and b[0]>.9 and abs(abs((b[1] if rot%2==0 else b[2])-along)-60)<6 and abs((b[2] if rot%2==0 else b[1])-normal)<6 for b in candidates)
            if not supported:continue
        if any(abs(cx-b[1])<(24 if rot%2==0 else 32) and abs(cy-b[2])<(32 if rot%2==0 else 24)for b in result):continue
        result.append(a[:5])
    return result

def fixture_quantities(plan,walls,debug=None):
    names={'produce':'produce_tables','checkout':'checkouts','self_checkout':'self_checkouts','endcap':'endcaps'}
    quantities={key:0 for key in ['gondola_sections_4ft','gondola_sections_3ft','endcaps','checkouts','self_checkouts','cooler_doors','produce_tables']}
    detected={}
    for name,key in names.items():
        t=cv2.imread(str(ASSETS/(name+'.png')),0)
        ds=find_template(plan,t,.80 if name=='endcap' else .78,4)
        quantities[key]=len(ds);detected[name]=ds
    # Joint nonmaximum suppression avoids counting the first 3 feet of a 4-foot
    # section as another fixture. A double-sided bay is represented by one box.
    section_candidates=[]
    for size in [3,4]:
        t=cv2.imread(str(ASSETS/('gondola'+str(size)+'.png')),0)
        section_candidates.extend((d,size)for d in find_template(plan,t,.78,4))
    sections=[]
    for d,size in sorted(section_candidates,key=lambda a:a[0][0],reverse=True):
        if any(overlap_fraction(d,b)>.28 for b,ss in sections):continue
        if any(overlap_fraction(d,b)>.28 for b in detected['endcap']):continue
        sections.append((d,size))
        quantities['gondola_sections_'+str(size)+'ft']+=1
    coolers=count_coolers(plan,walls)
    quantities['cooler_doors']=len(coolers)
    if debug is not None:
        debug.update(detected);debug['sections']=sections;debug['coolers']=coolers
    return quantities
