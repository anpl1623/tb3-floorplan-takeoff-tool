"""Raster drawing geometry for the Cedar Valley CAD standard."""
import cv2
import fitz
import numpy as np
from scipy import ndimage as ndi


def components(binary):
    return cv2.connectedComponentsWithStats(np.asarray(binary, dtype=np.uint8), 8)


def groups(values):
    v=np.asarray(values, bool)
    p=np.flatnonzero(np.diff(np.r_[False,v,False]))
    return list(zip(p[::2],p[1::2]))


def load_sheet(path):
    doc=fitz.open(path)
    page=doc[0]
    ims=page.get_images(full=True)
    if ims:
        ref=max(ims,key=lambda r:r[2]*r[3])
        im=cv2.imdecode(np.frombuffer(doc.extract_image(ref[0])['image'],np.uint8),cv2.IMREAD_GRAYSCALE)
    else:
        pix=page.get_pixmap(matrix=fitz.Matrix(2.5,2.5),colorspace=fitz.csGRAY)
        im=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width)
    doc.close()
    if max(im.shape)>8500:
        im=cv2.resize(im,None,fx=8000/max(im.shape),fy=8000/max(im.shape),interpolation=cv2.INTER_AREA)
    small=cv2.resize(im,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    lines=cv2.createLineSegmentDetector(0).detect(small)[0]
    if lines is None:raise ValueError('No drawing lines found')
    d=lines[:,0,2:4]-lines[:,0,:2]
    length=np.linalg.norm(d,axis=1)
    angle=(np.degrees(np.arctan2(d[:,1],d[:,0]))+45)%90-45
    ix=(length>200)&(abs(angle)<5)
    a=float(np.median(angle[ix]))
    m=cv2.getRotationMatrix2D((im.shape[1]/2,im.shape[0]/2),a,1)
    im=cv2.warpAffine(im,m,(im.shape[1],im.shape[0]),flags=cv2.INTER_LINEAR,borderValue=255)
    # Calibrate the paper and solid ink levels. Subsampling keeps this cheap;
    # the filled graphic scale and title lettering supply the dark reference.
    low,high=np.percentile(im[::3,::3],[.02,75])
    if high-low>80:
        lut=np.clip((np.arange(256)-low)*(226/(high-low))+25,0,255).astype(np.uint8)
        im=cv2.LUT(im,lut)
    return im, a


def find_scale(im):
    H,W=im.shape
    # The solid 4 ft and 8 ft portions identify the graphic scale, without OCR.
    n,lab,stats,cent=components(im[int(H*.8):int(H*.94),:int(W*.67)]<110)
    options=[]
    for x,y,w,h,area in stats[1:]:
        y+=int(H*.8)
        if w>35 and h>8 and 2.5<w/h<14 and area/(w*h)>.80:
            options.append((int(x),int(y),int(w),int(h)))
    for x,y,w,h in sorted(options):
        # Measure the full outlined 32 ft bar, not the ink-expanded black block.
        block=im[max(0,y-4):y+h+4,max(0,x-5):min(W,x+w*10+30)]
        b=(block<185).astype(np.uint8)
        b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((1,4),np.uint8))
        horiz=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,max(30,int(w*3))),np.uint8))
        n2,_,s2,_=components(horiz)
        for xx,yy,ww,hh,aa in sorted(s2[1:],key=lambda q:-q[2]):
            if 6.5*w<ww<8.5*w:
                return float((ww-2)/32), (x-5+int(xx),y-4+int(yy),int(ww),int(hh))
    raise ValueError('Could not locate the 0–32 foot graphic scale')


def find_building(im,s):
    H,W=im.shape
    xlo,xhi=int(W*.055),int(W*.65)
    ylo,yhi=int(H*.12),int(H*.88)
    a=(im[ylo:yhi,xlo:xhi]<140).astype(np.uint8)
    h=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((1,round(s*8)),np.uint8))
    rows=h.sum(1)
    ys=[(a+b-1)/2+ylo for a,b in groups(rows>W*.25)]
    paired=[y for y in ys if any(.45*s<abs(z-y)<.9*s for z in ys)]
    if len(paired)<4:raise ValueError('Could not find exterior horizontal wall faces')
    top,bot=min(paired),max(paired)
    a=(im[round(top):round(bot)+1,xlo:xhi]<140).astype(np.uint8)
    v=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((round(s*8),1),np.uint8))
    cols=v.sum(0)
    xs=[(a+b-1)/2+xlo for a,b in groups(cols>(bot-top)*.65)]
    paired=[x for x in xs if any(.45*s<abs(z-x)<.9*s for z in xs)]
    if len(paired)<4:raise ValueError('Could not find exterior vertical wall faces')
    return (min(paired),top,max(paired),bot)


def wall_candidates(im,s):
    b=(im<140).astype(np.uint8)
    out=[]
    all_lines=[]
    for ori in ['h','v']:
        a=b if ori=='h' else b.T.copy()
        a=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((1,round(s*3)),np.uint8))
        a=cv2.morphologyEx(a,cv2.MORPH_CLOSE,np.ones((1,round(s*(6.8 if ori=='h' else 3.8))),np.uint8))
        n,lab,st,cen=components(a)
        li=[]
        for x,y,w,h,area in st[1:]:
            if w>s*3 and h<s*.3:
                li.append([float(y+(h-1)/2),float(x),float(x+w-1)])
        li.sort()
        all_lines.extend([(ori,*line) for line in li])
        for i,(c,l,r) in enumerate(li):
            for c2,l2,r2 in li[i+1:]:
                t=c2-c
                if not .21*s<t<.9*s:continue
                lo=max(l,l2);hi=min(r,r2)
                if hi-lo<s*4:continue
                out.append(dict(ori=ori,c=(c+c2)/2,lo=lo,hi=hi,t=t))
    return out,all_lines


def wall_profile(im,wall,s):
    # Sample each run along its length, refine the two wall faces and recover
    # changes of construction even where their outlines share a connected stroke.
    a=im if wall['ori']=='h' else im.T
    c=wall['c']; lo=wall['lo'];hi=wall['hi']
    positions=np.arange(lo+s*.5,hi-s*.35,s*.5)
    out=[]
    for p in positions:
        l=max(0,round(c-s*.6));r=min(a.shape[0],round(c+s*.6)+1)
        q=a[l:r,max(0,round(p-s*.20)):min(a.shape[1],round(p+s*.20)+1)]
        prof=np.median(q,axis=1)
        dark=groups(prof<145)
        minima=[(float((aa+bb-1)/2+l),float(np.min(prof[aa:bb]))) for aa,bb in dark]
        pairs=[(aa,bb) for aa,va in minima for bb,vb in minima if .20*s<bb-aa<.9*s and abs((aa+bb)/2-c)<.25*s]
        if not pairs:continue
        aa,bb=min(pairs,key=lambda z:abs((z[0]+z[1])/2-c)+.2*abs(z[1]-z[0]-wall['t']))
        inside=q[max(0,round(aa-l+3)):max(0,round(bb-l-2))]
        if not inside.size:continue
        med=float(np.median(inside));mean=float(np.mean(inside));std=float(np.std(inside))
        # Nominal widths: 4-7/8 inch stud; 7-1/4 inch stud; 8 inch CMU.
        if bb-aa<s*.49:typ='thin'
        elif med<205:typ='solid'
        else:typ='masonry'
        out.append(dict(p=float(p),a=aa,b=bb,c=(aa+bb)/2,t=bb-aa,median=med,mean=mean,std=std,style=typ))
    return out


def refine_walls(im,candidates,s):
    # Split at actual wall intersections before assigning construction. Face
    # strokes can remain connected where a thin stud partition becomes a
    # thicker rated partition on the same centerline.
    base=[]
    for w0 in candidates:
        prof=wall_profile(im,w0,s)
        if not prof:continue
        valid=[p for p in prof if p['median']<244 and p['t']>s*.34]
        if not valid:valid=prof
        counts={t:sum(p['style']==t for p in valid) for t in ['thin','solid','masonry']}
        style=max(counts,key=counts.get)
        good=[p for p in valid if p['style']==style]
        w=w0.copy()
        w['style']=style
        w['c']=float(np.median([p['c'] for p in good]))
        w['t']=float(np.median([p['t'] for p in good]))
        w['_profile']=prof
        base.append(w)
    for w in base:
        for end in ['lo','hi']:
            possible=[q for q in base if q['ori']!=w['ori'] and abs(q['c']-w[end])<2*s and q['lo']-s<w['c']<q['hi']+s]
            if possible:w[end]=min(possible,key=lambda q:abs(q['c']-w[end]))['c']
    walls=[]
    for w in base:
        cuts=[w['lo'],w['hi']]
        for q in base:
            if q['ori']!=w['ori'] and w['lo']+s*.4<q['c']<w['hi']-s*.4 and q['lo']-s*.7<w['c']<q['hi']+s*.7:
                cuts.append(q['c'])
        cuts=sorted(cuts)
        merged=[cuts[0]]
        for cut in cuts[1:]:
            if cut-merged[-1]>.2*s:merged.append(cut)
        for lo,hi in zip(merged,merged[1:]):
            valid=[p for p in w['_profile'] if lo+s*.35<p['p']<hi-s*.35 and p['median']<244 and p['t']>s*.34]
            if not valid:valid=[p for p in w['_profile'] if lo<p['p']<hi]
            if not valid:continue
            counts={t:sum(p['style']==t for p in valid) for t in ['thin','solid','masonry']}
            style=max(counts,key=counts.get)
            good=[p for p in valid if p['style']==style]
            piece={k:v for k,v in w.items() if not k.startswith('_')}
            piece.update(lo=float(lo),hi=float(hi),style=style,
                         c=float(np.median([p['c'] for p in good])),
                         t=float(np.median([p['t'] for p in good])))
            walls.append(piece)
    return walls


def draw_walls(shape,walls,extra=0):
    mask=np.zeros(shape,np.uint8)
    for w in walls:
        a=round(w['c']-w['t']/2-extra);b=round(w['c']+w['t']/2+extra)
        l=round(w['lo']-extra);r=round(w['hi']+extra)
        if w['ori']=='h': cv2.rectangle(mask,(l,a),(r,b),1,-1)
        else: cv2.rectangle(mask,(a,l),(b,r),1,-1)
    return mask


def floor_regions(im,walls,s):
    # A wall strip spans its openings: door thresholds are excluded from floors.
    wm=draw_walls(im.shape,walls)
    exclusion=draw_walls(im.shape,walls,4)
    b=(im<145).astype(np.uint8)
    b[exclusion>0]=0
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,st,cent=components(b)
    keep=np.zeros(n,np.uint8)
    for k,(x,y,w,h,area) in enumerate(st[1:],1):
        if max(w,h)>s*8 and area>s*3:
            keep[k]=1
    trans=keep[labels]
    del labels,b
    # Extend transition ends to wall faces, without retaining door arcs.
    trans=cv2.morphologyEx(trans,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    barrier=wm|trans
    # bridge the small removal buffer between a transition and the face of a wall
    barrier=cv2.morphologyEx(barrier,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    n,regions,st,cent=components(1-barrier)
    # Remove the outside, as well as holes in letters or boundary linework.
    outer=set(np.unique(np.r_[regions[0,:],regions[-1,:],regions[:,0],regions[:,-1]]))
    good=[k for k in range(1,n) if k not in outer and st[k,4]>s*s*10]
    # Give the finish transition ink to the nearest adjoining floor; wall ink stays excluded.
    # The midpoint of a transition stroke is the area boundary.
    if np.any(trans):
        indices=ndi.distance_transform_edt(regions==0,return_distances=False,return_indices=True)
        yy,xx=np.where((trans>0)&(wm==0))
        regions[yy,xx]=regions[indices[0,yy,xx],indices[1,yy,xx]]
    return regions,good,wm,trans
