"""Image normalization and geometry used by the takeoff program."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import cv2
import numpy as np
import fitz
from pathlib import Path
from scipy import ndimage, signal
cv2.setNumThreads(2)


def paper_normalize(im):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h,w=im.shape
    # A smooth paper model, fitted to the light pixels, preserves both solid
    # poches and faint, sparse linework.
    step=max(64, round(max(w,h)/60))
    samples=[]
    for y in range(0,h,step):
        for x in range(0,w,step):
            patch=im[y:y+step,x:x+step]
            samples.append(((x+patch.shape[1]/2)/w,(y+patch.shape[0]/2)/h,np.percentile(patch,96)))
    q=np.array(samples);x,y=q[:,:2].T
    A=np.array([np.ones_like(x),x,y,x*x,x*y,y*y]).T
    keep=np.ones(len(q),bool)
    for _ in range(5):
        coef=np.linalg.lstsq(A[keep],q[keep,2],rcond=None)[0]
        resid=q[:,2]-A@coef
        keep=(resid > -6)&(resid < 10)
    xx=np.linspace(0,1,w,dtype=np.float32)[None,:];yy=np.linspace(0,1,h,dtype=np.float32)[:,None]
    bg=coef[0]+coef[1]*xx+coef[2]*yy+coef[3]*xx*xx+coef[4]*xx*yy+coef[5]*yy*yy
    contrast=np.clip((bg-im)/bg,0,1).astype(np.float32)
    return contrast


def load_pdf(path):
    d=fitz.open(path)
    page=d[0]
    imgs=page.get_images(full=True)
    if len(imgs)==1 and not page.get_text().strip():
        b=d.extract_image(imgs[0][0])['image']
        im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
    else:
        scale=min(3,7200/page.rect.width)
        pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
        im=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)
        im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    if max(im.shape[:2])>9000:
        im=cv2.resize(im,None,fx=9000/max(im.shape[:2]),fy=9000/max(im.shape[:2]),interpolation=cv2.INTER_AREA)
    return im


def deskew(a):
    h,w=a.shape;f=min(1,1800/w)
    small=cv2.resize(a,None,fx=f,fy=f)
    binary=(small>.22).astype(np.uint8)*255
    lines=cv2.HoughLinesP(binary,1,np.pi/1800,80,minLineLength=150,maxLineGap=5)
    angles=[];weights=[]
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            dx=x2-x1;dy=y2-y1
            angle=(np.degrees(np.arctan2(dy,dx))+45)%90-45
            if abs(angle)<7:
                angles.append(angle);weights.append(np.hypot(dx,dy))
    if not angles:return a,0
    idx=np.argsort(angles);angs=np.array(angles)[idx];weights=np.array(weights)[idx]
    angle=angs[np.searchsorted(np.cumsum(weights),sum(weights)/2)]
    # Refine at full line resolution; a thick line biases Hough endpoints.
    from scipy.optimize import minimize_scalar
    patch=a[round(h*.13):round(h*.85),round(w*.04):round(w*.62)]
    patch=cv2.resize(patch,None,fx=.5,fy=.5) ** 2
    ph,pw=patch.shape
    def objective(deg):
        mat=cv2.getRotationMatrix2D((pw/2,ph/2),deg,1)
        q=cv2.warpAffine(patch,mat,(pw,ph),flags=cv2.INTER_LINEAR)
        return -float(np.square(q.sum(0)).sum()/ph+np.square(q.sum(1)).sum()/pw)
    opt=minimize_scalar(objective,bounds=(angle-.4,angle+.4),method='bounded',options={'xatol':.0007})
    angle=opt.x
    if abs(angle)<.005:return a,0
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    return cv2.warpAffine(a,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=0),angle


def components(mask):
    n,lab,stats,cents=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return [(i,*stats[i],*cents[i]) for i in range(1,n)],lab


def graphic_scale(a):
    h,w=a.shape
    # The scale's connected outline has two solid divisions: 4 and 8 feet.
    b=(a>.19).astype(np.uint8)
    cs,lab=components(b)
    candidates=[]
    for i,x,y,ww,hh,area,cx,cy in cs:
        if not(w*.22< x < w*.7 and h*.74<y<h*.95 and w*.05<ww<w*.2 and 8<hh<h*.02):continue
        crop=b[y:y+hh,x:x+ww]
        profile=crop[hh//3:2*hh//3].mean(axis=0)
        high=profile>.7
        transitions=np.where(np.diff(np.r_[False,high,False]))[0]
        runs=list(zip(transitions[::2],transitions[1::2]))
        rr=[(l,r) for l,r in runs if r-l > ww*.05]
        if len(rr)>=2:
            l1,r1=rr[0];l2,r2=rr[1]
            scale=ww/32 if ww/(l2-l1)>3.7 else (l2-l1)/8
            score=abs((r1-l1)/scale-4)+abs((r2-l2)/scale-8)+abs(ww/scale-32)
            candidates.append((score,scale,(x,y,ww,hh)))
    if not candidates:
        raise ValueError('Could not identify the graphic scale')
    ppf,box=min(candidates)[1:]
    x,y,ww,hh=map(int,box)
    prof=a[y+hh//3:y+2*hh//3,x-5:x+ww+5].mean(0)
    # Use the center of the last outline stroke, rather than its outside edge.
    end=prof[-15:]-np.median(prof[-25:-15]);end=np.maximum(end,0)
    end_center=float(np.dot(np.arange(len(prof)-15,len(prof)),end)/end.sum())
    derivative=ndimage.gaussian_filter1d(prof,1,order=1)
    beg=int(np.argmax(derivative[:15]))
    start=beg
    if 0<beg<len(prof)-1:
        aa,bb,cc=derivative[beg-1:beg+2];start+=.5*(aa-cc)/(aa-2*bb+cc)
    # The filled bars extend half the outline weight past its center.
    stroke_area=float(end.sum()/max(float(np.median(prof[10:30])),.1))
    ppf=(end_center-start-stroke_area/2)/32
    return ppf,box


def save_debug(path,a):
    cv2.imwrite(str(path),np.clip(255*(1-a),0,255).astype(np.uint8))


def prepare(path, debug=None):
    im=load_pdf(path)
    a=paper_normalize(im)
    a,angle=deskew(a)
    ppf,scale_box=graphic_scale(a)
    if debug:
        save_debug(str(debug)+'-normalized.png',a)
        print('Deskew',angle,'px/foot',ppf,'scale',scale_box)
    return a,ppf


def main_bounds(a,p):
    h,w=a.shape
    y0,y1=round(h*.13),round(h*.85);x0,x1=round(w*.04),round(w*.62)
    crop=a[y0:y1,x0:x1]
    coords=[]
    for axis,offset in [(0,x0),(1,y0)]:
        pr=np.mean(crop**3,axis=axis)
        peaks,_=signal.find_peaks(pr,distance=max(3,round(p*.2)),prominence=pr.max()*.1)
        peaks=peaks[pr[peaks]>.58*pr.max()]
        lo,hi=peaks.min(),peaks.max()
        # The outermost significant edge is the outside wall face.
        coords.append((float(lo+offset),float(hi+offset)))
    return coords[0][0],coords[1][0],coords[0][1],coords[1][1]


def axis_lines(a,p,threshold,orient):
    b=(a>threshold).astype(np.uint8)
    length=round(p*2)
    kernel=np.ones((1,length) if orient=='h' else (length,1),np.uint8)
    opened=cv2.morphologyEx(b,cv2.MORPH_OPEN,kernel)
    cs,_=components(opened)
    ans=[]
    for i,x,y,w,h,area,cx,cy in cs:
        length=w if orient=='h' else h
        thick=h if orient=='h' else w
        if thick > p or length < p*2:continue
        # Weight the coordinate by dark ink, minimizing threshold/blur bias.
        patch=a[y:y+h,x:x+w]
        if orient=='h':
            profile=(patch*opened[y:y+h,x:x+w]).sum(1)
            peak=int(np.argmax(profile));sel=np.maximum(0,1-np.abs(np.arange(h)-peak)/2)
            pos=y+np.dot(np.arange(h),profile*sel)/(profile*sel).sum();lo=x;hi=x+w-1
        else:
            profile=(patch*opened[y:y+h,x:x+w]).sum(0)
            peak=int(np.argmax(profile));sel=np.maximum(0,1-np.abs(np.arange(w)-peak)/2)
            pos=x+np.dot(np.arange(w),profile*sel)/(profile*sel).sum();lo=y;hi=y+h-1
        ans.append(dict(axis=orient,pos=pos,lo=float(lo),hi=float(hi),thick=thick))
    return ans,opened


def wall_segments(a,p,debug=None):
    bounds=main_bounds(a,p)
    x0,y0,x1,y1=bounds
    pad=round(p*1)
    xx,yy=round(x0)-pad,round(y0)-pad
    a=a[yy:round(y1)+pad+1,xx:round(x1)+pad+1]
    x0-=xx;x1-=xx;y0-=yy;y1-=yy
    # Ink strength comes from the building edges, with local poché excluded.
    ink=[]
    for x in [x0,x1]:
        v=a[round(y0):round(y1),round(x)-2:round(x)+3]
        ink.append(np.quantile(v,.85))
    for y in [y0,y1]:
        v=a[round(y)-2:round(y)+3,round(x0):round(x1)]
        ink.append(np.quantile(v,.85))
    threshold=float(np.median(ink)*.71)
    all_lines=[];walls=[]
    for orient in ['h','v']:
        lines,opened=axis_lines(a,p,threshold,orient)
        all_lines+=lines
        if debug:save_debug(str(debug)+'-'+orient+'-ink.png',opened)
        for i,l in enumerate(lines):
            for r in lines[i+1:]:
                sep=abs(r['pos']-l['pos'])
                if sep < p*.2 or sep>p*.8:continue
                lo=max(l['lo'],r['lo']);hi=min(l['hi'],r['hi'])
                if hi-lo < p*1.8:continue
                # A connected face can step at a change in wall thickness.
                # Refit each edge only over the shared portion of this pair.
                if orient=='h':profile=a[:,round(lo+2):round(hi-2)].mean(1)
                else:profile=a[round(lo+2):round(hi-2)].mean(0)
                def edge_at(pos):
                    beg=max(1,round(pos)-4);end=min(len(profile)-1,round(pos)+5)
                    j=beg+int(np.argmax(profile[beg:end]))
                    aa,bb,cc=profile[j-1:j+2]
                    delta=.5*(aa-cc)/(aa-2*bb+cc) if abs(aa-2*bb+cc)>1e-8 else 0
                    return j+np.clip(delta,-.5,.5)
                pl=edge_at(l['pos']);pr=edge_at(r['pos']);sep=abs(pr-pl)
                widths={'P1':.4,'P2':.65}
                typ=min(widths,key=lambda k:abs(widths[k]*p-sep))
                if abs(widths[typ]*p-sep)>max(2.3,p*.075):continue
                pos=(pl+pr)/2
                walls.append(dict(axis=orient,pos=pos,lo=lo,hi=hi,width=widths[typ]*p,type=typ,observed=sep))
    return a,p,(xx,yy),(x0,y0,x1,y1),walls,all_lines,threshold


def legend_samples(a):
    h,w=a.shape;x0=round(w*.62);y0=round(h*.1)
    patch=a[y0:round(h*.55),x0:round(w*.88)]
    b=(patch>.18).astype(np.uint8)
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    cnts,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for c in cnts:
        x,y,ww,hh=cv2.boundingRect(c)
        if .012*w<ww<.035*w and .006*h<hh<.025*h and 1.75<ww/hh<2.25:
            r=(x+x0,y+y0,ww,hh)
            if any(abs(r[0]-q[0])<10 and abs(r[1]-q[1])<10 for q in rects):
                for i,q in enumerate(rects):
                    if abs(r[0]-q[0])<10 and abs(r[1]-q[1])<10 and r[2]*r[3]>q[2]*q[3]:rects[i]=r
            else:rects.append(r)
    # Pick the column of eight equal swatches.
    groups=[]
    for r in rects:
        group=[s for s in rects if abs(s[0]-r[0])<10 and abs(s[2]-r[2])<10]
        if len(group)>=8:groups.append(sorted(group,key=lambda q:q[1]))
    if not groups:raise ValueError('Could not locate finish legend swatches')
    floors=max(groups,key=len)[:8]
    sx=np.median([r[0] for r in floors]);sw=np.median([r[2] for r in floors]);last=floors[-1]
    sy=last[1]+last[3]
    wall=[]
    for c in cnts:
        x,y,ww,hh=cv2.boundingRect(c);x+=x0;y+=y0
        if abs(x-sx)<10 and abs(ww-sw)<12 and 5<hh<40 and sy<y<sy+700:
            r=(x,y,ww,hh)
            if not any(abs(y-q[1])<10 for q in wall):wall.append(r)
            else:
                for i,q in enumerate(wall):
                    if abs(y-q[1])<10 and ww*hh>q[2]*q[3]:wall[i]=r
    wall=sorted(wall,key=lambda q:q[1])[:4]
    if len(wall)!=4:raise ValueError('Could not locate partition legend swatches')
    return floors,dict(zip(['EXT','P3','P2','P1'],wall))


def wall_descriptor(patch):
    patch=patch.astype(np.float32)
    # Invariant to mirrored diagonal patterns and to the direction of a wall.
    if patch.shape[0]>patch.shape[1]:patch=patch.T
    gx=cv2.Sobel(patch,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(patch,cv2.CV_32F,0,1,ksize=3)
    gx=gx[1:-1,1:-1];gy=gy[1:-1,1:-1]
    correlation=abs(float(np.mean(gx*gy)))/np.sqrt(float(np.mean(gx*gx)*np.mean(gy*gy))+1e-10)
    return np.array([patch.mean(),patch.std(),np.quantile(patch,.25),np.quantile(patch,.75),np.abs(np.diff(patch,axis=1)).mean(),correlation])


def classify_walls(a,p,walls,swatches):
    desc={}
    for k,(x,y,w,h) in swatches.items():
        desc[k]=wall_descriptor(a[y+4:y+h-4,x+8:x+w-8])
    for wall in walls:
        ori,pos,lo,hi=wall['axis'],wall['pos'],wall['lo'],wall['hi']
        if wall['type']=='P1':continue
        half=wall['observed']/2
        if ori=='h':patch=a[round(pos-half+3):round(pos+half-3),round(lo+5):round(hi-5)]
        else:patch=a[round(lo+5):round(hi-5),round(pos-half+3):round(pos+half-3)]
        d=wall_descriptor(patch)
        wall['type']=min(['P2','P3'],key=lambda k:np.linalg.norm((d-desc[k])*np.array([2,1,1,1,2])))
        wall['descriptor']=d.tolist()
    return desc


def assemble_walls(walls,p,bounds):
    x0,y0,x1,y1=bounds
    interior=[]
    for w in walls:
        border=(y0,y1) if w['axis']=='h' else (x0,x1)
        if min(abs(w['pos']-e) for e in border)<p*.85:continue
        interior.append(w.copy())
    exterior=[]
    ext_width=2/3*p
    cx0=x0+ext_width/2;cx1=x1-ext_width/2
    cy0=y0+ext_width/2;cy1=y1-ext_width/2
    for axis,pos,lo,hi in [('h',cy0,cx0,cx1),('h',cy1,cx0,cx1),('v',cx0,cy0,cy1),('v',cx1,cy0,cy1)]:
        exterior.append(dict(axis=axis,pos=pos,lo=lo,hi=hi,type='EXT',width=ext_width,observed=ext_width))
    groups=[]
    for w in sorted(interior,key=lambda q:(q['axis'],q['type'],q['pos'],q['lo'])):
        hits=[g for g in groups if g['axis']==w['axis'] and g['type']==w['type'] and abs(g['pos']-w['pos'])<p*.13]
        hit=None
        for g in hits:
            gap=max(w['lo']-g['hi'],g['lo']-w['hi'],0)
            if gap>p*12:continue
            if gap>p*3.5:
                gl,gh=(g['hi'],w['lo']) if w['lo']>g['hi'] else (w['hi'],g['lo'])
                obstacles=[v for v in interior+exterior if v['axis']!=w['axis'] and gl<v['pos']<gh and v['lo']-p*.5<g['pos']<v['hi']+p*.5]
                if obstacles:continue
            hit=g;break
        if hit is None:
            g=w.copy();g['weight']=w['hi']-w['lo'];groups.append(g)
        else:
            wt=w['hi']-w['lo'];total=hit['weight']+wt
            hit['pos']=(hit['pos']*hit['weight']+w['pos']*wt)/total
            hit['observed']=(hit['observed']*hit['weight']+w['observed']*wt)/total
            hit['width']=hit['observed'];hit['weight']=total
            hit['lo']=min(w['lo'],hit['lo']);hit['hi']=max(w['hi'],hit['hi'])
    for w in groups:
        for key in ['lo','hi']:
            options=[v for v in groups+exterior if v['axis']!=w['axis'] and abs(v['pos']-w[key])<p*2 and v['lo']-p*.9<w['pos']<v['hi']+p*.9]
            if options:
                closest=min(options,key=lambda v:abs(v['pos']-w[key]))
                w[key]=closest['pos']
    return groups+exterior
