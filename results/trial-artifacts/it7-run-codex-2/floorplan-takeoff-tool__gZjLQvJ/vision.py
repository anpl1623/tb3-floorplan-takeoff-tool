"""Image calibration and low-level drawing geometry. No external programs required."""
import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
cv2.setNumThreads(2)


def read_scan(filename):
    import fitz
    with fitz.open(filename) as doc:
        page = doc[0]
        images = page.get_images(full=True)
        # A scan normally has one image; render when the PDF contains a composite.
        if len(images) == 1:
            item = doc.extract_image(images[0][0])
            im = cv2.imdecode(np.frombuffer(item['image'], np.uint8), cv2.IMREAD_COLOR)
            if im is not None and im.shape[0] > 1000:
                return im
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), colorspace=fitz.csRGB, alpha=False)
        return cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)


def order_corners(pts):
    pts = np.asarray(pts, np.float32)
    return np.array([pts[np.argmin(pts.sum(1))], pts[np.argmax(pts[:,0]-pts[:,1])],
                     pts[np.argmax(pts.sum(1))], pts[np.argmax(pts[:,1]-pts[:,0])]], np.float32)


def rectify(im):
    h,w = im.shape[:2]
    f = min(1., 3600/w)
    g = cv2.cvtColor(cv2.resize(im, (round(w*f),round(h*f))), cv2.COLOR_BGR2GRAY)
    ink = cv2.subtract(cv2.GaussianBlur(g, (0,0), 8), g)
    b = np.uint8(ink > max(10, np.percentile(ink,99.8)*.13))*255
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    cs,_ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = sorted(cs,key=cv2.contourArea,reverse=True)
    corners = None
    for c in cs[:10]:
        if cv2.contourArea(c) < .55*g.size: continue
        ap = cv2.approxPolyDP(c, .002*cv2.arcLength(c,True),True)
        if len(ap)==4:
            corners = order_corners(ap.reshape(4,2)/f)
            break
    if corners is None:
        # Long straight sheet lines are a fallback if a frame is interrupted.
        ls=cv2.HoughLinesP(cv2.Canny(ink,20,60),1,np.pi/1800,100,
                           minLineLength=g.shape[1]*.15,maxLineGap=20)
        angle=0.
        if ls is not None:
            a=ls[:,0]; ang=np.degrees(np.arctan2(a[:,3]-a[:,1],a[:,2]-a[:,0])); ang=(ang+45)%90-45
            ang=ang[np.abs(ang)<5]
            if len(ang): angle=float(np.median(ang))
        M=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        return cv2.warpAffine(im,M,(w,h),borderValue=(255,255,255)), np.array([0,0,w,h])
    tl,tr,br,bl = corners
    ww=(np.linalg.norm(tr-tl)+np.linalg.norm(br-bl))/2
    hh=(np.linalg.norm(bl-tl)+np.linalg.norm(br-tr))/2
    x0,y0=np.round(tl).astype(int); x1,y1=round(x0+ww),round(y0+hh)
    dst=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float32)
    H=cv2.getPerspectiveTransform(corners,dst)
    return cv2.warpPerspective(im,H,(w,h),borderValue=(255,255,255)),np.array([x0,y0,x1,y1])


@dataclass
class Edge:
    axis: int   # 0 = horizontal, 1 = vertical
    p: float    # normal coordinate
    a: float
    b: float
    strength: float = 0.
    thickness: float = 1.


def line_edges(ink, threshold, length, gap=0, split=False):
    edges=[]
    for axis in (0,1):
        b=np.uint8(ink>threshold)
        if gap:
            ker=(gap,1) if axis==0 else (1,gap)
            b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones(ker[::-1],np.uint8))
        ker=(length,1) if axis==0 else (1,length)
        b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones(ker[::-1],np.uint8))
        n,labels,stats,centers=cv2.connectedComponentsWithStats(b)
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if split and (h if axis==0 else w)>6:
                # A single component may change from a filled wall to an outline
                # at a junction. Trace its local bands instead of averaging them.
                local=labels[y:y+h,x:x+w]==i
                canonical=local if axis==0 else local.T
                normal0=y if axis==0 else x
                along0=x if axis==0 else y
                step=max(10,min(20,round(length/4)))
                for start in range(0,canonical.shape[1],step):
                    end=min(canonical.shape[1],start+step)
                    coverage=np.mean(canonical[:,start:end],axis=1)
                    present=coverage>.5
                    transitions=np.diff(np.r_[0,present,0].astype(int))
                    starts=np.where(transitions==1)[0];ends=np.where(transitions==-1)[0]
                    for lo,hi in zip(starts,ends):
                        pp=normal0+float(np.average(np.arange(lo,hi),weights=coverage[lo:hi]))
                        aa=along0+start;bb=along0+end-1
                        if axis==0:vals=ink[max(0,normal0+lo-1):normal0+hi+1,aa:bb+1]
                        else:vals=ink[aa:bb+1,max(0,normal0+lo-1):normal0+hi+1]
                        strength=float(np.mean(np.max(vals,axis=axis))) if vals.size else 0.
                        edges.append(Edge(axis,pp,float(aa),float(bb),strength,float(hi-lo)))
                continue
            if axis==0:
                p=centers[i,1];a=x;b=x+w-1
                vals=ink[max(0,y-1):min(ink.shape[0],y+h+1),x:x+w]
            else:
                p=centers[i,0];a=y;b=y+h-1
                vals=ink[y:y+h,max(0,x-1):min(ink.shape[1],x+w+1)]
            if vals.size:strength=float(np.mean(np.max(vals,axis=axis)))
            else:strength=0.
            edges.append(Edge(axis,p,float(a),float(b),strength,float(h if axis==0 else w)))
    return edges


def find_scale(g,ink,frame):
    """Locate the alternating 0/4/8/16/32 foot graphic scale."""
    h,w=g.shape
    # The bar has two long parallel outlines with alternating filled segments.
    y0=int(h*.82);x0=int(w*.05);x1=int(w*.95)
    crop=ink[y0:int(h*.96),x0:x1]
    es=line_edges(crop,max(18,np.percentile(ink,99.8)*.20),max(80,round(w*.05)),gap=3)
    es=[e for e in es if e.axis==0 and e.b-e.a>w*.055]
    best=None
    for i,e in enumerate(es):
        for f in es[i+1:]:
            dy=abs(e.p-f.p)
            if not (w*.001 < dy < w*.009):continue
            left=max(e.a,f.a);right=min(e.b,f.b)
            if right-left < w*.055:continue
            ya,yb=sorted([round(e.p)+y0,round(f.p)+y0]);xa=round(left)+x0;xb=round(right)+x0
            ww=xb-xa
            if ww/dy<12 or ww/dy>50:continue
            p=np.mean(g[ya+2:yb-1,xa:xb],axis=0)
            if len(p)<40:continue
            # Printed dark intervals [0,1/8] and [1/4,1/2].
            dark=np.r_[p[int(.02*ww):int(.11*ww)],p[int(.27*ww):int(.48*ww)]]
            light=np.r_[p[int(.14*ww):int(.23*ww)],p[int(.53*ww):int(.96*ww)]]
            score=np.mean(light)-np.mean(dark)
            if best is None or score>best[0]:best=(score,xa,xb,ya,yb)
    if best is None or best[0]<30:raise ValueError('Could not calibrate the graphic scale')
    _,xa,xb,ya,yb=best
    # Measure the graphic's geometric ends, not the outer edges of its ink stroke.
    # The filled left end includes half a stroke beyond the geometric endpoint.
    yy=slice(ya+5,yb-4)
    black=float(np.median(g[yy,xa+5:xa+25]))
    white_l=float(np.median(g[yy,xa-25:xa-5]))
    white_r=float(np.median(g[yy,xb+5:xb+25]))
    full=max(1.,white_l-black)
    xx=np.arange(xb-6,xb+7)
    darkness=np.maximum(0.,white_r-np.mean(g[yy,xx],axis=0))
    stroke=float(darkness.sum()/full)
    right=float(np.sum(xx*darkness)/max(1.,darkness.sum()))
    xx=np.arange(xa-5,xa+5)
    profile=np.mean(g[yy,xx],axis=0)
    level=(white_l+black)/2
    crossings=np.where((profile[:-1]>=level)&(profile[1:]<level))[0]
    span=float(xb-xa)
    if len(crossings) and .3<stroke<6:
        i=int(crossings[0])
        left=xx[i]+(profile[i]-level)/(profile[i]-profile[i+1])+stroke/2
        refined=right-left
        if abs(refined-span)<6:span=float(refined)
    from scale_ocr import scale_caption
    feet=scale_caption(g,float(np.percentile(ink,99.8)),(xa,ya,xb,yb))
    return span/feet,(xa,ya,xb,yb)


def find_legend(ink,contrast):
    h,w=ink.shape
    x0=0;x1=w
    b=np.uint8(ink[:,x0:x1]>max(15,contrast*.20))*255
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    cs,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cs:
        x,y,ww,hh=cv2.boundingRect(c);x+=x0
        if ww<.01*w or ww>.05*w or hh<.009*h or hh>.035*h:continue
        if 1.65<ww/hh<2.35 and cv2.contourArea(c)>.75*ww*hh:
            boxes.append((x,y,ww,hh))
    boxes=sorted(boxes,key=lambda q:q[1])
    # Group the swatches by their common left edge and width.
    best=[]
    for anchor in boxes:
        group=[]
        for q in boxes:
            if abs(q[0]-anchor[0])<6 and abs(q[2]-anchor[2])<8:
                if not group or q[1]-group[-1][1]>q[3]*.7:group.append(q)
        if len(group)>len(best):best=group
    if len(best)<8:raise ValueError('Could not locate the eight finish legend swatches')
    # The finish group precedes the shorter partition swatches.
    return best[:8]


def merge_edges(edges,tolerance=2.,maxgap=0.):
    out=[]
    for axis in (0,1):
        es=sorted([e for e in edges if e.axis==axis],key=lambda e:e.p)
        groups=[]
        for e in es:
            if groups and abs(np.mean([t.p for t in groups[-1]])-e.p)<=tolerance:groups[-1].append(e)
            else:groups.append([e])
        for group in groups:
            group=sorted(group,key=lambda e:e.a)
            cur=[]
            for e in group:
                if cur and e.a>max(t.b for t in cur)+maxgap:
                    weights=[t.b-t.a for t in cur]
                    out.append(Edge(axis,float(np.average([t.p for t in cur],weights=weights)),min(t.a for t in cur),max(t.b for t in cur),float(np.mean([t.strength for t in cur])),float(np.average([t.thickness for t in cur],weights=weights))))
                    cur=[]
                cur.append(e)
            if cur:
                weights=[max(1,t.b-t.a) for t in cur]
                out.append(Edge(axis,float(np.average([t.p for t in cur],weights=weights)),min(t.a for t in cur),max(t.b for t in cur),float(np.mean([t.strength for t in cur])),float(np.average([t.thickness for t in cur],weights=weights))))
    return out


def find_wall_swatches(ink,contrast,finish_boxes):
    x,y,w,h=finish_boxes[0]
    y0=finish_boxes[-1][1]+finish_boxes[-1][3]
    crop=ink[y0:min(ink.shape[0],y0+round(ink.shape[0]*.17)),x-5:x+w+5]
    b=np.uint8(crop>max(15,contrast*.2))*255
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    cs,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in cs:
        xx,yy,ww,hh=cv2.boundingRect(c)
        if abs(ww-w)<10 and 5<hh<h*.4 and cv2.contourArea(c)>.6*ww*hh:
            out.append((xx+x-5,yy+y0,ww,hh))
    out.sort(key=lambda q:q[1])
    boxes=[]
    for q in out:
        if not boxes or q[1]-boxes[-1][1]>h*.4:boxes.append(q)
    if len(boxes)<4:raise ValueError('Could not locate the four partition legend swatches')
    return boxes[:4]


def paper_model(g):
    h,w=g.shape
    vals=[];xy=[]
    step=max(60,round(w/50))
    for y in range(step//2,h-step//2,step):
        for x in range(step//2,w-step//2,step):
            patch=g[y-step//2:y+step//2:3,x-step//2:x+step//2:3]
            vals.append(np.percentile(patch,92));xy.append((x/w,y/h))
    xy=np.asarray(xy);x,y=xy.T
    A=np.c_[np.ones(len(x)),x,y,x*x,x*y,y*y];v=np.array(vals)
    keep=np.ones(len(v),bool)
    for _ in range(5):
        coeff=np.linalg.lstsq(A[keep],v[keep],rcond=None)[0]
        err=v-A@coeff
        sd=max(1.,1.48*np.median(np.abs(err-np.median(err))))
        keep=(err>-2.0*sd)&(err<2.5*sd)
    def value(x,y):
        xx=np.asarray(x)/w;yy=np.asarray(y)/h
        return coeff[0]+coeff[1]*xx+coeff[2]*yy+coeff[3]*xx**2+coeff[4]*xx*yy+coeff[5]*yy**2
    return value



def normalize_gray(g,deblur_sigma=0.):
    """Remove paper shading and standardize ink contrast before shape recognition."""
    paper=paper_model(g)
    norm=np.empty_like(g)
    xx=np.arange(g.shape[1],dtype=np.float32)[None,:]
    for yy in range(0,g.shape[0],128):
        ys=np.arange(yy,min(yy+128,g.shape[0]),dtype=np.float32)[:,None]
        bg=np.maximum(40,paper(xx,ys))
        norm[yy:yy+len(ys)]=np.clip(250*g[yy:yy+len(ys)].astype(np.float32)/bg,0,255)
    if deblur_sigma>.1:norm=wiener_sharpen(norm,deblur_sigma)
    rawink=cv2.subtract(cv2.GaussianBlur(norm,(0,0),12),norm)
    gain=150/max(10.,float(np.percentile(rawink,99.8)))
    # A mild unsharp filter recovers fine CAD strokes after scan/photocopy blur.
    blurred=cv2.GaussianBlur(norm,(0,0),1.)
    return np.uint8(np.clip(250+(norm.astype(np.float32)-250+.8*(norm.astype(np.float32)-blurred))*gain,0,255))


def correct_aspect(im,frame):
    """Recover unequal scan/print axis scales from the drawing's circular symbols."""
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ink=cv2.subtract(cv2.GaussianBlur(g,(0,0),12),g)
    con=float(np.percentile(ink,99.8))
    binary=np.uint8(ink>con*.25)*255
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    ratios=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if not (27<min(w,h) and max(w,h)<85 and .75<w/h<1.3 and
                .62<cv2.contourArea(c)/(w*h)<.86 and len(c)>12):continue
        (cx,cy),(a,b),angle=cv2.fitEllipse(c)
        theta=np.deg2rad(angle);ct=np.cos(theta);st=np.sin(theta)
        points=c.reshape(-1,2)-[cx,cy]
        u=(points[:,0]*ct+points[:,1]*st)/(a/2)
        v=(-points[:,0]*st+points[:,1]*ct)/(b/2)
        if np.std(np.sqrt(u*u+v*v))>.04:continue
        ratios.append(np.sqrt((a*a*ct*ct+b*b*st*st)/(a*a*st*st+b*b*ct*ct)))
    if len(ratios)<8:return im,frame,1.
    factor=float(np.median(ratios))
    if not (.80<factor<1.25) or abs(factor-1)<.008:return im,frame,1.
    if np.percentile(ratios,75)-np.percentile(ratios,25)>.06:return im,frame,1.
    h,w=im.shape[:2];newh=round(h*factor);factor=newh/h
    out=cv2.resize(im,(w,newh),interpolation=cv2.INTER_CUBIC)
    frame=np.asarray(frame,dtype=float)*[1,factor,1,factor]
    return out,frame,factor


def edge_blur(g,bar):
    """Estimate the scan point-spread width at filled scale-bar transitions."""
    xa,ya,xb,yb=bar
    p=np.mean(g[ya+4:yb-3],axis=0)
    sigmas=[]
    for fraction,sign in [(.125,1),(.25,-1),(.5,1)]:
        x=round(xa+(xb-xa)*fraction)
        d=sign*np.diff(gaussian_filter1d(p[x-10:x+11],.35))
        d=np.maximum(0,d-np.max(d)*.05)
        if d.sum()<5:continue
        xx=np.arange(len(d))+.5;mean=np.sum(d*xx)/d.sum()
        sigmas.append(np.sqrt(np.sum(d*(xx-mean)**2)/d.sum()))
    return float(np.median(sigmas)) if sigmas else 1.


def wiener_sharpen(g,sigma):
    """Regularized, tiled deconvolution; halos are discarded before assembling tiles."""
    size=1024;halo=32;core=size-2*halo
    freq=np.fft.fftfreq(size).astype(np.float32)
    r2=freq[:,None]**2+freq[None,:]**2
    transfer=np.exp(-2*np.pi**2*sigma*sigma*r2)
    hh,ww=g.shape
    paper=g[round(hh*.07):round(hh*.10),round(ww*.2):round(ww*.5)].astype(np.float32)
    residual=paper-cv2.GaussianBlur(paper,(0,0),2.)
    noise=float(np.std(residual))
    contrast=max(60.,float(np.percentile(g,95)-np.percentile(g,1)))
    regularizer=float(np.clip(20*(noise/contrast)**2,.001,.02))
    inverse=(1+regularizer)*transfer/(transfer*transfer+regularizer)
    out=np.empty(g.shape,np.float32)
    h,w=g.shape
    padded=cv2.copyMakeBorder(g,halo,halo+core,halo,halo+core,cv2.BORDER_REFLECT_101)
    for y in range(0,h,core):
        for x in range(0,w,core):
            tile=padded[y:y+size,x:x+size].astype(np.float32)
            spectrum=cv2.dft(tile,flags=cv2.DFT_COMPLEX_OUTPUT)
            spectrum*=inverse[:,:,None]
            recovered=cv2.idft(spectrum,flags=cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)
            hh=min(core,h-y);ww=min(core,w-x)
            out[y:y+hh,x:x+ww]=recovered[halo:halo+hh,halo:halo+ww]
    return np.uint8(np.clip(out,0,255))
