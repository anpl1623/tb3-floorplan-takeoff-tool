"""Raster extraction, scan normalization, and graphic-scale calibration."""
import cv2
import fitz
import numpy as np

cv2.setNumThreads(2)
PPF=24.0


def components(mask):
    _,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return labels,stats[1:],centers[1:]


def load_scan(path):
    with fitz.open(path) as document:
        if not len(document):raise ValueError('The PDF has no pages')
        page=document[0]
        images=page.get_images(full=True)
        if len(images)==1 and images[0][2]>2000:
            pixmap=fitz.Pixmap(document,images[0][0])
            if pixmap.n>1:pixmap=fitz.Pixmap(fitz.csGRAY,pixmap)
            if pixmap.alpha:pixmap=fitz.Pixmap(pixmap,0)
        else:
            pixmap=page.get_pixmap(matrix=fitz.Matrix(2.78,2.78),colorspace=fitz.csGRAY,alpha=False)
        return np.frombuffer(pixmap.samples,np.uint8).reshape(pixmap.height,pixmap.width).copy()


def normalize_tones(image):
    # Filled columns and small print provide a stable black reference; the large
    # sheet margins provide the paper reference. This also handles faded copies.
    height,width=image.shape
    pooled=image[:height-height%8,:width-width%8].reshape(height//8,8,width//8,8).max(axis=(1,3))
    background=cv2.morphologyEx(pooled,cv2.MORPH_CLOSE,np.ones((21,21),np.uint8))
    background=cv2.GaussianBlur(background.astype('float32'),(0,0),8)
    low,high=np.percentile(background,[10,90])
    if high-low>8:
        field=cv2.resize(background,(width,height),interpolation=cv2.INTER_LINEAR)
        image=np.clip(image.astype('float32')*(254/np.maximum(40,field)),0,255).astype('uint8')
    sample=image[::4,::4]
    black,white=np.percentile(sample,[.05,80])
    if white-black<35:raise ValueError('The scan has insufficient contrast')
    lut=np.clip((np.arange(256,dtype='float32')-black)*(224/(white-black))+30,0,255).astype('uint8')
    return cv2.LUT(image,lut)


def deskew(image):
    h,w=image.shape
    small=cv2.resize(image,(w//2,h//2),interpolation=cv2.INTER_AREA)
    lines=cv2.HoughLinesP((small<155).astype('uint8'),1,np.pi/3600,threshold=200,minLineLength=w*.075,maxLineGap=30)
    if lines is None:raise ValueError('Unable to find drawing axes')
    lines=lines[:,0,:]
    angle=np.degrees(np.arctan2(lines[:,3]-lines[:,1],lines[:,2]-lines[:,0]))
    angle=(angle+45)%90-45
    angle=angle[np.abs(angle)<7]
    if not len(angle):raise ValueError('Unable to determine sheet rotation')
    angle=float(np.median(angle))
    def rotate(a):
        matrix=cv2.getRotationMatrix2D((w/2,h/2),a,1)
        return cv2.warpAffine(image,matrix,(w,h),borderValue=254)
    aligned=rotate(angle)
    binary=(aligned<155).astype('uint8')
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,int(w*.12)),np.uint8))
    _,stats,_=components(horizontal)
    borders=[(x,y,ww,hh)for x,y,ww,hh,n in stats if ww>w*.25 and y<h*.15 and hh<12]
    if borders:
        x,y,ww,hh=max(borders,key=lambda a:a[2])
        points=[]
        y0=max(0,y-10);y1=min(h,y+hh+10)
        for xx in range(int(x+10),int(x+ww-10),7):
            ys=np.flatnonzero(binary[y0:y1,xx])
            if len(ys):points.append((xx,float(np.median(ys))+y0))
        if len(points)>50:
            points=np.array(points)
            slope=np.polyfit(points[:,0],points[:,1],1)[0]
            correction=float(np.degrees(np.arctan(slope)))
            if abs(correction)<.5:
                angle+=correction
                aligned=rotate(angle)
    return aligned,angle


def exterior_bounds(image):
    h,w=image.shape
    binary=(image<155).astype('uint8')
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,int(w*.25)),np.uint8))
    _,stats,_=components(horizontal)
    candidates=[(x,y,ww,hh)for x,y,ww,hh,n in stats if x<w*.25 and w*.3<ww<w*.7 and h*.15<y<h*.87]
    pairs=[]
    for top in candidates:
        for bottom in candidates:
            if bottom[1]-top[1]>h*.28 and abs(top[0]-bottom[0])<12 and abs(top[2]-bottom[2])<15:
                pairs.append((min(top[2],bottom[2]),top,bottom))
    if not pairs:raise ValueError('Unable to locate exterior wall rectangle')
    _,top,bottom=max(pairs,key=lambda a:a[0])
    x0=(top[0]+bottom[0])/2
    x1=x0+(top[2]+bottom[2])/2-1
    y0=top[1]+(top[3]-1)/2
    y1=bottom[1]+(bottom[3]-1)/2
    return list(map(float,(x0,y0,x1,y1)))


def graphic_scale(image,outer):
    h,w=image.shape
    x0,y0,x1,y1=outer
    # Use the solid 0--4 and 8--16 foot blocks. Unlike the outline's far end,
    # these survive blur, resampling and broken one-pixel strokes.
    binary=(image<125).astype('uint8')
    binary=cv2.erode(binary,np.ones((3,3),np.uint8))
    _,stats,centers=components(binary)
    blocks=[]
    for (x,y,ww,hh,n),center in zip(stats,centers):
        if y>y1+h*.025 and x0<x<x1 and 20<ww<w*.05 and 7<hh<h*.015 and n/(ww*hh)>.7:
            blocks.append((x,y,ww,hh,n,center))
    pairs=[]
    for first in blocks:
        for second in blocks:
            x,y,ww,hh,n,c=first
            xx,yy,vv,zz,nn,cc=second
            if not xx>x or abs(c[1]-cc[1])>max(3,hh*.15):continue
            if abs(zz/hh-1)>.2 or abs(vv/ww-2)>.25:continue
            spacing=cc[0]-c[0]
            if abs(spacing/ww-2.5)>.25:continue
            pairs.append((n+nn,spacing/10,first,second))
    if pairs:
        _,ppf,first,second=max(pairs,key=lambda a:a[0])
        return float(ppf),{'method':'solid_blocks','first':list(map(int,first[:4])),'second':list(map(int,second[:4]))}
    # Fallback for a damaged filled segment with an intact scale outline.
    _,stats,_=components((image<155).astype('uint8'))
    bars=[(x,y,ww,hh,n)for x,y,ww,hh,n in stats if y>y1+h*.025 and x>x0 and x+ww<x1+300 and w*.07<ww<w*.16 and 8<hh<h*.015 and n>ww*4]
    if not bars:raise ValueError('Unable to locate 32-foot graphic scale')
    bar=max(bars,key=lambda a:a[2])
    return (bar[2]-1)/32,{'method':'outline','box':list(map(int,bar[:4]))}


def normalize(path,debug=None,margin=0):
    raw=normalize_tones(load_scan(path))
    if raw.shape[0]>raw.shape[1]:raw=cv2.rotate(raw,cv2.ROTATE_90_CLOCKWISE)
    aligned,angle=deskew(raw)
    error=None
    for turn in (0,180):
        image=aligned if not turn else cv2.rotate(aligned,cv2.ROTATE_180)
        try:
            outer=exterior_bounds(image)
            ppf,bar=graphic_scale(image,outer)
            break
        except ValueError as exc:error=exc
    else:raise error
    x0,y0,x1,y1=outer
    # Architectural dimensions lie on a regular design grid. Only remove the
    # scale-end sampling error when both dimensions independently support it.
    width=(x1-x0)/ppf;height=(y1-y0)/ppf
    round_width=round(width);round_height=round(height)
    px=(x1-x0)/round_width;py=(y1-y0)/round_height
    if abs(px/py-1)<.0015 and abs(px/ppf-1)<.004 and abs(py/ppf-1)<.004:
        ppf=(px+py)/2
    width_px=round((x1-x0)/ppf*PPF)+1
    height_px=round((y1-y0)/ppf*PPF)+1
    padding=round(margin*PPF)
    matrix=np.array([[PPF/ppf,0,padding-x0*PPF/ppf],[0,PPF/ppf,padding-y0*PPF/ppf]],float)
    plan=cv2.warpAffine(image,matrix,(width_px+2*padding,height_px+2*padding),borderValue=254)
    metadata={'padding':padding,'ppf':float(ppf),'outer':outer,'scale_bar':bar,'size_ft':[(x1-x0)/ppf,(y1-y0)/ppf],'angle':angle,'turned':turn}
    if debug:cv2.imwrite(str(debug)+'-plan.png',plan)
    return plan,metadata
