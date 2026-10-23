import cv2,numpy as np,math,json,os
from scipy import ndimage as ndi
R=16

def template(kind,res=16,stroke=.055):
    dims={'produce':(5,4),'self':(3,3),'checkout':(3.7,12),'cooler':(2.5,1.8),'endcap':(4,1.5),'gondola4':(4,4),'gondola3':(4,3)}
    w,h=dims[kind];pad=.15;ss=4
    a=np.zeros((int(round((h+pad*2)*res*ss))+1,int(round((w+pad*2)*res*ss))+1),np.uint8)
    def point(x,y):return (round((x+pad)*res*ss),round((y+pad)*res*ss))
    def line(x0,y0,x1,y1):cv2.line(a,point(x0,y0),point(x1,y1),255,max(2,round(stroke*res*ss)),cv2.LINE_AA)
    def rect(x0,y0,x1,y1):
        line(x0,y0,x1,y0);line(x1,y0,x1,y1);line(x1,y1,x0,y1);line(x0,y1,x0,y0)
    if kind=='produce':rect(0,0,w,h);line(0,0,w,h);line(w,0,0,h)
    elif kind=='self':rect(0,0,w,h);cv2.circle(a,point(1.5,1.5),round(.8*res*ss),255,max(2,round(stroke*res*ss)),cv2.LINE_AA)
    elif kind=='checkout':
        rect(0,0,2.5,12);rect(2.5,4.8,3.7,6);line(0,9,2.5,9);line(0,9,2.5,12);line(0,12,2.5,9);line(1.25,.5,1.25,8.5)
    elif kind=='cooler':rect(0,1.2,w,h);line(0,1.2,w,1.8)
    elif kind=='endcap':rect(0,0,w,h);line(0,1.5,w,0)
    elif kind.startswith('gondola'):rect(0,0,w,h);line(2,0,2,h)
    a=cv2.resize(a,(round(a.shape[1]/ss),round(a.shape[0]/ss)),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    a=cv2.GaussianBlur(a,(0,0),.4)
    return a,(w,h,pad)

def nms(candidates,min_dist=10,iou_thresh=1,score_key="score"):
    out=[]
    for c in sorted(candidates,key=lambda x:-x[score_key]):
        def overlap(d):
            ow=max(0,min(c['x']+c['w']/2,d['x']+d['w']/2)-max(c['x']-c['w']/2,d['x']-d['w']/2))
            oh=max(0,min(c['y']+c['h']/2,d['y']+d['h']/2)-max(c['y']-c['h']/2,d['y']-d['h']/2))
            area=ow*oh
            return area/max(1,c['w']*c['h']+d['w']*d['h']-area)
        if any((c['x']-d['x'])**2+(c['y']-d['y'])**2<min_dist**2 or overlap(d)>iou_thresh for d in out):continue
        out.append(c)
    return out

def detect(plan,kinds=None,threshold=.7,debug=None):
    if kinds is None:kinds=['produce','self','checkout','cooler','endcap','gondola4','gondola3']
    out={}
    # Suppress paper noise but preserve line gradients.
    src=cv2.GaussianBlur(plan,(0,0),.35)
    for kind in kinds:
        t,dim=template(kind);candidates=[]
        for rot in range(4):
            # Negative-handed checkout blocks are also used by some layouts.
            flips=[False,True] if kind in ['checkout','endcap','cooler'] else [False]
            for flip in flips:
                tt=np.rot90(np.fliplr(t) if flip else t,rot).copy()
                score=cv2.matchTemplate(src,tt,cv2.TM_CCOEFF_NORMED)
                peaks=(score==ndi.maximum_filter(score,size=11))&(score>threshold)
                ys,xs=np.nonzero(peaks)
                for y,x in zip(ys,xs):
                    xx,yy=float(x+tt.shape[1]/2),float(y+tt.shape[0]/2)
                    if kind=='cooler':
                        # Anchor at the door panel, which prevents opposite
                        # orientations from counting the same panel twice.
                        dx,dy=[(0,.6*R),(.6*R,0),(0,-.6*R),(-.6*R,0)][rot]
                        xx+=dx;yy+=dy
                    candidates.append(dict(x=xx,y=yy,score=float(score[y,x]),rotation=rot,flip=flip,w=tt.shape[1],h=tt.shape[0]))
        ds=nms(candidates,20 if kind in ['cooler','endcap'] else 26,.3 if kind=='checkout' else 1)
        out[kind]=ds
        if debug:
            a=cv2.cvtColor(np.uint8(255*(1-plan)),cv2.COLOR_GRAY2BGR)
            for d in ds:
                x,y=d['x'],d['y'];w,h=d['w'],d['h'];cv2.rectangle(a,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,0,255),2);cv2.putText(a,'%.2f'%d['score'],(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            cv2.imwrite(debug+'/'+kind+'.png',a)
    return out


def panel_kernel():
    t,_=template('cooler');h,w=t.shape;ss=4
    a=np.zeros((h*ss,w*ss),np.uint8)
    # Middle of the door's diagonal; omit corners and jambs.
    def point(x,y):return (round((x+.15)*R*ss),round((y+.15)*R*ss))
    cv2.line(a,point(.3,1.2+.3*.6/2.5),point(2.2,1.2+2.2*.6/2.5),255,round(.10*R*ss),cv2.LINE_AA)
    a=cv2.resize(a,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    a/=a.sum();blur=cv2.GaussianBlur(a,(0,0),1.7)
    return a-blur/blur.sum()

def verify_panels(src,candidates):
    k=panel_kernel()
    for d in candidates:
        rot,flip=d['rotation'],d['flip'];kk=np.rot90(np.fliplr(k) if flip else k,rot).copy()
        dx,dy=[(0,.6*R),(.6*R,0),(0,-.6*R),(-.6*R,0)][rot]
        x=round(d['x']-dx-d['w']/2);y=round(d['y']-dy-d['h']/2)
        sub=src[y:y+kk.shape[0],x:x+kk.shape[1]]
        d['diagonal']=float((sub*kk).sum())
    return candidates

def quantities(src,rect,debug=None,return_detections=False):
    raw=detect(src,threshold=.7)
    selected={}
    thresholds={'produce':.78,'self':.78,'checkout':.75,'endcap':.76,'gondola4':.80,'gondola3':.80}
    for kind in thresholds:
        selected[kind]=[dict(q,kind=kind) for q in raw[kind] if q['score']>=thresholds[kind]]
    selected['cooler']=[dict(q,kind='cooler') for q in verify_panels(src,raw['cooler']) if q['diagonal']>.045]
    # A shorter, partly matching section must not count again inside a complete
    # four-foot section (and vice versa).
    gondolas=nms(selected['gondola4']+selected['gondola3'],24,.4)
    selected['gondola4']=[q for q in gondolas if q['kind']=='gondola4']
    selected['gondola3']=[q for q in gondolas if q['kind']=='gondola3']
    for k in selected:
        selected[k]=[q for q in selected[k] if rect[0]<q['x']<rect[2] and rect[1]<q['y']<rect[3]]
    mapping={'gondola_sections_4ft':'gondola4','gondola_sections_3ft':'gondola3','endcaps':'endcap','checkouts':'checkout','self_checkouts':'self','cooler_doors':'cooler','produce_tables':'produce'}
    if debug:
        json.dump(selected,open(debug+'/fixtures.json','w'),indent=2)
        a=cv2.cvtColor(np.uint8((1-src)*255),cv2.COLOR_GRAY2BGR)
        for i,(k,ls) in enumerate(selected.items()):
            color=[(0,0,230),(0,160,0),(230,0,0),(150,0,150),(0,140,210),(180,100,0),(160,160,0)][i]
            for q in ls:
                x,y,w,h=q['x'],q['y'],q['w'],q['h'];cv2.rectangle(a,(round(x-w/2),round(y-h/2)),(round(x+w/2),round(y+h/2)),color,2)
        cv2.imwrite(debug+'/fixtures.png',a)
    result={k:len(selected[v]) for k,v in mapping.items()}
    return (result,selected) if return_detections else result


def wall_source(src,selected,rect,return_mask=False):
    """Remove fixture strokes before looking for parallel partition faces."""
    a=src.copy();mask=np.zeros(src.shape,np.uint8)
    for kind,ls in selected.items():
        for q in ls:
            x,y=q['x'],q['y']
            if kind=='cooler':
                dx,dy=[(0,.6*R),(.6*R,0),(0,-.6*R),(-.6*R,0)][q['rotation']]
                x-=dx;y-=dy
            w,h=q['w'],q['h']
            lo,hi=max(0,round(x-w/2)),min(src.shape[1],round(x+w/2)+1)
            top,bot=max(0,round(y-h/2)),min(src.shape[0],round(y+h/2)+1)
            mask[top:bot,lo:hi]=1
    # Keep the entire exterior wall even when a case backs onto it.
    mask[:round(rect[1]+R)]=0;mask[round(rect[3]-R):]=0
    mask[:,:round(rect[0]+R)]=0;mask[:,round(rect[2]-R):]=0
    a[mask!=0]=0
    return (a,mask) if return_mask else a
