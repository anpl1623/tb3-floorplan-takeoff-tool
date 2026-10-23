import cv2,numpy as np,json,math,sys
from scipy import ndimage as ndi
from blocks import nms
R=16

def swing_template(pair=False,leaf=3,stroke=.045):
    w=leaf*2 if pair else leaf;h=leaf;pad=.15;ss=4
    a=np.zeros((round((h+2*pad)*R*ss)+1,round((w+2*pad)*R*ss)+1),np.uint8)
    def pt(x,y):return round((x+pad)*R*ss),round((y+pad)*R*ss)
    thick=max(2,round(stroke*R*ss))
    def line(x,y,u,v):cv2.line(a,pt(x,y),pt(u,v),255,thick,cv2.LINE_AA)
    def leaf_at(x,right=False):
        dx=-.12 if right else .12
        line(x,0,x,h);line(x+dx,0,x+dx,h);line(x,0,x+dx,0);line(x,h,x+dx,h)
    leaf_at(0)
    cv2.ellipse(a,pt(0,h),(round(leaf*R*ss),round(leaf*R*ss)),0,270,360,255,thick,cv2.LINE_AA)
    if pair:
        leaf_at(w,True)
        cv2.ellipse(a,pt(w,h),(round(leaf*R*ss),round(leaf*R*ss)),0,180,270,255,thick,cv2.LINE_AA)
    a=cv2.resize(a,(round(a.shape[1]/ss),round(a.shape[0]/ss)),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    return cv2.GaussianBlur(a,(0,0),.4)

def detect_swings(src,walls,debug=None):
    out={};img=cv2.GaussianBlur(src,(0,0),.35)
    for kind in ['pair','single']:
        cs=[]
        for leaf in [2.5,8/3,3.,3.5,4.]:
            t=swing_template(kind=='pair',leaf=leaf)
            for rot in range(4):
                for flip in ([False] if kind=='pair' else [False,True]):
                    tt=np.rot90(np.fliplr(t) if flip else t,rot).copy()
                    scores=cv2.matchTemplate(img,tt,cv2.TM_CCOEFF_NORMED)
                    peaks=(scores==ndi.maximum_filter(scores,size=13))&(scores>.25)
                    ys,xs=np.nonzero(peaks)
                    radius=leaf*R/2
                    dx,dy=[(0,radius),(radius,0),(0,-radius),(-radius,0)][rot]
                    for y,x in zip(ys,xs):
                        xx,yy=x+tt.shape[1]/2,y+tt.shape[0]/2;bx,by=xx+dx,yy+dy
                        axis='h' if rot%2==0 else 'v';wc=by if axis=='h' else bx;sc=bx if axis=='h' else by
                        near=[w for w in walls if w['axis']==axis and abs(wc-w['c'])<R*.6 and w['start']-R<=sc<=w['end']+R]
                        if not near:continue
                        cs.append(dict(x=float(xx),y=float(yy),bx=float(bx),by=float(by),rotation=rot,flip=flip,leaf=leaf,w=tt.shape[1],h=tt.shape[0],score=float(scores[y,x])))
        scored=enrich(src,walls,{kind:cs})[kind]
        cs=[q for q in scored if q['arc']>.065 and q['clear']<.14]
        for q in cs:q['rank']=q['score']+2*q['arc']
        ds=nms(cs,26,.3,score_key='rank')
        out[kind]=ds
        if debug:
            a=cv2.cvtColor(np.uint8((1-src)*255),cv2.COLOR_GRAY2BGR)
            for d in ds:
                x,y=d['x'],d['y'];w,h=d['w'],d['h'];cv2.rectangle(a,(round(x-w/2),round(y-h/2)),(round(x+w/2),round(y+h/2)),(0,0,255),2);cv2.putText(a,'%.2f'%d['score'],(round(x),round(y)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            cv2.imwrite(debug+'/doors-'+kind+'.png',a)
    return out


def arc_kernels(pair=False,leaf=3.):
    ss=4;pad=.15;w=leaf*2 if pair else leaf;h=leaf
    kernels=[]
    for right in ([False,True] if pair else [False]):
        a=np.zeros((round((h+2*pad)*R*ss)+1,round((w+2*pad)*R*ss)+1),np.uint8)
        center=(round(((w if right else 0)+pad)*R*ss),round((h+pad)*R*ss))
        cv2.ellipse(a,center,(round(leaf*R*ss),round(leaf*R*ss)),0,192 if right else 282,258 if right else 348,255,round(.10*R*ss),cv2.LINE_AA)
        a=cv2.resize(a,(round(a.shape[1]/ss),round(a.shape[0]/ss)),interpolation=cv2.INTER_AREA).astype(np.float32)/255
        a/=a.sum();blur=cv2.GaussianBlur(a,(0,0),2.5);k=a-blur/blur.sum();kernels.append(k)
    return kernels

def enrich(src,walls,ds):
    for kind,ls in ds.items():
        cache={}
        for d in ls:
            leaf=d.get('leaf',3.)
            if leaf not in cache:cache[leaf]=arc_kernels(kind=='pair',leaf)
            kernels=cache[leaf]
            rot,flip=d['rotation'],d['flip'];ks=[np.rot90(np.fliplr(k) if flip else k,rot).copy() for k in kernels]
            x=round(d['x']-d['w']/2);y=round(d['y']-d['h']/2)
            scores=[]
            for k in ks:
                sub=src[y:y+k.shape[0],x:x+k.shape[1]]
                scores.append(float((sub*k).sum()))
            d['arc']=min(scores)
            axis='h' if rot%2==0 else 'v';wc=d['by'] if axis=='h' else d['bx'];sc=d['bx'] if axis=='h' else d['by']
            near=[w for w in walls if w['axis']==axis and abs(wc-w['c'])<R*.6 and w['start']-R<=sc<=w['end']+R]
            wall=min(near,key=lambda w:abs(w['c']-wc));a=src if axis=='h' else src.T
            c=wall['c'];r=R*leaf*(.8 if kind=='pair' else 1.1/3);p=a[round(c)-1:round(c)+2,round(sc-r):round(sc+r)+1]
            d['clear']=float(p.mean());d['wallkind']=wall['kind']
    return ds


def detect_sliding(src,walls,debug=None):
    # The two sliding leaves overlap at the center and lie on opposite sides
    # of the opening centerline. Their opposite gray contrasts distinguish the
    # block from both a plain opening and a solid wall of any hatch.
    cs=[]
    for axis in ['h','v']:
        a=src if axis=='h' else src.T
        mean=cv2.boxFilter(a,-1,(39,1),normalize=True)
        for wall in walls:
            if wall['axis']!=axis:continue
            c=round(wall['c']);s=round(wall['start']);e=round(wall['end'])
            for cc in range(c-2,c+3):
                if cc<3 or cc+3>=a.shape[0]:continue
                upper=mean[cc-2:cc].mean(axis=0)
                lower=mean[cc+1:cc+3].mean(axis=0)
                diff=upper-lower
                left=np.roll(diff,24);right=np.roll(diff,-24)
                score=np.maximum(np.minimum(left,-right),np.minimum(-left,right))
                lo=max(48,s+10);hi=min(len(score)-48,e-10)
                peaks=np.flatnonzero((score==ndi.maximum_filter1d(score,21))&(score>.28)&(np.arange(len(score))>=lo)&(np.arange(len(score))<hi))
                for pos in peaks:
                    x,y=(pos,cc) if axis=='h' else (cc,pos)
                    jamb_profile=a[max(0,c-3):c+4].mean(axis=0)
                    jambs=[]
                    for end in [pos-48,pos+48]:
                        ix=np.arange(max(4,end-6),min(len(jamb_profile)-4,end+7))
                        value=jamb_profile[ix]-(jamb_profile[ix-4]+jamb_profile[ix+4])*.5
                        jambs.append(float(value.max()) if len(value) else 0.)
                    if min(jambs)<.16:continue
                    cs.append(dict(x=float(x),y=float(y),bx=float(x),by=float(y),axis=axis,score=float(score[pos]),jambs=jambs,w=98 if axis=='h' else 14,h=14 if axis=='h' else 98))
    ds=nms(cs,64,.2)
    if debug:
        json.dump(ds,open(debug+'/sliding.json','w'),indent=2)
        a=cv2.cvtColor(np.uint8((1-src)*255),cv2.COLOR_GRAY2BGR)
        for d in ds:cv2.circle(a,(round(d['bx']),round(d['by'])),30,(0,0,255),2)
        cv2.imwrite(debug+'/doors-sliding.png',a)
    return ds

def detect_coiling(src,walls,sliding=(),debug=None):
    ext=[]
    for q in sorted([q for q in walls if q['kind']=='EXT'],key=lambda q:(q['axis'],q['c'],q['start'])):
        if ext and ext[-1]['axis']==q['axis'] and abs(ext[-1]['c']-q['c'])<1 and q['start']-ext[-1]['end']<2:ext[-1]['end']=q['end']
        else:ext.append(dict(q))
    ds=[]
    for q in ext:
        a=src if q['axis']=='h' else src.T;c=round(q['c']);s,e=round(q['start']),round(q['end'])
        # Sample on either side of the coiling door's dashed center line.
        profile=a[[c-3,c+3],s:e].mean(axis=0)
        sm=cv2.blur(profile[None,:],(17,1))[0];mu=np.median(sm)
        low=sm<mu*.6;edges=np.flatnonzero(np.diff(np.r_[False,low,False]))
        for i,j in zip(edges[::2],edges[1::2]):
            length=(j-i)/R
            if not 7.5<length<24:continue
            center=s+(i+j)/2;x,y=(center,q['c']) if q['axis']=='h' else (q['c'],center)
            if any((x-v['bx'])**2+(y-v['by'])**2<(5*R)**2 for v in sliding):continue
            ds.append(dict(x=float(x),y=float(y),axis=q['axis'],span_ft=float(length)))
    if debug:json.dump(ds,open(debug+'/coiling.json','w'),indent=2)
    return ds

def quantities(src,walls,debug=None):
    swings=detect_swings(src,walls);pairs=swings['pair'];singles=[]
    for s in swings['single']:
        if any(s['rotation']%2==p['rotation']%2 and (s['bx']-p['bx'])**2+(s['by']-p['by'])**2<((p.get('leaf',3.)+.4)*R)**2 for p in pairs):continue
        singles.append(s)
    sliding=detect_sliding(src,walls,debug);coiling=detect_coiling(src,walls,sliding,debug)
    if debug:
        accepted={'single':singles,'pair':pairs,'overhead_coiling':coiling,'auto_sliding':sliding}
        json.dump(accepted,open(debug+'/doors.json','w'),indent=2)
        a=cv2.cvtColor(np.uint8((1-src)*255),cv2.COLOR_GRAY2BGR)
        colors=[(0,0,220),(0,140,0),(0,130,230),(220,0,0)]
        for (kind,ls),color in zip(accepted.items(),colors):
            for i,q in enumerate(ls):
                x,y=q['x'],q['y']
                if 'w' in q:w,h=q['w'],q['h']
                else:w,h=(q['span_ft']*R,14) if q['axis']=='h' else (14,q['span_ft']*R)
                cv2.rectangle(a,(round(x-w/2),round(y-h/2)),(round(x+w/2),round(y+h/2)),color,2)
                cv2.putText(a,kind+' '+str(i+1),(round(x),round(y)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        cv2.imwrite(debug+'/doors.png',a)
    return {'single':len(singles),'pair':len(pairs),'overhead_coiling':len(coiling),'auto_sliding':len(sliding)}
