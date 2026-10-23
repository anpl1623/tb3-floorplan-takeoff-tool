"""Count the architect's fixture symbols from their physical geometry."""
from pathlib import Path
import cv2
import numpy as np

ASSETS=Path(__file__).resolve().parent/'assets'


def _add(items,pos,box,s,radius=.65):
    if not any(np.hypot(p[0]-pos[0],p[1]-pos[1])<s*radius for p,b in items):
        items.append((tuple(map(float,pos)),tuple(map(int,box))))


def _module_counts(cells,s):
    # Each two-foot-deep rectangle is one face of a gondola module. Match
    # touching backs, and count that double-sided module only once.
    used=set();counts={3:0,4:0}
    for i,(pos,box) in enumerate(cells):
        if i in used:continue
        x,y,w,h=box
        vertical=h>w
        length=round((h if vertical else w)/s)
        choices=[]
        for j,(q,b) in enumerate(cells):
            if j==i or j in used:continue
            xx,yy,ww,hh=b
            if (hh>ww)!=vertical:continue
            if round((hh if vertical else ww)/s)!=length:continue
            if vertical:
                error=abs(abs(q[0]-pos[0])-s*2)+abs(q[1]-pos[1])
            else:
                error=abs(abs(q[1]-pos[1])-s*2)+abs(q[0]-pos[0])
            if error<s*.35:choices.append((error,j))
        used.add(i)
        if choices:used.add(min(choices)[1])
        counts[length]+=1
    return counts


def count_fixtures(im,s):
    found={k:[] for k in ['endcaps','produce_tables','checkouts','self_checkouts']}
    cells=[]
    for threshold in [180,190,200,210,220,230]:
        b=(im<threshold).astype(np.uint8)
        b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(b,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if w<8 or h<8:continue
            wf=w/s;hf=h/s
            if max(wf,hf)>10 or min(wf,hf)<.8:continue
            area=cv2.contourArea(c);ratio=area/(w*h)
            per=cv2.arcLength(c,True)
            box=(x,y,w,h);pos=(x+w/2,y+h/2)
            # Empty 2 x 3 or 2 x 4 ft cells. This white interior check rejects
            # fortuitously joined squares in the VCT floor hatch.
            if ratio>.88 and ((1.7<wf<2.15 and (2.7<hf<3.2 or 3.7<hf<4.2)) or (1.7<hf<2.15 and (2.7<wf<3.2 or 3.7<wf<4.2))):
                q=im[y+5:y+h-5,x+5:x+w-5]
                if q.size and np.mean(q<230)<.03:
                    _add(cells,pos,box,s,.3)
            # Checkout body: 2 ft 6 in wide, 9 ft long, followed by its 3 ft
            # bagging symbol. Counting the body avoids confusing the X with
            # triangular fragments at gondola endcaps.
            if ratio>.85 and ((2.2<wf<2.8 and 8.7<hf<9.3) or (2.2<hf<2.8 and 8.7<wf<9.3)):
                _add(found['checkouts'],pos,box,s)
            if 1.4<wf<1.85 and 1.4<hf<1.85 and .70<ratio<.88 and 4*np.pi*area/per**2>.80:
                _add(found['self_checkouts'],pos,box,s)
            if not .35<ratio<.60:continue
            approx=cv2.approxPolyDP(c,per*.03,True)
            if len(approx)!=3:continue
            dims=sorted([wf,hf])
            if 1.2<dims[0]<1.7 and 3.2<dims[1]<4.2:
                _add(found['endcaps'],pos,box,s,.8)
            # The four triangular openings of a produce table all point at
            # the same center. The longest side is the triangle's baseline.
            if ((2.25<wf<2.75 and 3.7<hf<4.2) or (4.7<wf<5.2 and 1.7<hf<2.2)
                or (2.25<hf<2.75 and 3.7<wf<4.2) or (4.7<hf<5.2 and 1.7<wf<2.2)):
                pts=approx[:,0,:]
                lens=[np.linalg.norm(pts[(j+1)%3]-pts[j]) for j in range(3)]
                apex=pts[(int(np.argmax(lens))+2)%3]
                _add(found['produce_tables'],apex,box,s,.8)
    # Endcaps belong at the ends of gondola modules, rather than elsewhere in
    # the drawing's triangular linework.
    caps=[]
    for pos,box in found['endcaps']:
        x,y,w,h=box;vertical=w>h
        for q,b in cells:
            xx,yy,ww,hh=b
            if (hh>ww)!=vertical:continue
            if vertical:
                adjacent=abs(q[0]-pos[0])<1.5*s and min(abs(pos[1]-yy),abs(pos[1]-(yy+hh)))<1.25*s
            else:
                adjacent=abs(q[1]-pos[1])<1.5*s and min(abs(pos[0]-xx),abs(pos[0]-(xx+ww)))<1.25*s
            if adjacent:
                caps.append((pos,box));break
    found['endcaps']=caps
    modules=_module_counts(cells,s)
    coolers=count_coolers(im,s)
    counts=dict(gondola_sections_4ft=modules[4],gondola_sections_3ft=modules[3],
                endcaps=len(found['endcaps']),checkouts=len(found['checkouts']),
                self_checkouts=len(found['self_checkouts']),cooler_doors=len(coolers),
                produce_tables=len(found['produce_tables']))
    debug=dict(found,cells=cells,cooler_doors=coolers)
    return counts,debug


def count_coolers(im,s):
    template=cv2.imread(str(ASSETS/'cooler.png'),cv2.IMREAD_GRAYSCALE)
    a=cv2.resize(im,None,fx=24/s,fy=24/s,interpolation=cv2.INTER_AREA if s>24 else cv2.INTER_LINEAR)
    matches=[]
    for k in range(4):
        t=np.rot90(template,k).copy()
        corr=cv2.matchTemplate(a,t,cv2.TM_CCOEFF_NORMED)
        peaks=(corr==cv2.dilate(corr,np.ones((15,15),np.uint8)))&(corr>.80)
        ys,xs=np.where(peaks)
        for y,x in zip(ys,xs):
            matches.append((float(corr[y,x]),(x+t.shape[1]/2)*s/24,(y+t.shape[0]/2)*s/24,k))
    kept=[]
    for score,x,y,k in sorted(matches,reverse=True):
        if not any(np.hypot(q[0]-x,q[1]-y)<s*.7 for q in kept):
            kept.append((float(x),float(y),float(score),int(k)))
    return kept
