"""Small-vocabulary recognition of the chain's boxed finish annotations."""
from pathlib import Path
import cv2,numpy as np,json,sys
from scipy import ndimage as ndi
from shapely.geometry import Point


def detect(src,debug=None):
    templates=np.load(Path(__file__).parent/'assets'/'finish_tags.npz')
    candidates=[]
    a=cv2.GaussianBlur(src,(0,0),.25)
    for code in templates.files:
        t=templates[code]
        for scale in [.94,1.,1.06]:
            tt=cv2.resize(t,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            scores=cv2.matchTemplate(a,tt,cv2.TM_CCOEFF_NORMED)
            ys,xs=np.nonzero((scores==ndi.maximum_filter(scores,size=9))&(scores>.59))
            for y,x in zip(ys,xs):
                candidates.append(dict(code=code,x=float(x+tt.shape[1]/2),y=float(y+tt.shape[0]/2),score=float(scores[y,x]),w=tt.shape[1],h=tt.shape[0]))
    out=[]
    for c in sorted(candidates,key=lambda c:-(c['score']+.04*max(0,len(c['code'])-4))):
        if any(abs(c['x']-q['x'])<max(c['w'],q['w'])*.65 and abs(c['y']-q['y'])<10 for q in out):continue
        out.append(c)
    if debug:
        a=cv2.cvtColor(np.uint8((1-src)*255),cv2.COLOR_GRAY2BGR)
        for c in [q for q in out if q['score']>=.78]:
            cv2.putText(a,c['code']+' %.2f'%c['score'],(round(c['x']),round(c['y'])),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
            cv2.circle(a,(round(c['x']),round(c['y'])),10,(0,0,255),1)
        cv2.imwrite(debug+'/tags.png',a);json.dump([q for q in out if q['score']>=.78],open(debug+'/tags.json','w'),indent=2)
    return out

def in_polygon(tags,poly):
    votes={}
    for t in tags:
        if t['score']<.78 or not poly.contains(Point(t['x'],t['y'])):continue
        # Full annotations get precedence over shorter partial string matches.
        votes[t['code']]=votes.get(t['code'],0)+max(0,(t['score']-.62))**2
    return votes

