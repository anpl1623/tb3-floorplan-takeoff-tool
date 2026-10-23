"""Read the numeric divisions of a graphic scale using a tiny digit vocabulary."""
from pathlib import Path
import cv2,numpy as np


def glyph(p):
    p=np.maximum(p-.035,0)
    if not p.size or p.max()<.1:return None
    yy,xx=np.nonzero(p>max(.1,float(p.max())*.27))
    if not len(xx):return None
    p=p[yy.min():yy.max()+1,xx.min():xx.max()+1]
    p=cv2.copyMakeBorder(p,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    p=cv2.resize(p,(18,26),interpolation=cv2.INTER_LINEAR)
    p=p-p.mean();norm=np.linalg.norm(p)
    return p/max(norm,1e-8)


def read_number(ink,x,y,bar_height,templates):
    height=bar_height
    x0=max(0,round(x-height*1.8));x1=min(ink.shape[1],round(x+height*1.8)+1)
    y0=max(0,round(y+height*.25));y1=min(ink.shape[0],round(y+height*2.4))
    crop=ink[y0:y1,x0:x1]
    if crop.size==0:return None
    mask=np.uint8(crop>.35)
    n,lab,stats,centers=cv2.connectedComponentsWithStats(mask)
    boxes=[s for s in stats[1:] if max(4,height*.22)<s[3]<height*1.3 and s[2]>=2 and s[4]>=s[3]*1.5]
    if not boxes:return None
    # Keep one text baseline rather than stray dust or an annotation above it.
    centers_y=np.array([s[1]+s[3]/2 for s in boxes]);target=np.median(centers_y)
    boxes=[s for s in boxes if abs(s[1]+s[3]/2-target)<height*.3]
    boxes.sort(key=lambda s:s[0])
    if not 1<=len(boxes)<=3:return None
    text='';confidence=[]
    for xx,yy,w,h,area in boxes:
        p=crop[max(0,yy-1):yy+h+1,max(0,xx-1):xx+w+1]
        f=glyph(p)
        if f is None:return None
        scores=sorted([(max(float(np.sum(f*t)) for t in variants),c) for c,variants in templates.items()],reverse=True)
        score,char=scores[0]
        if score<.72 or score-scores[1][0]<.025:return None
        text+=char;confidence.append(score)
    return int(text),float(min(confidence))


def endpoint_units(ink,x0,bar_y1,bar_height,assumed_ppf):
    path=Path(__file__).parent/'assets'/'scale_digits.npz'
    raw=np.load(path);templates={}
    for key in raw.files:templates.setdefault(key.split('_')[0],[]).append(glyph(raw[key]))
    votes={};details=[]
    for division,multiplier in [(4,8),(8,4),(16,2),(32,1)]:
        reading=read_number(ink,x0+division*assumed_ppf,bar_y1,bar_height,templates)
        if reading is None:continue
        value,confidence=reading;end=value*multiplier
        if not 4<=end<=500 or confidence<.78:continue
        votes.setdefault(end,[]).append(confidence)
        details.append((division,value,confidence))
    supported=[(sum(scores),value) for value,scores in votes.items() if len(scores)>=2]
    if not supported:return 32,details
    return max(supported)[1],details
