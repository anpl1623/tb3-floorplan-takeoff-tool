"""Recognition of the chain's fixed CAD blocks, in physical coordinates."""
import cv2
import numpy as np
from scipy import ndimage


def template(kind,p=20):
    dims={'produce_tables':(5,4),'self_checkouts':(3,3),'cooler_doors':(2.5,3),
          'endcaps':(4,1.5),'gondola_sections_4ft':(4,4),'gondola_sections_3ft':(4,3),
          'checkouts':(2.5,12),'single_leaf':(3.2,3.2)}
    w,h=dims[kind];pad=.15;scale=4
    q=p*scale
    im=np.zeros((round((h+2*pad)*q)+1,round((w+2*pad)*q)+1),np.uint8)
    def pt(x,y):return (round((x+pad)*q),round((y+pad)*q))
    def line(x1,y1,x2,y2,th=.045):cv2.line(im,pt(x1,y1),pt(x2,y2),255,max(1,round(th*q)),cv2.LINE_AA)
    def rect(x0,y0,x1,y1):
        line(x0,y0,x1,y0);line(x1,y0,x1,y1);line(x1,y1,x0,y1);line(x0,y1,x0,y0)
    if kind!='single_leaf':rect(0,0,w,h)
    if kind=='produce_tables':line(0,0,w,h);line(w,0,0,h)
    elif kind=='self_checkouts':cv2.circle(im,pt(1.5,1.5),round(.8*q),255,max(1,round(.045*q)),cv2.LINE_AA)
    elif kind=='cooler_doors':line(0,2.4,2.5,2.4);line(0,2.4,2.5,3)
    elif kind=='endcaps':line(0,1.5,4,0)
    elif kind.startswith('gondola_'):line(2,0,2,h)
    elif kind=='checkouts':
        line(0,9,2.5,9);line(0,9,2.5,12);line(2.5,9,0,12)
        line(1.25,.5,1.25,8.5)
        # Cashier's projection lies on the right side.
        # Add canvas space for the projection.
        out=np.zeros((im.shape[0],im.shape[1]+round(1.2*q)),np.uint8);out[:,:im.shape[1]]=im;im=out
        line(2.5,4.8,3.7,4.8);line(3.7,4.8,3.7,6);line(3.7,6,2.5,6)
    elif kind=='single_leaf':
        # Hinge at the lower left; open leaf rises vertically, with the quarter
        # circle sweeping down to the wall opening at its lower right.
        line(.1,3.1,.1,.1);line(.23,3.1,.23,.1);line(.1,.1,.23,.1)
        cv2.ellipse(im,pt(.1,3.1),(round(3*q),round(3*q)),0,270,360,255,max(1,round(.045*q)),cv2.LINE_AA)
    return cv2.resize(im.astype(np.float32)/255,(round(im.shape[1]/scale),round(im.shape[0]/scale)),interpolation=cv2.INTER_AREA)


def match_symbols(a,ppf,kind,threshold=.65,debug=None):
    p=20;f=p/ppf
    work=cv2.resize(a,None,fx=f,fy=f,interpolation=cv2.INTER_AREA)
    work=cv2.GaussianBlur(work,(0,0),.4)
    t=template(kind,p)
    variants=[]
    for rot in range(4):
        variants.append((np.rot90(t,rot).copy(),rot,False))
        if kind in ['single_leaf','endcaps','cooler_doors','checkouts']:
            variants.append((np.fliplr(np.rot90(t,rot)).copy(),rot,True))
    hits=[]
    for tmpl,rot,flip in variants:
        result=cv2.matchTemplate(work,tmpl,cv2.TM_CCOEFF_NORMED)
        maxima=(result==cv2.dilate(result,np.ones((11,11),np.uint8)))&(result>threshold)
        ys,xs=np.where(maxima)
        h,w=tmpl.shape
        for x,y in zip(xs,ys):hits.append(dict(x=float((x+w/2)/f),y=float((y+h/2)/f),score=float(result[y,x]),rotation=rot,flip=flip,w=w/f,h=h/f,kind=kind))
    hits.sort(key=lambda q:-q['score']);selected=[]
    dims={'produce_tables':2,'self_checkouts':1.5,'cooler_doors':1.2,'endcaps':1,'gondola_sections_4ft':1.8,'gondola_sections_3ft':1.3,'checkouts':3,'single_leaf':1.3}
    for hit in hits:
        if all(np.hypot(hit['x']-q['x'],hit['y']-q['y'])>ppf*dims[kind] for q in selected):selected.append(hit)
    if debug:
        out=cv2.cvtColor((255*(1-a)).clip(0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        for q in selected:
            cx,cy=round(q['x']),round(q['y']);w,h=round(q['w']/2),round(q['h']/2)
            cv2.rectangle(out,(cx-w,cy-h),(cx+w,cy+h),(0,0,255),2)
            cv2.putText(out,str(round(q['score'],2)),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1,cv2.LINE_AA)
        cv2.imwrite(str(debug)+'-'+kind+'.png',out)
    return selected
