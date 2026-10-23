"""Read the wall identifiers printed next to the construction swatches."""
from pathlib import Path
import cv2

ASSETS=Path(__file__).resolve().parent/'assets'


def wall_codes(im,s):
    H,W=im.shape
    a=im[int(.35*H):int(.50*H),int(.65*W):int(.89*W)]
    a=cv2.resize(a,None,fx=24/s,fy=24/s,interpolation=cv2.INTER_AREA if s>24 else cv2.INTER_LINEAR)
    rows=[]
    for code in ['P1','P2','P3']:
        t=cv2.imread(str(ASSETS/(code+'.png')),cv2.IMREAD_GRAYSCALE)
        corr=cv2.matchTemplate(a,t,cv2.TM_CCOEFF_NORMED)
        _,score,_,location=cv2.minMaxLoc(corr)
        if score<.70:raise ValueError('Unrecognized partition legend identifier '+code)
        rows.append((location[1],code))
    rows.sort()
    if any(abs(rows[i+1][0]-rows[i][0])<30 for i in range(2)):
        raise ValueError('Ambiguous partition legend')
    return dict(zip(['masonry','solid','thin'],[code for _,code in rows]))
