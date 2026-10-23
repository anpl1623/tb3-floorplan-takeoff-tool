"""Independent horizontal/vertical calibration from the 12-inch VCT grid."""
import cv2
import numpy as np
from scipy import ndimage
from floors import room_polygons,texture_features,classify_region
from walls import get_walls


def periodic_pitch(response,mask,axis):
    if axis:response=response.T;mask=mask.T
    count=mask.sum(axis=1)
    valid=count>count.max()*.5
    if valid.sum()<80:return None
    indices=np.flatnonzero(valid)
    first,last=indices[0],indices[-1]+1
    profile=(response*mask).sum(axis=1)/np.maximum(1,count)
    profile=profile[first:last]
    profile-=ndimage.gaussian_filter1d(profile,16)
    profile*=np.hanning(len(profile))
    spectrum=np.abs(np.fft.rfft(profile,n=32768))
    frequencies=np.fft.rfftfreq(32768)
    band=(frequencies>1/28)&(frequencies<1/20)
    peak=np.argmax(np.where(band,spectrum,0))
    if peak<=1 or peak>=len(spectrum)-2:return None
    before,at,after=spectrum[peak-1:peak+2]
    offset=.5*(before-after)/(before-2*at+after) if before-2*at+after else 0
    pitch=32768/(peak+offset)
    # Uniform grids put substantial power in the fundamental. Labels and an
    # occasional line alone do not provide a reliable aspect-ratio calibration.
    strength=float(at/(np.sqrt(np.mean(profile**2))*len(profile)+1e-8))
    if strength<.12:return None
    return float(pitch),strength


def calibrate_grid(padded,metadata):
    padding=metadata['padding']
    plan=padded[padding:-padding,padding:-padding] if padding else padded
    walls=get_walls(plan)
    polys=room_polygons(plan,walls)
    _,line=texture_features(plan)
    candidates=[]
    for poly in sorted(polys,key=lambda p:p.area,reverse=True):
        if poly.area<90:continue
        code,features=classify_region(plan,poly,line)
        hh,vv,dd,aa=features
        if code!='VCT-1' or max(hh,vv)>min(hh,vv)*1.7:continue
        x0,y0,x1,y1=np.round(np.array(poly.bounds)*24).astype(int)
        x0=max(0,x0);y0=max(0,y0);x1=min(plan.shape[1]-1,x1);y1=min(plan.shape[0]-1,y1)
        mask=np.zeros((y1-y0+1,x1-x0+1),np.uint8)
        xy=np.round(np.array(poly.exterior.coords)*24-[x0,y0]).astype('int32')
        cv2.fillPoly(mask,[xy],1)
        mask=cv2.erode(mask,np.ones((25,25),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        py=periodic_pitch(line[0][y0:y1+1,x0:x1+1],mask,0)
        px=periodic_pitch(line[1][y0:y1+1,x0:x1+1],mask,1)
        if px is not None and py is not None:
            candidates.append((px[0],py[0],poly.area))
        if len(candidates)>=3:break
    if not candidates:
        metadata['tile_grid']={'used':False,'reason':'no reliable 12-inch grid'}
        return padded,metadata
    # The largest room has the best frequency resolution. Other rooms guard
    # against an unrelated repeated symbol being mistaken for a tile grid.
    px,py,_=candidates[0]
    agreeing=[c for c in candidates if abs(c[0]/px-1)<.01 and abs(c[1]/py-1)<.01]
    weights=np.array([c[2]for c in agreeing])
    px=float(np.average([c[0]for c in agreeing],weights=weights))
    py=float(np.average([c[1]for c in agreeing],weights=weights))
    sx=24/px;sy=24/py
    metadata['tile_grid']={'used':False,'pitch_x':px,'pitch_y':py,'rooms':len(agreeing)}
    if max(abs(sx-1),abs(sy-1))<.0035:return padded,metadata
    if max(abs(sx-1),abs(sy-1))>.12:return padded,metadata
    height,width=plan.shape
    matrix=np.array([[sx,0,padding*(1-sx)],[0,sy,padding*(1-sy)]],float)
    new_width=round((width-1)*sx)+1+2*padding
    new_height=round((height-1)*sy)+1+2*padding
    corrected=cv2.warpAffine(padded,matrix,(new_width,new_height),borderValue=254)
    metadata['tile_grid'].update(used=True,correction_x=sx,correction_y=sy)
    metadata['size_ft']=[(new_width-2*padding-1)/24,(new_height-2*padding-1)/24]
    return corrected,metadata
