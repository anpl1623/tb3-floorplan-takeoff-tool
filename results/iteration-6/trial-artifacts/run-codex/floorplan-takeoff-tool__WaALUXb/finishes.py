"""Finish identification from scale-normalized hatch spectra.

The bundled references describe the CAD hatch textures, independent of the
store layout, names, room numbers, and takeoff totals.
"""
from pathlib import Path
import cv2
import numpy as np

FINISHES=['LVT-1','PC-1','SC-1','EP-1','VCT-1','CT-1','CPT-1','WOM-1']

def features(patches):
    # Four orientation wedges, and spatial frequencies in cycles per foot.
    n=patches.shape[-1]
    f=np.fft.fftfreq(n)
    yy,xx=np.meshgrid(f,f,indexing='ij')
    rr=np.hypot(xx,yy)
    theta=np.arctan2(yy,xx)%np.pi
    window=np.outer(np.hanning(n),np.hanning(n))
    arrays=[]
    for p in patches:
        blurred=cv2.GaussianBlur(p.astype(np.float32),(0,0),5)
        ink=np.clip(blurred-p,-25,25)
        spectrum=np.abs(np.fft.fft2(ink*window))**2/n**4
        vals=[]
        for angle in [0,np.pi/4,np.pi/2,3*np.pi/4]:
            dt=np.abs((theta-angle+np.pi/2)%np.pi-np.pi/2)
            wedge=dt<np.pi/10
            for low,high in zip([.015,.035,.055,.075,.10,.14,.20,.28],[.035,.055,.075,.10,.14,.20,.28,.45]):
                vals.append(spectrum[wedge&(rr>=low)&(rr<high)].sum())
        vals.append(np.mean(ink**2))
        vals.append(np.mean(np.abs(ink)))
        arrays.append(np.log1p(np.array(vals)))
    return np.array(arrays)

def get_patches(plan,region,scale):
    x,y,w,h=region['bounds']
    factor=20/scale
    p=cv2.resize(plan[y:y+h,x:x+w],None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    m=cv2.resize(region['mask'],(p.shape[1],p.shape[0]),interpolation=cv2.INTER_NEAREST)
    sz=64
    coords=[(xx,yy) for yy in range(0,p.shape[0]-sz+1,24) for xx in range(0,p.shape[1]-sz+1,24) if np.mean(m[yy:yy+sz,xx:xx+sz])>.96]
    if len(coords)>240:
        coords=[coords[i] for i in np.linspace(0,len(coords)-1,240).astype(int)]
    if not coords:
        coords=[((p.shape[1]-sz)//2,(p.shape[0]-sz)//2)]
    return np.array([p[yy:yy+sz,xx:xx+sz] for xx,yy in coords])


def classify(plan, regions, scale, assets):
    with np.load(Path(assets)/'finish_features.npz') as model:
        prototypes=model['features'];labels=model['labels'];std=model['std']
    totals={code:0. for code in FINISHES}
    for region in regions:
        patches=get_patches(plan,region,scale)
        vector=np.median(features(patches),axis=0)
        distances=np.mean(((prototypes-vector)/std)**2,axis=1)
        best=int(np.argmin(distances))
        code=FINISHES[int(labels[best])]
        region['finish']=code
        region['texture_distance']=float(distances[best])
        totals[code]+=region['polygon'].area
    return totals
