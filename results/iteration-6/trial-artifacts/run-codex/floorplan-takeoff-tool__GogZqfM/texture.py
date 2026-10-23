"""Phase-independent finish hatch recognition, sampled away from boundaries."""
import cv2
import numpy as np
from scipy.spatial.distance import cdist

SIZE=96
_y,_x=np.mgrid[-SIZE//2:SIZE//2,-SIZE//2:SIZE//2]
_freq=np.hypot(_x,_y)*24/SIZE
_angle=np.mod(np.arctan2(_y,_x),np.pi)
_edges=np.array([.18,.45,.75,1.15,1.55,2.1,2.8,3.8,5.5,8,12,17])
_bin=np.digitize(_freq,_edges)-1
_dir=(np.floor((_angle+np.pi/16)/(np.pi/8)).astype(int)%8)
_valid=(_bin>=0)&(_bin<len(_edges)-1)
_ids=(_bin*8+_dir)[_valid]
_window=np.outer(np.hanning(SIZE),np.hanning(SIZE)).astype(np.float32)


def features(patch):
    a=np.clip(patch.astype(np.float32),170,252)
    a=cv2.GaussianBlur(a,(0,0),.55)
    a-=cv2.GaussianBlur(a,(0,0),8)
    f=np.fft.fftshift(np.fft.fft2(a*_window))
    power=abs(f)**2/np.sum(_window**2)
    v=np.bincount(_ids,weights=power[_valid],minlength=(len(_edges)-1)*8)
    counts=np.bincount(_ids,minlength=len(v))
    v=v/np.maximum(counts,1)
    return np.log1p(v).astype(np.float32)


def sample_region(im,mask,s,max_samples=70):
    f=24/s
    if abs(f-1)>.001:
        im=cv2.resize(im,None,fx=f,fy=f,interpolation=cv2.INTER_AREA if f<1 else cv2.INTER_LINEAR)
        mask=cv2.resize(mask.astype(np.uint8),(im.shape[1],im.shape[0]),interpolation=cv2.INTER_NEAREST)
    # Require a complete 4 ft square from the floor region.
    ok=cv2.erode(mask.astype(np.uint8),np.ones((SIZE+2,SIZE+2),np.uint8))
    ys,xs=np.where(ok)
    if not len(xs):
        return np.zeros((0,(len(_edges)-1)*8),np.float32)
    rng=np.random.default_rng(72)
    ix=rng.choice(len(xs),size=min(max_samples*3,len(xs)),replace=False)
    res=[]
    for i in ix:
        x=int(xs[i]);y=int(ys[i]);p=im[y-SIZE//2:y+SIZE//2,x-SIZE//2:x+SIZE//2]
        if p.shape!=(SIZE,SIZE):continue
        if np.mean(p<140)>.02:continue
        res.append(features(p))
        if len(res)>=max_samples:break
    return np.array(res)


def classify(feat,train,labels):
    # Compare patch spectra in log space. Trim extremes due to symbols/fixtures.
    d=cdist(feat,train,'sqeuclidean')/train.shape[1]
    scores=[]
    codes=sorted(set(labels.tolist()))
    for code in codes:
        sub=d[:,labels==code]
        k=min(3,sub.shape[1])
        scores.append(np.mean(np.partition(sub,k-1,axis=1)[:,:k],axis=1))
    scores=np.stack(scores,axis=1)
    winners=np.argmin(scores,1)
    votes=np.bincount(winners,minlength=len(codes))
    return codes[np.argmax(votes)],{c:float(v/len(feat)) for c,v in zip(codes,votes)}
