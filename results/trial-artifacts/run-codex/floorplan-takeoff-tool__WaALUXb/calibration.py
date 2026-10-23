"""Scanner tone and pixel-aspect checks using printed calibration marks."""
import cv2
import numpy as np
from scipy import ndimage,signal


def normalize_tone(image,scale,bar):
    x,y,w,h=bar
    border=max(3,round(scale*.16))
    black=image[y+border:y+h-border,x+border:x+round(scale*4)-border]
    white=image[y+border:y+h-border,x+round(scale*20):x+round(scale*28)]
    if black.size==0 or white.size==0:return image,False
    low=float(np.median(black));high=float(np.median(white))
    if 18<=low<=36 and 235<=high<=253:return image,False
    if high-low<80:return image,False
    normalized=(image.astype(np.float32)-low)*(247-26)/(high-low)+26
    return np.rint(normalized).clip(0,255).astype(np.uint8),True


def grid_period(profile,estimate):
    a=ndimage.gaussian_filter1d(profile.astype(float),.45)
    peaks,prop=signal.find_peaks(-a,distance=.7*estimate,prominence=8)
    sub=[]
    for i in peaks:
        if i<2 or i>=len(a)-2:continue
        denominator=2*(a[i-1]-2*a[i]+a[i+1])
        offset=(a[i-1]-a[i+1])/denominator if denominator else 0
        sub.append(i+offset)
    sub=np.array(sub)
    if len(sub)<5:return None
    gaps=np.diff(sub)
    gaps=gaps[(gaps>.8*estimate)&(gaps<1.2*estimate)]
    if len(gaps)<4:return None
    pitch=float(np.median(gaps))
    indices=np.rint((sub-sub[0])/pitch)
    valid=np.abs(sub-sub[0]-indices*pitch)<.20*estimate
    if np.count_nonzero(valid)<5:return None
    coefficients=np.polyfit(indices[valid],sub[valid],1)
    residual=float(np.std(sub[valid]-np.polyval(coefficients,indices[valid])))
    if residual>.03*estimate:return None
    return float(coefficients[0]),len(sub)


def aspect_ratio(plan,regions,scale):
    """The legend's 12 by 12 inch VCT grid checks the two scanner axes."""
    ratios=[];weights=[]
    for region in regions:
        if region['finish']!='VCT-1':continue
        x,y,w,h=region['bounds']
        p=plan[y+4:y+h-4,x+4:x+w-4]
        if min(p.shape)<5*scale:continue
        sx=grid_period(np.median(p,axis=0),scale)
        sy=grid_period(np.median(p,axis=1),scale)
        if sx and sy:
            ratio=sy[0]/sx[0]
            if .85<ratio<1.15:
                ratios.append(ratio);weights.append(min(sx[1],sy[1]))
    if not ratios:return 1.
    ratio=float(np.median(ratios))
    # Multiple rooms must corroborate the correction, when available.
    inliers=[(r,w) for r,w in zip(ratios,weights) if abs(r-ratio)<.003]
    if not inliers:return 1.
    return float(np.average([r for r,w in inliers],weights=[w for r,w in inliers]))
