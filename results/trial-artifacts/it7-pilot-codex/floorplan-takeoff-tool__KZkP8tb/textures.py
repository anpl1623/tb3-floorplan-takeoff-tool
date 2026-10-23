"""Texture recognition calibrated on the current sheet's legend."""
import numpy as np,cv2
from scipy import ndimage as ndi

N=48
# Spatial frequency bands and direction sectors in the raster coordinate frame.
fy,fx=np.mgrid[-N//2:N//2,-N//2:N//2]/N
rad=np.hypot(fx,fy);ang=np.mod(np.arctan2(fy,fx),np.pi)
bands=[.015,.045,.075,.115,.17,.24,.34,.5,.72]
filters=[]
for lo,hi in zip(bands[:-1],bands[1:]):
 for j in range(4):
  delta=np.abs((ang-j*np.pi/4+np.pi/2)%np.pi-np.pi/2)
  m=(rad>=lo)&(rad<hi)&(delta<=np.pi/8)
  filters.append(m.astype(float)/max(1,m.sum()))
filters=np.array(filters).reshape(-1,N*N)
window=np.outer(np.hanning(N),np.hanning(N))

def features(patches):
    a=np.array(patches,dtype=np.float32)
    # Every feature is calculated from the gray ink, retaining faint hatches.
    means=a.mean(axis=(1,2));std=a.std(axis=(1,2))
    q=np.percentile(a,[10,50,90],axis=(1,2)).T
    centered=(a-a.mean(axis=(1,2),keepdims=True))*window
    p=np.abs(np.fft.fftshift(np.fft.fft2(centered),axes=(1,2)))**2
    p=p.reshape(-1,N*N)@filters.T
    p=np.sqrt(p)/N
    dx=np.mean(np.abs(np.diff(a,axis=2)),axis=(1,2));dy=np.mean(np.abs(np.diff(a,axis=1)),axis=(1,2))
    return np.column_stack([means,std,q,dx,dy,p]).astype(np.float32)

class TextureModel:
    def __init__(self,legend):
        ps=[];labels=[]
        for k,p in enumerate(legend['floors']):
            p=p[3:-3,3:-3]
            if min(p.shape)<N:p=cv2.copyMakeBorder(p,0,max(0,N-p.shape[0]),0,max(0,N-p.shape[1]),cv2.BORDER_REFLECT_101)
            for y in range(0,p.shape[0]-N+1,4):
                for x in range(0,p.shape[1]-N+1,6):
                    patch=p[y:y+N,x:x+N]
                    for factor in [.85,1,1.15]:
                        ps.append(np.clip(patch*factor,0,1));labels.append(k)
        self.freqs=dominant_frequencies(legend)
        self.refs=np.column_stack([features(ps),coherent_features(ps,self.freqs)]);self.labels=np.array(labels)
        # Relative energy across frequency bands is more robust than absolute
        # contrast, but blank and solid fills also need their gray levels.
        self.scale=np.maximum(np.std(self.refs,axis=0),np.r_[np.full(7,.015),np.full(self.refs.shape[1]-7,.003)])
        self.scale[:7]*=1.8
        self.scale[7:39]*=3
        self.scale[39:]=np.maximum(self.scale[39:],.04)
        self.refnorm=self.refs/self.scale
        # A coarse concrete joint grid leaves large blank floor patches. The
        # legend can be smaller than one grid cell, so sample its clear paper
        # to represent patches without a visible joint.
        p=legend['floors'][1][4:-4,4:-4];clear=[]
        for y in range(0,p.shape[0]-24+1,6):
            for x in range(0,p.shape[1]-24+1,6):
                q=p[y:y+24,x:x+24]
                if q.mean()<.08 and q.std()<.035:clear.append((float(q.mean()+q.std()),q))
        ps=[]
        for _,q in sorted(clear,key=lambda q:q[0])[:3]:
            q=cv2.copyMakeBorder(q,12,12,12,12,cv2.BORDER_REFLECT_101)
            for factor in [.85,1,1.15]:ps.append(q*factor)
        if ps:
            extra=np.column_stack([features(ps),coherent_features(ps,self.freqs)])
            self.refs=np.concatenate([self.refs,extra]);self.labels=np.r_[self.labels,np.full(len(extra),1)]
            self.refnorm=self.refs/self.scale
    def predict(self,patches):
        fs=np.column_stack([features(patches),coherent_features(patches,self.freqs)]);out=[];all_dist=[]
        for f in fs/self.scale:
            ds=np.sum((self.refnorm-f)**2,axis=1)
            dist=np.array([np.mean(np.sort(ds[self.labels==k])[:3]) for k in range(8)])
            out.append(int(np.argmin(dist)));all_dist.append(dist)
        return np.array(out),np.array(all_dist)

    def classify_polygon(self,src,poly,occlusions=None):
        from shapely.geometry import Point
        x0,y0,x1,y1=map(int,poly.bounds);ps=[];coords=[];best_covered=None
        for y in range(y0+N//2+3,y1-N//2-1,12):
            for x in range(x0+N//2+3,x1-N//2-1,12):
                if poly.contains(Point(x,y)) and poly.boundary.distance(Point(x,y))>N/2+1:
                    patch=src[y-N//2:y+N//2,x-N//2:x+N//2]
                    if occlusions is not None:
                        covered=float(occlusions[y-N//2:y+N//2,x-N//2:x+N//2].mean())
                        if covered>.2:
                            if best_covered is None or covered<best_covered[0]:best_covered=(covered,(x,y),patch)
                            continue
                    ps.append(patch);coords.append((x,y))
        if not ps:
            if best_covered is not None:
                _,point,patch=best_covered;ps=[patch];coords=[point]
            else:
                pt=poly.representative_point();x,y=round(pt.x),round(pt.y);ps=[src[y-N//2:y+N//2,x-N//2:x+N//2]];coords=[(x,y)]
        if len(ps)>500:
            ix=np.linspace(0,len(ps)-1,500,dtype=int);ps=[ps[i] for i in ix];coords=[coords[i] for i in ix]
        # Text and room-number tags are much darker than the floor hatches.
        # Select clean samples across the whole region, including its corners.
        bad=np.array([(p>.6).mean() for p in ps])
        good=bad<=max(.006,float(np.percentile(bad,25))*1.8)
        if good.sum()>=3:
            ps=[p for p,g in zip(ps,good) if g];coords=[p for p,g in zip(coords,good) if g]
        labels,dist=self.predict(ps)
        votes=np.bincount(labels,minlength=8)
        return int(np.argmax(votes)),votes,coords,labels,dist

def dominant_frequencies(legend):
    freqs=[]
    for patch in legend['floors']:
        a=patch[3:-3,3:-3].astype(float);h,w=a.shape
        a-=np.mean(a);win=np.outer(np.hanning(h),np.hanning(w));power=np.abs(np.fft.fftshift(np.fft.fft2(a*win,s=(256,256))))**2
        yy,xx=np.mgrid[-128:128,-128:128]/256;rad=np.hypot(xx,yy)
        power[(rad<.03)|(rad>.43)|(yy<-.001)]=0
        peaks=(power==ndi.maximum_filter(power,size=9));ys,xs=np.nonzero(peaks)
        selected=[]
        for y,x in sorted(zip(ys,xs),key=lambda q:-power[q])[:25]:
            if power[y,x]<power.max()*.08:break
            fx,fy=(x-128)/256,(y-128)/256
            amp=np.sqrt(power[y,x])/win.sum()
            if amp/max(np.std(a),.015)<.12:continue
            if any(np.hypot(fx-u,fy-v)<.018 or np.hypot(fx+u,fy+v)<.018 for u,v in selected):continue
            selected.append((fx,fy))
            if len(selected)>=4:break
        for f in selected:
            if not any(np.hypot(f[0]-u,f[1]-v)<.012 or np.hypot(f[0]+u,f[1]+v)<.012 for u,v in freqs):freqs.append(f)
    return freqs

def coherent_features(patches,freqs):
    a=np.asarray(patches,dtype=np.float32);std=a.std(axis=(1,2));a=(a-a.mean(axis=(1,2),keepdims=True))*window
    y,x=np.mgrid[:N,:N];kernels=np.array([np.exp(-2j*np.pi*(fx*x+fy*y))*1/window.sum() for fx,fy in freqs],dtype=np.complex64).reshape(-1,N*N).T
    amps=np.abs(a.reshape(-1,N*N)@kernels)/np.maximum(std[:,None],.015)
    return amps.astype(np.float32)
