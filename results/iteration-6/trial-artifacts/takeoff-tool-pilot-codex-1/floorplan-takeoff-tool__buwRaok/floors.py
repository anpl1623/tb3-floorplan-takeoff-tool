"""Wall-face floor polygons and finish-hatch segmentation.

Finish boundaries use a darker pen than hatches and fixture outlines. Closed
boundary regions provide labels; nearest-region filling accounts for notes,
columns, and the floor underneath fixture symbols. Areas are apportioned within
continuous room polygons so wall and doorway thickness is never counted.
"""
import math
import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import unary_union
from vision import PPF

THICKNESS={'EXT':2/3,'P1':4.875/12,'P2':7.25/12,'P3':2/3}
FINISHES=('LVT-1','PC-1','SC-1','EP-1','VCT-1','CT-1','CPT-1','WOM-1')


def snap(value):
    feet=value/PPF
    grid=round(feet*2)/2
    return grid if abs(feet-grid)<.13 else feet


def wall_geometry(plan,walls):
    height,width=plan.shape
    width=snap(width-1);height=snap(height-1)
    geometry=[{**wall,'lo':snap(wall['lo']),'hi':snap(wall['hi']),'pos':snap(wall['pos']),'width':THICKNESS[wall['typ']]}for wall in walls]
    exterior=[
        dict(axis=0,pos=1/3,lo=1/3,hi=width-1/3,width=2/3,typ='EXT'),
        dict(axis=0,pos=height-1/3,lo=1/3,hi=width-1/3,width=2/3,typ='EXT'),
        dict(axis=1,pos=1/3,lo=1/3,hi=height-1/3,width=2/3,typ='EXT'),
        dict(axis=1,pos=width-1/3,lo=1/3,hi=height-1/3,width=2/3,typ='EXT'),
    ]
    for wall in geometry:
        for endpoint in ('lo','hi'):
            junctions=[other['pos']for other in geometry+exterior if other['axis']!=wall['axis'] and other['lo']-1<=wall['pos']<=other['hi']+1 and abs(other['pos']-wall[endpoint])<1]
            if junctions:wall[endpoint]=min(junctions,key=lambda p:abs(p-wall[endpoint]))
    return geometry+exterior


def room_polygons(plan,walls):
    height,width=plan.shape
    width=snap(width-1);height=snap(height-1)
    polygons=[]
    for wall in wall_geometry(plan,walls):
        low,high,coordinate,thick=wall['lo'],wall['hi'],wall['pos'],wall['width']
        if wall['axis']==0:polygon=box(low,coordinate-thick/2,high,coordinate+thick/2)
        else:polygon=box(coordinate-thick/2,low,coordinate+thick/2,high)
        polygons.append(polygon)
    free=box(2/3,2/3,width-2/3,height-2/3).difference(unary_union(polygons))
    rooms=list(free.geoms)if hasattr(free,'geoms')else[free]
    return sorted([room for room in rooms if room.area>.1],key=lambda p:(p.bounds[1],p.bounds[0]))


def polygon_mask(polygon,shape,origin=(0,0)):
    mask=np.zeros(shape,np.uint8)
    xy=np.round(np.array(polygon.exterior.coords)*PPF-np.array(origin)).astype('int32')
    cv2.fillPoly(mask,[xy],1)
    for hole in polygon.interiors:
        xy=np.round(np.array(hole.coords)*PPF-np.array(origin)).astype('int32')
        cv2.fillPoly(mask,[xy],0)
    return mask


def texture_features(plan):
    ink=np.clip((240-plan.astype(np.float32))/100,0,1)
    kernels=[np.ones((1,15),np.uint8),np.ones((15,1),np.uint8),np.eye(15,dtype='uint8'),np.fliplr(np.eye(15,dtype='uint8'))]
    lines=[cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)for kernel in kernels]
    return ink,lines


def classify_region(plan,polygon,lines):
    # Used for finding the independent 12-inch tile calibration grid.
    mask=polygon_mask(polygon,plan.shape)
    mask=cv2.erode(mask,np.ones((17,17),np.uint8))>0
    features=np.array([float(np.mean(response[mask]))for response in lines])
    horizontal,vertical,diagonal,anti=features
    if min(diagonal,anti)>.014:code='CT-1'
    elif diagonal>.018 and diagonal>anti*2.2:code='CPT-1'
    elif anti>.010 and anti>diagonal*2.2:code='EP-1'
    elif horizontal>.055 and horizontal>vertical*2:code='WOM-1'
    elif min(horizontal,vertical)>.008 and max(horizontal,vertical)<min(horizontal,vertical)*1.7:code='VCT-1'
    else:code='SC-1'
    return code,features


def boundary_threshold(plan,walls):
    levels=[]
    for wall in wall_geometry(plan,walls):
        image=plan if wall['axis']==0 else plan.T
        for side in (-1,1):
            y=round((wall['pos']+side*wall['width']/2)*24)
            for fraction in (.15,.3,.5,.7,.85):
                x=round((wall['lo']*(1-fraction)+wall['hi']*fraction)*24)
                patch=image[max(0,y-3):y+4,max(0,x-1):x+2]
                if patch.size:levels.append(float(np.min(patch)))
    level=float(np.median(levels)) if levels else 75
    return float(np.clip(145+.9*(level-65),145,195))


def hatch_bands(response,mask):
    count=mask.sum(axis=1)
    profile=(response*mask).sum(axis=1)/np.maximum(1,count)
    profile-=ndimage.gaussian_filter1d(profile,16)
    profile*=np.hanning(len(profile))
    spectrum=np.abs(np.fft.rfft(profile,n=16384))
    frequencies=np.fft.rfftfreq(16384)
    values=[]
    for periods in ((6,),(7,14),(12,24)):
        strength=0.
        for period in periods:
            band=(frequencies>1/(period*1.06))&(frequencies<1/(period*.94))
            strength=max(strength,float(spectrum[band].max()))
        values.append(strength)
    return values


def classify_component(lines,residual,mask,gray):
    interior=cv2.erode(mask,np.ones((9,9),np.uint8))>0
    dark=(residual+np.maximum.reduce(lines)>.9).astype('uint8')
    annotation=cv2.dilate(dark,np.ones((15,15),np.uint8))>0
    interior&=~annotation
    if not np.any(interior):return None,None
    features=np.array([float(np.mean(response[interior]))for response in lines]+[float(np.mean(residual[interior]))])
    horizontal,vertical,diagonal,anti,random=features
    paper,upper=np.percentile(gray[interior],[50,90])
    sigma=max(.5,(upper-paper)/1.28155)
    z=(240-paper)/sigma
    noise=(sigma*math.exp(-.5*z*z)/math.sqrt(2*math.pi)+(240-paper)*.5*(1+math.erf(z/math.sqrt(2))))/100
    unhatched='SC-1' if random-max(0,noise)>.0035 else 'PC-1'
    if min(diagonal,anti)>.011:code='CT-1'
    elif diagonal>.012 and diagonal>anti*2.2:code='CPT-1'
    elif anti>.0045 and anti>diagonal*2.2 and anti>random*.8:code='EP-1'
    elif horizontal>.008:
        powers=hatch_bands(lines[0],interior)
        best=int(np.argmax(powers))
        code=('WOM-1','LVT-1','VCT-1')[best]
        if (best==2 and (vertical<.008 or vertical<horizontal*.45)) or (best==1 and horizontal<.012) or (best==0 and horizontal<.022):
            code=unhatched
        # Weak, nonperiodic horizontal fragments can occur in stipple and notes.
        if max(powers)<.05:code=unhatched
    else:code=unhatched
    return code,features


def floor_quantities(plan,walls,debug=None):
    rooms=room_polygons(plan,walls)
    if not rooms:raise ValueError('No enclosed floor regions were found')
    ink,lines=texture_features(plan)
    residual=np.maximum(0,ink-np.maximum.reduce(lines))
    free=np.zeros(plan.shape,np.uint8)
    room_masks=[]
    for room in rooms:
        mask=polygon_mask(room,plan.shape)
        room_masks.append(mask)
        free|=mask
    threshold=boundary_threshold(plan,walls)
    barrier=(plan<threshold).astype('uint8')
    barrier=cv2.morphologyEx(barrier,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    barrier|=1-free
    number,components,stats,centers=cv2.connectedComponentsWithStats(1-barrier)
    seeds=np.zeros(plan.shape,np.uint8)
    regions=[]
    for index in range(1,number):
        x,y,width,height,area=stats[index]
        if area<PPF*PPF*15:continue
        component=(components[y:y+height,x:x+width]==index).astype('uint8')
        responses=[line[y:y+height,x:x+width]for line in lines]
        code,features=classify_component(responses,residual[y:y+height,x:x+width],component,plan[y:y+height,x:x+width])
        if code is None:continue
        # An isolated white fixture bay has no texture; the surrounding floor,
        # rather than the empty bay interior, determines its finish.
        if code in ('PC-1','SC-1') and area<PPF*PPF*50:continue
        seeds[y:y+height,x:x+width][component>0]=FINISHES.index(code)+1
        regions.append({'code':code,'bounds_px':[int(x),int(y),int(width),int(height)],'features':features.tolist()})
    if not np.any(seeds):raise ValueError('No floor-finish hatch regions were recognized')
    nearest=ndimage.distance_transform_edt(seeds==0,return_distances=False,return_indices=True)
    labels=seeds[tuple(nearest)]
    quantities={code:0. for code in FINISHES}
    for room,mask in zip(rooms,room_masks):
        counts=np.bincount(labels[mask>0],minlength=9)[1:]
        total=int(counts.sum())
        if not total:continue
        for code,count in zip(FINISHES,counts):quantities[code]+=float(count)/total*room.area
    if debug is not None:debug.update(threshold=threshold,regions=regions,_labels=np.where(free,labels,0).astype('uint8'))
    return {code:round(float(area),1)for code,area in quantities.items()}
