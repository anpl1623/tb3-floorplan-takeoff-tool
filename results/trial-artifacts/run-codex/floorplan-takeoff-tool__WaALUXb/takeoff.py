#!/usr/bin/env python3
"""Quantity takeoff for Cedar Valley's scanned A101 drawing standard.

Usage: python3 takeoff.py scan.pdf takeoff.json
All measurements are calibrated from the printed graphic scale.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import cv2
import fitz
import numpy as np
from shapely.geometry import Polygon

cv2.setNumThreads(2)
ASSETS = Path(__file__).resolve().parent / 'assets'
FINISHES = ['LVT-1', 'PC-1', 'SC-1', 'EP-1', 'VCT-1', 'CT-1', 'CPT-1', 'WOM-1']


def read_scan(path):
    with fitz.open(path) as doc:
        page = doc[0]
        images = page.get_images(full=True)
        if images:
            item = max(images, key=lambda z: z[2] * z[3])
            rects = page.get_image_rects(item[0])
            if len(images)==1 or (rects and rects[0].get_area() > .85 * page.rect.get_area()):
                data = doc.extract_image(item[0])['image']
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    return image
        pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72), colorspace=fitz.csGRAY)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def straighten(image):
    """Estimate the two CAD axes from long rules, independent of the hatch."""
    factor = min(1., 3600 / max(image.shape))
    small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(small)[0]
    if detected is None:
        raise ValueError('No drawing rules found in scan')
    lines = detected[:, 0, :]
    d = lines[:, 2:] - lines[:, :2]
    length = np.linalg.norm(d, axis=1)
    horizontal = (np.abs(d[:, 0]) > 2 * np.abs(d[:, 1])) & (length > 500 * factor)
    vertical = (np.abs(d[:, 1]) > 2 * np.abs(d[:, 0])) & (length > 500 * factor)
    def slope(mask, numerator, denominator):
        ids = np.flatnonzero(mask)
        ids = ids[np.argsort(length[ids])[-8:]]
        if len(ids) < 2:
            raise ValueError('Cannot identify the drawing axes')
        return float(np.median(d[ids, numerator] / d[ids, denominator]))
    hs = slope(horizontal, 1, 0)
    vs = slope(vertical, 0, 1)
    # Preserve axis lengths; remove rotation and modest scan shear.
    axes = np.array([[1., vs], [hs, 1.]])
    axes /= np.linalg.norm(axes, axis=0)
    transform = np.linalg.inv(axes)
    h, w = image.shape
    corners = np.array([[0,0], [w,0], [w,h], [0,h]]) @ transform.T
    low = np.floor(corners.min(axis=0)) - 10
    high = np.ceil(corners.max(axis=0)) + 10
    matrix = np.column_stack([transform, -low])
    return cv2.warpAffine(image, matrix, tuple((high-low).astype(int)),
                          flags=cv2.INTER_LINEAR, borderValue=255)


def graphic_scale(image):
    for threshold in [95,70,45,120,150,180]:
        try:
            return _graphic_scale(image,threshold)
        except ValueError:
            pass
    raise ValueError('Cannot find the 0–32 foot graphic scale')


def _graphic_scale(image,threshold):
    """Locate the adjacent 4 and 8 foot solid intervals of the 32 foot bar."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats((image < threshold).astype(np.uint8))
    candidates = []
    h, w = image.shape
    for i, (x,y,ww,hh,area) in enumerate(stats[1:], 1):
        if (x < .68*w and y > .55*h and 9 < hh < .025*h and
                3.1 < ww/hh < 9.5 and area > .65*ww*hh):
            candidates.append((i, x, y, ww, hh))
    pairs = []
    for a in candidates:
        for b in candidates:
            _, ax,ay,aw,ah = a
            _, bx,by,bw,bh = b
            if (1.7 < bw/aw < 2.3 and abs(ay-by) < max(3,.15*ah) and
                    abs(ah-bh) < max(3,.2*ah) and abs((bx-ax)/aw-2) < .2):
                pairs.append((abs((bx-ax)/aw-2) + abs(bw/aw-2), a,b))
    if not pairs:
        raise ValueError('Cannot find the 0–32 foot graphic scale')
    _, a,b = min(pairs)
    _,x,y,ww,hh = a
    estimate = (b[3]-2)/8
    x0=max(0,int(x-estimate));y0=max(0,y-hh)
    x1=min(w,int(x+34*estimate));y1=min(h,y+2*hh)
    crop=image[y0:y1,x0:x1]
    mask=(crop<170).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
    nn,ll,ss,cc=cv2.connectedComponentsWithStats(mask)
    choices=[]
    for xx,yy,bww,bhh,area in ss[1:]:
        if 29*estimate < bww < 34*estimate and .6*hh < bhh < 1.8*hh:
            choices.append((abs(bww/estimate-32),xx,yy,bww,bhh))
    if not choices:
        # The solid interval still provides a scale if the light outline is broken.
        return float(estimate), (int(x),int(y),int(32*estimate),int(hh))
    _,xx,yy,bww,bhh=min(choices)
    # Center of the outline stroke at each endpoint (about 2.5 native pixels).
    scale=float((bww-2.5)/32)
    return scale, (int(x0+xx),int(y0+yy),int(bww),int(bhh))


def projection_centers(profile, minimum):
    """Return stroke centers, merging the adjacent pixels of each printed rule."""
    ids=np.flatnonzero(profile > minimum)
    if len(ids)==0:return []
    groups=np.split(ids,np.where(np.diff(ids)>2)[0]+1)
    return [(float(np.average(g,weights=profile[g])),float(profile[g].max())) for g in groups]


def plan_bounds(image, scale):
    h,w=image.shape
    # Fixture and hatch strokes are lighter than the wall-face rules.
    binary=(image[:,:int(.68*w)] < 145).astype(np.uint8)
    kernel=max(40,round(4*scale))
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((kernel,1),np.uint8))
    profile=vertical.sum(axis=0)
    centers=projection_centers(profile, 3*scale)
    pairs=[]
    for a in centers:
        for b in centers:
            if .4*scale < b[0]-a[0] < .9*scale:
                overlap=np.count_nonzero(vertical[:,round(a[0])]&vertical[:,round(b[0])])
                if overlap>3*scale:
                    pairs.append((overlap,a[0],b[0]))
    if len(pairs)<2:raise ValueError('Cannot find the exterior wall faces')
    outer=pairs
    left=min(outer,key=lambda p:p[1])[1]
    right=max(outer,key=lambda p:p[2])[2]
    x1,x2=round(left),round(right)
    horizontal=cv2.morphologyEx(binary[:,x1:x2+1],cv2.MORPH_OPEN,np.ones((1,kernel),np.uint8))
    profile=horizontal.sum(axis=1)
    centers=projection_centers(profile, 3*scale)
    pairs=[]
    for a in centers:
        for b in centers:
            if .4*scale < b[0]-a[0] < .9*scale:
                overlap=np.count_nonzero(horizontal[round(a[0])]&horizontal[round(b[0])])
                if overlap>3*scale:
                    pairs.append((overlap,a[0],b[0]))
    if len(pairs)<2:raise ValueError('Cannot find the exterior horizontal walls')
    top=min(pairs,key=lambda p:p[1])[1]
    bottom=max(pairs,key=lambda p:p[2])[2]
    return (x1,round(top),x2+1,round(bottom)+1)


def floor_regions(image, scale, threshold=145):
    """Separate closed architectural regions, retaining floor under furniture."""
    binary=(image < threshold).astype(np.uint8)
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    binary=cv2.dilate(binary,np.ones((3,3),np.uint8))
    binary[:2,:]=1;binary[-2:,:]=1;binary[:,:2]=1;binary[:,-2:]=1
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(1-binary)
    regions=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area < 30*scale**2:continue
        # Unbuilt space in a re-entrant exterior corner reaches the crop edge.
        if x<=3 or y<=3 or x+w>=image.shape[1]-3 or y+h>=image.shape[0]-3:
            continue
        mask=(labels[y:y+h,x:x+w]==i).astype(np.uint8)
        contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        contour=max(contours,key=cv2.contourArea)
        filled=np.zeros_like(mask)
        cv2.drawContours(filled,[contour],-1,1,-1)
        # The narrow spaces inside walls, gondolas, and cooler cases are not rooms.
        if cv2.distanceTransform(filled,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).max() < 2.1*scale:
            continue
        contour=contour[:,0,:]+[x,y]
        polygon=Polygon(contour/scale).buffer(3.3/scale,join_style=2)
        regions.append(dict(id=i,polygon=polygon,outer=polygon,bounds=(x,y,w,h),mask=mask))
    # Preserve other floor regions enclosed by a surrounding region; fill text,
    # equipment, fixtures, and columns back into their floor finish.
    for region in regions:
        polygon=region['outer']
        for other in regions:
            if (other is not region and polygon.area > other['outer'].area and
                    polygon.contains(other['outer'].representative_point())):
                polygon=polygon.difference(other['outer'])
        region['polygon']=polygon
    return regions,labels


def debug_regions(image,regions,scale,path):
    out=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    colors=np.array([[210,130,90],[120,210,120],[150,80,210],[90,180,240],
                     [220,210,80],[170,120,220],[150,150,90],[200,130,170]],np.uint8)
    overlay=out.copy()
    for i,r in enumerate(regions):
        p=r['polygon'];parts=[p] if p.geom_type=='Polygon' else list(p.geoms)
        mask=np.zeros(image.shape,np.uint8)
        for part in parts:
            cv2.fillPoly(mask,[np.rint(np.array(part.exterior.coords)*scale).astype(np.int32)],1)
            for hole in part.interiors:
                cv2.fillPoly(mask,[np.rint(np.array(hole.coords)*scale).astype(np.int32)],0)
        index=FINISHES.index(r['finish']) if r.get('finish') in FINISHES else i%8
        overlay[mask>0]=colors[index]
    out=cv2.addWeighted(out,.4,overlay,.6,0)
    for i,r in enumerate(regions):
        center=r['polygon'].representative_point();pos=(round(center.x*scale),round(center.y*scale))
        label=f"{i}: {r.get('finish','?')} {r['polygon'].area:.1f}"
        cv2.putText(out,label,pos,cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
    cv2.imwrite(str(path),cv2.resize(out,None,fx=.5,fy=.5))


def prepare_sheet(image):
    """Orient the landscape sheet and locate its printed scale."""
    if image.shape[0]>image.shape[1]:
        image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    if image.shape[1]!=7200:
        factor=7200/image.shape[1]
        image=cv2.resize(image,None,fx=factor,fy=factor,
                         interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_CUBIC)
    image=straighten(image)
    try:
        scale,bar=graphic_scale(image)
    except ValueError:
        image=cv2.rotate(image,cv2.ROTATE_180)
        scale,bar=graphic_scale(image)
    return image,scale,bar


def analyze(path,debug=None):
    import finishes
    import walls
    import doors
    import fixtures
    import calibration
    image,scale,bar=prepare_sheet(read_scan(path))
    image,tone_changed=calibration.normalize_tone(image,scale,bar)
    if tone_changed:
        scale,bar=graphic_scale(image)
    box=plan_bounds(image,scale)
    x1,y1,x2,y2=box
    plan=image[y1:y2,x1:x2]
    regions,labels=floor_regions(plan,scale)
    if not regions:
        raise ValueError('No enclosed floor finish regions could be identified')
    areas=finishes.classify(plan,regions,scale,ASSETS)
    pixel_aspect=calibration.aspect_ratio(plan,regions,scale)
    if abs(pixel_aspect-1)>.0025:
        image=cv2.resize(image,None,fx=1,fy=1/pixel_aspect,interpolation=cv2.INTER_LINEAR)
        scale,bar=graphic_scale(image)
        box=plan_bounds(image,scale)
        x1,y1,x2,y2=box
        plan=image[y1:y2,x1:x2]
        regions,labels=floor_regions(plan,scale)
        areas=finishes.classify(plan,regions,scale,ASSETS)
    lengths,wall_segments=walls.measure(plan,scale,regions)
    door_counts,door_hits=doors.count(image,box,scale,wall_segments,regions)
    fixture_counts,fixture_hits,normalized=fixtures.count(plan,scale,ASSETS)
    result={'floor_sf':{code:round(float(areas[code]),1) for code in FINISHES},
            'wall_lf':{code:round(float(lengths[code]),2) for code in ['EXT','P1','P2','P3']},
            'doors':door_counts,'fixtures':fixture_counts}
    if debug:
        d=Path(debug);d.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(d/'sheet.png'),image)
        cv2.imwrite(str(d/'plan.png'),plan)
        debug_regions(plan,regions,scale,d/'regions.png')
        out=cv2.cvtColor(normalized,cv2.COLOR_GRAY2BGR)
        for hit in fixture_hits:
            x,y,w,h=(hit[k] for k in ['x','y','w','h'])
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,220),2)
            cv2.putText(out,hit['kind'],(x,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,220),1)
        cv2.imwrite(str(d/'fixtures.png'),out)
        out=cv2.cvtColor(plan,cv2.COLOR_GRAY2BGR)
        wall_colors={'EXT':(0,120,230),'P1':(210,80,30),'P2':(40,30,220),'P3':(40,160,40)}
        for wall in wall_segments:
            a=(wall['a'],wall['p']);b=(wall['b'],wall['p'])
            if wall['axis']=='V':a=a[::-1];b=b[::-1]
            a=tuple(round(v*scale) for v in a);b=tuple(round(v*scale) for v in b)
            cv2.line(out,a,b,wall_colors[wall['kind']],3)
            middle=tuple((v+w)//2 for v,w in zip(a,b))
            label=f"{wall['kind']} {wall['b']-wall['a']:.1f} ft"
            cv2.putText(out,label,middle,cv2.FONT_HERSHEY_SIMPLEX,.5,wall_colors[wall['kind']],1)
        cv2.imwrite(str(d/'walls.png'),out)
        out=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        for hit in door_hits:
            x=round(x1+hit['x']*scale);y=round(y1+hit['y']*scale)
            cv2.circle(out,(x,y),round(.9*scale),(0,0,220),2)
            cv2.putText(out,hit['kind'],(x+20,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,220),2)
        cv2.imwrite(str(d/'doors.png'),cv2.resize(out,None,fx=.5,fy=.5))
        audit={'pixels_per_foot':scale,'scanner_pixel_aspect':pixel_aspect,'graphic_scale_box':bar,'plan_box':box,
               'regions':[dict(finish=r['finish'],area=r['polygon'].area,
                               texture_distance=r['texture_distance']) for r in regions],
               'walls':wall_segments,'doors':door_hits,'fixtures':fixture_hits,'takeoff':result}
        (d/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan',help='one scanned A101 PDF')
    parser.add_argument('output',help='destination takeoff JSON')
    args=parser.parse_args()
    try:
        result=analyze(args.scan,os.environ.get('TAKEOFF_DEBUG'))
        # Write only after every quantity has been calculated successfully.
        import tempfile
        destination=Path(args.output)
        with tempfile.NamedTemporaryFile(mode='w',dir=destination.parent,
                                          prefix='.'+destination.name+'.',suffix='.tmp',
                                          delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(result,stream,indent=2,allow_nan=False)
            stream.write('\n')
        os.replace(temporary,destination)
    except Exception as exc:
        if 'temporary' in locals():
            temporary.unlink(missing_ok=True)
        print(f'takeoff: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
