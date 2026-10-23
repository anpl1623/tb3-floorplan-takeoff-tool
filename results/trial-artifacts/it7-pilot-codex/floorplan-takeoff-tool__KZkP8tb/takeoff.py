#!/usr/bin/env python3
"""Quantity takeoff for a scanned Cedar Valley Market A101 plan.

Usage: python3 takeoff.py input.pdf output.json [--debug directory]
"""
import os
# Bound numerical-library parallelism before importing NumPy/OpenCV/SciPy.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

from engine import init
from legend import extract
from walls import detect as detect_walls, classify_walls, quantities as wall_quantities,draw_typed_walls
from regions import floor_quantities
from doors import quantities as door_quantities
from blocks import quantities as fixture_quantities, wall_source
from tags import detect as finish_annotations


def run(scan,debug=None):
    ink,ppf,box,plan,rect=init(scan,debug)
    legend=extract(ink,ppf,box,debug)
    del ink
    annotations=finish_annotations(plan,debug)
    fixture_counts,fixture_objects=fixture_quantities(plan,rect,debug,return_detections=True)
    wall_image,fixture_mask=wall_source(plan,fixture_objects,rect,return_mask=True)
    raw_walls,footprint=detect_walls(wall_image,rect,debug,legend,return_footprint=True,annotations=annotations)
    walls=classify_walls(plan,raw_walls,legend,rect,footprint)
    if debug:
        from shapely.geometry import mapping
        with open(Path(debug)/'footprint.geojson','w') as f:json.dump(mapping(footprint),f)
    if debug:
        with open(Path(debug)/'typed-walls.json','w') as f:json.dump(walls,f,indent=2)
        draw_typed_walls(plan,walls,debug)
    floors=floor_quantities(plan,walls,rect,legend,debug,annotations=annotations,boundary_src=wall_image,footprint=footprint,occlusions=fixture_mask)
    del wall_image,fixture_mask
    return {
        'floor_sf':floors,
        'wall_lf':wall_quantities(walls),
        'doors':door_quantities(plan,walls,debug),
        'fixtures':fixture_counts,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--debug',metavar='DIRECTORY',help='save measurement overlays and diagnostic geometry')
    args=parser.parse_args()
    if args.scan.resolve()==args.output.resolve():
        parser.error('The output must be different from the input PDF')
    started=time.monotonic()
    try:
        result=run(str(args.scan),args.debug)
        # An interrupted or failed run cannot leave a partly written takeoff.
        temp_path=None
        try:
            with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=args.output.parent,prefix='.'+args.output.name+'.',suffix='.tmp',delete=False) as f:
                temp_path=f.name
                json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
            os.replace(temp_path,args.output)
        finally:
            if temp_path and os.path.exists(temp_path):os.unlink(temp_path)
        if args.debug:
            print('Completed in %.1f seconds'%(time.monotonic()-started),file=sys.stderr)
    except (ValueError,OSError,RuntimeError) as exc:
        print('Takeoff failed: '+str(exc),file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
