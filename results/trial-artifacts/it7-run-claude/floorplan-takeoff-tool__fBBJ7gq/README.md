# A101 scan quantity takeoff

    python3 takeoff.py <scan.pdf> <takeoff.json>

Reads one scanned fixture & finish plan (single raster page) and writes the
takeoff in the estimators' JSON format. Only numpy, OpenCV, PyMuPDF and SciPy
are used. A typical run takes 20–40 s on 2 CPUs with a peak of about 0.8 GB.
Set `TAKEOFF_DEBUG=1` to log each stage to stderr.

## Pipeline

1. **prep.py**: extracts the page image and converts it to an ink map. The
   paper level is a robust smooth surface, so large solid fills survive. It
   then deskews the image (projection sharpness) and measures the graphic
   scale. Scale comes from the centres of the black 0–4 and 8–16 ft blocks and
   the 32 ft end stroke, so it is independent of stroke width. The image is
   resampled to 24.5 px/ft.
2. **layout.py**: finds the legend: 8 floor-finish swatches and 4
   partition-type bars, in the chain's standard order (LVT-1 … WOM-1;
   EXT, P3, P2, P1). Ink darkness is then calibrated so that the legend bar
   outlines (the heavy pen) read 0.875.
3. **walls.py**: wall bands are pairs of heavy outlines (or solid poche). It
   runs this detection at two thresholds so dark floor fills don't merge with
   outlines.
   - Each band is classified by thickness (P1 is thin) and by its poche,
     compared with that sheet's legend bars (fill level, hatch energy,
     diagonal orientation).
   - EXT is the perimeter, measured along its centreline. Interior walls are
     extended to the centreline of the wall they meet.
   - Collinear pieces are merged across junctions and across door openings
     (doors are not deducted).
4. **floors.py / texture.py**: the floor is the building interior minus the
   full walls (door openings excluded).
   - It is split into zones by heavy finish-transition lines. Short gaps hidden
     under fixtures are bridged, and solid fills are not mistaken for lines.
   - Each zone is matched against the legend swatches with a shift-invariant
     descriptor: noise-subtracted power spectrum, energy and mean ink,
     compared to many crops of each swatch. Blank paper counts only as
     evidence for the sparsest pattern (the saw-cut PC-1).
   - Line pixels and fixture interiors take the finish of the nearest zone.
5. **doors.py**: openings lie between jamb lines crossing a wall. Each opening
   is classified by its symbol:
   - single or pair swing: quarter arcs (radial-profile test);
   - auto sliding: offset panels inside the wall;
   - overhead coiling: dashed centre line, or a wide clear exterior opening.
6. **fixtures.py**: the chain's CAD blocks (produce table, self-checkout,
   checkout, cooler door, 3'/4' gondola section, endcap) are rendered at the
   sheet scale. They are matched by normalised cross-correlation in all
   orientations. Overlaps across types go through joint non-max suppression,
   and cooler doors get row completion at a 2.5 ft pitch.

Every stage is wrapped so that a failure in one quantity still produces a
complete JSON with the others.

## Assumptions and known limits

- Assumes a rectangular building with axis-aligned walls and 8" CMU exterior
  walls, the 0-4-8-16-32 ft graphic scale, and the legend layout and standard
  order seen in the sample sheets. If the scale bar is missing, it falls back
  to the legend swatch width.
- Validation so far:
  - The three example scans match the estimators' takeoffs.
  - Synthetic sheets (random layouts, hatch and poche styles, mirrored plans,
    clean, faded, photocopy and yellowed states) pass nearly all checks.
  - Degraded copies of the examples (noise, fading, rotation, rescaling,
    lower resolution, JPEG artefacts) also pass nearly all checks.
- Weakest cases:
  - Heavy blur (σ ≈ 1.8 px on top of the scan blur) can miss the
    overhead-door dashes and cooler doors.
  - Strongly darkened photocopies can shift a few feet of wall between types.
  - Fading that removes thin lines much more than heavy ones can break finish
    transition lines.
