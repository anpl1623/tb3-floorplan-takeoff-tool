# A101 fixture & finish plan takeoff

    python3 /app/takeoff/takeoff.py <scan.pdf> <takeoff.json>

Reads one scanned A101 sheet and writes the takeoff JSON in the same format as the
estimators' examples in `/app/examples/`. Requires only numpy, scipy, OpenCV and PyMuPDF.
A scan takes about 10 s and about 1.2 GB of memory. Set `TAKEOFF_VERBOSE=1` to print
progress to stderr.

## How it works (`cvm_takeoff/`)

* **imgio**: extracts the scan image from the PDF, or renders the page if the scan isn't a
  single full-page image. Removes the scan's skew, which is found by maximising
  line-projection sharpness. Reads px/ft from the graphic scale bar (0-4-8-16-32 FEET),
  using same-polarity edges so stroke width doesn't bias it; the sheet border size is the
  fallback. Scans far from about 24.5 px/ft are resampled.
* **walls**: wall faces are heavy straight lines, measured by their integrated darkness
  (insensitive to blur). Pairs of faces at wall thickness are classified by the fill between
  them: P1 thin with light fill, P2 uniform gray, P3/EXT 45° hatch. Collinear pieces join
  through junctions and door openings. EXT is any hatched wall that touches the outside,
  found by flood-filling from the sheet edge. Lengths run along the centerline, and each
  end is snapped to the centerline of the wall it runs into.
* **floors**: floor regions are bounded by the walls (door openings included) plus heavy
  finish transition lines. Each region's hatch is identified from its local demodulated
  amplitude at the CAD standard's pattern frequencies:
  LVT 7" planks, VCT 12" grid, CT 9" diagonal grid, EP "/" at 12", CPT "\" at 6", WOM 3".
  Unhatched regions are sealed concrete if stippled, otherwise polished concrete. Line,
  text and column pixels go to the nearest region, so areas run to the face of walls and
  include the floor under fixtures and columns.
* **doors**: each opening in a wall run is tested against swing-arc hypotheses (one leaf vs
  a pair). Sliding panels in the opening mark auto-sliding doors; a wide opening with a
  dashed coil line marks an overhead coiling door.
* **fixtures**: parallel fixture-weight lines at a symbol's width form a "ladder" whose cross
  lines are found from 1-D profiles. Cells are then classified by size and content:
  - 4'-deep gondola runs (3' and 4' sections, endcaps with a diagonal)
  - cooler-door strips
  - 5'x4' produce tables with an X
  - 3'x3' self-checkouts with a circle
  - checkout stands (an X-marked bagging box next to a belt box)

Darkness is measured against a smooth local map of the paper tone, which absorbs uneven
illumination. Thin-line thresholds are scaled by the line response of the scan's own
exterior wall faces, which absorbs differences in blur and contrast between scans.

The finish codes, the wall-type legend (EXT/P1/P2/P3) and the fixture symbol sizes are
taken from the CAD standard used on these sheets; the legend text is not read.
