# A101 fixture & finish plan takeoff

    python3 /app/takeoff/takeoff.py <scan.pdf> <takeoff.json>

Needs only numpy, opencv, PyMuPDF and scipy. Set `TAKEOFF_DEBUG=1` to print diagnostics to stderr.

## How it works

1. **Image** (`common.py`): the embedded page scan is extracted and turned into an "ink darkness" map
   (paper = 0, solid ink = 1, local background removed), then deskewed using long border/wall lines.
   The sheet is turned the right way up if the legend is not on the right.
2. **Scale** (`scalebar.py`): the graphic scale bar (0-4-8-16-32 ft, solid 0-4 and 8-16 segments) is
   located and its edges measured to sub-pixel accuracy, which gives pixels per foot. This is cross-checked
   against the width of the legend swatches.
3. **Legend** (`legend.py`): the 8 floor-finish swatches (LVT-1, PC-1, SC-1, EP-1, VCT-1, CT-1, CPT-1,
   WOM-1) and the 4 partition swatches (EXT, P3, P2, P1) are found in the chain's standard order. They are
   the only reference for the office's hatching and poché.
4. **Walls** (`walls.py`): pairs of parallel outline strokes at partition thickness are typed by comparing
   their poché (mean darkness, texture energy, diagonal direction, thickness) with the partition swatches.
   Collinear pieces are merged across junctions and door openings. Walls that touch the outside are the
   exterior wall. Each wall's centreline is extended to the centreline of the wall it runs into.
5. **Doors** (`doors.py`): openings are the gaps between jamb lines across a wall band. Each opening is
   classified from the door block drawn there: sliding panels (automatic sliding), two leaves with arcs
   (pair), one leaf with an arc (single), or a dashed hood line / wide exterior opening (overhead coiling).
6. **Fixtures** (`fixtures.py`): the chain's fixture blocks are drawn from their dimensions in feet at the
   measured scale and found by normalised cross-correlation in all 8 orientations. 3'-0" and 4'-0" gondola
   sections compete for each location. Cooler doors have to sit in a row in front of a blank cooler case.
7. **Floors** (`floors.py`, `takeoff.py`): the floor is the area inside the exterior wall minus all wall
   bands, with door openings counted as wall. It is split into zones along wall faces and dark finish
   transition lines. Zones with mixed texture are split again along fainter lines. Each zone is
   classified by voting over texture patches, using an orientation and frequency power-spectrum
   descriptor compared with the legend swatches. Text and fixtures are left out of the sampling, but the
   floor under them counts. Every floor pixel is then assigned to the nearest classified zone.
