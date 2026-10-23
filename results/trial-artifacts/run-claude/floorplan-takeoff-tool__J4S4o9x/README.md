# A101 scan takeoff

    python3 takeoff.py <scan.pdf> <takeoff.json>

Needs only numpy, opencv-python-headless, PyMuPDF and scipy (all preinstalled).
Runs in a few seconds per sheet.

| module        | what it does |
|---------------|--------------|
| `canon.py`    | extracts the scan, deskews it (projection profiles), finds the graphic scale (0-4-8-16-32 ft) and resamples to 25 px/ft; normalises paper/ink levels and measures a thin-line contrast factor used to adapt thresholds (`params.py`) |
| `walls.py`    | pairs of parallel dark outlines with a wall fill: CMU hatch (EXT/P3), dark gray (P2), light gray (P1); collinear pieces are joined through door openings and junctions; each run is measured between the centerlines of the walls it meets; CMU runs facing the outside are EXT |
| `floors.py`   | interior = inside the walls (nominal thicknesses EXT/P3 8", P2 7-1/4", P1 4-7/8", openings closed); split by the dark finish-transition lines; each region is classified by the amplitude of its hatch periodicity (LVT, WOM, VCT, CT, EP, CPT) or by stipple density (SC vs PC) |
| `doors.py`    | gaps in wall runs: one/two door leaves at the jambs -> single/pair, panels inside the wall -> auto sliding, dashed coil line -> overhead coiling |
| `fixtures.py` | axis-aligned rectangles from the light fixture linework, classified by size and content (diagonals, X, circle): produce tables 5x4, kiosks 3x3, checkout X-box 2.5x3, cooler doors ~2.3x0.6; gondola runs (4 ft wide) are cut at their cross lines into 3 ft / 4 ft sections and 1.5 ft endcaps |

The rules (centerline-to-centerline wall lengths, areas to wall faces, door
openings excluded from floor area) reproduce the three example takeoffs in
`/app/examples` to within 0.1 ft and ~3 sf.
