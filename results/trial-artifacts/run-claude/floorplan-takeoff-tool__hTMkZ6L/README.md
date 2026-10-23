# A101 fixture & finish takeoff

    python3 /app/takeoff/takeoff.py <scan.pdf> <takeoff.json>

Reads one scanned A101 sheet (single raster image in a PDF) and writes the takeoff JSON in the
same format as `/app/examples/store-*.takeoff.json`. Needs only the installed numpy, OpenCV,
SciPy and PyMuPDF; no network, ~5 s and ~1 GB RAM per sheet. Set `TAKEOFF_DEBUG=1` to print
per-wall, per-region and per-door details to stderr.

How it measures

- **Sheet**: the scan is de-rotated (projection-profile search) and scaled from the graphic
  scale bar (0-4-8-16-32 ft). The sheet border size is a fallback/cross-check.
- **Walls**: walls are pairs of heavy parallel face lines. The fill between them sets the type:
  diagonal CMU hatch (EXT if it bounds the building, otherwise P3), dark gray 6" stud (P2),
  light gray 3-5/8" stud (P1), or white/panels (door opening). Collinear pieces are chained
  through doors and junctions; each wall end that runs into another wall is extended to that
  wall's centerline. Door openings are not deducted.
- **Floors**: building interior minus wall bands (so door openings are excluded), split into
  regions by the heavy finish-transition lines. Each region is classified by its hatch
  (line orientation and lines-per-foot, stipple density); fixture footprints are ignored for
  classification but their floor area counts.
- **Doors**: openings in wall bands; leaves at one or both jambs give single/pair, sliding
  panels inside the band give automatic sliding, a wide opening without leaves is overhead coiling.
- **Fixtures**: strips of medium-weight parallel lines at the standard depths (gondola 4'-0",
  checkout 2'-6", kiosk 3'-0", produce table 5'-0" x 4'-0", cooler door 0.6'), cut into pieces
  at cross lines and classified by length and diagonals (endcaps, X-boxes, door diagonals).
  A double-sided gondola section is counted once.
