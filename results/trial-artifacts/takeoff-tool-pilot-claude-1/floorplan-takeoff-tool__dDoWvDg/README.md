# A101 quantity takeoff

    python3 takeoff.py <scan.pdf> <takeoff.json>

Reads one scanned Cedar Valley Market A101 fixture & finish plan and writes the
takeoff in the same format as `examples/store-*.takeoff.json`. Dependencies are
only those in the base environment (PyMuPDF, OpenCV, NumPy). Everything else the
tool needs (symbol and legend templates) is in `tk/templates/`. A typical sheet
takes about 10 s and peaks at about 1.1 GB of memory.

## Pipeline (`tk/`)

| module | what it does |
|---|---|
| `image.py` | Extracts the embedded scan. Deskews it, first with a coarse projection-profile search and then with a sub-pixel fit of long lines. Calibrates pixels per foot from the graphic scale bar: the 0-4 and 8-16 ft solid segments give 8 ft and 12 ft spans in which the stroke width cancels out. If the bar can't be found, it falls back to the title block's 1-inch check bar. It also normalizes intensities so the paper reads 250 and solid ink reads 24. |
| `walls.py` | Finds walls as pairs of heavy parallel face lines 0.33–0.78 ft apart, using ridges of integrated darkness with a contrast test. It classifies each wall by thickness and mean fill: thin with light fill is the stud partition, thick with gray fill is the rated stud partition, thick with hatch is CMU. Collinear pieces are merged across junctions. Pieces are bridged across door openings only where frame lines continue along both faces. Face positions are refined to sub-pixel accuracy. Every wall end that runs into another wall is snapped to that wall's centerline. Exterior walls are the hatched walls that border the outside of the footprint. Openings are stretches where the band's mean fill departs from the wall's own fill; their ends are refined to the jamb lines. |
| `doors.py` | Classifies each opening by its evidence. A quarter-arc dip in a hinge-centred polar profile means a single leaf (radius = width) or a pair (two arcs of radius width/2). Solid panel lines inside the wall band mean an automatic sliding door. A dashed coil-box line beside the wall, or a width of 8 ft or more, means an overhead coiling door. |
| `floors.py` | Draws the walls (continuous through openings) and the heavy finish-transition lines (straight lines and arcs) as barriers. Connected regions inside the exterior wall become finish regions, and barrier pixels go to the nearest region. Transition lines hidden under a fixture's white fill are re-bridged. Each region is classified by texture: gradient-orientation histogram and energy against per-finish prototypes, plus line-period checks for LVT-1 vs WOM-1 and for the CT-1 diagonal grid. Door-swing quarter-discs are masked out first. |
| `fixtures.py` | Resamples the plan to 24 px/ft. Matches templates over all rotations and mirrors for produce tables, self-checkout kiosks, checkout stands and cooler doors; cooler doors must also show their diagonal. Gondola runs are found as outer–centre–outer line triples 2 ft apart with blank interiors. The dividers along each run give 3 ft and 4 ft sections; a double-sided section counts once. A 1.5 ft cell with a diagonal at a run end counts as an endcap. |
| `legend.py` | Reads the legend: the partition tag on each partition-type swatch row, and the finish codes in legend order. This guards against a sheet whose legend assigns tags or codes differently. It falls back to the standard legend unless the reading is unambiguous. |
| `pipeline.py` | Orchestration. Every stage is guarded, so a failure in one section still yields the others. |

## Measurement rules implemented

* Floor: square feet per finish, measured to the faces of walls. The wall mask runs
  continuously through door openings, so the floor inside an opening isn't counted.
  Fixtures, equipment and columns don't interrupt a region, so the floor under them
  is counted.
* Walls: centerline length per type. Each end is measured to the centerline of the
  wall it meets, and door openings aren't deducted. The exterior wall is measured
  along its centerline all the way around the building.

## Calibration notes

* Wall face lines read about 0.02 ft proud of the faces the hand takeoffs use. The
  wall mask for floor areas is trimmed by that amount (`floors.FACE_TRIM_FT`). With
  the trim, the floor errors on the three example sheets are within ±0.35%; wall
  lengths are within 0.15 ft.
* The tool was checked on the three examples and on perturbed copies of them:
  rotation up to ±4.5°, print scale 95–105%, extra blur up to σ=1.3 px, noise, dust
  specks, JPEG quality 35, lighter or darker ink and gamma. Also on pages scanned
  portrait or upside down, PDFs with a rotated or inset image placement, a transposed
  or 180°-rotated plan, fixtures pasted into new contexts (including collinear and
  back-to-back gondola runs), and fixtures drawn over finish boundaries.
* If a stage fails, the tool still writes a complete takeoff file. The failed section
  reads as zeros and the traceback goes to stderr.
