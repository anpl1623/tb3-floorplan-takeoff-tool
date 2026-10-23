# Cedar Valley Market takeoff

Run:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json
```

The JSON follows the supplied example schema. Counts are integers; floor areas are rounded to 0.1 square foot and partition lengths to 0.01 foot. Output replacement is atomic, so an unsuccessful recognition does not overwrite an existing takeoff.

The installation is self-contained apart from the environment's installed PyMuPDF, OpenCV, NumPy, SciPy, and Shapely packages. It needs no network, external executable, model download, writable installation directory, or example files at runtime. Numerical libraries are limited to two threads.

## Measurement method

- Correct sheet orientation, skew, paper shading, and contrast. Calibrate from the graphic scale's solid blocks. Check horizontal and vertical calibration independently against the 12-inch VCT grid, correcting stretched scans when necessary.
- Find paired wall faces, split at junctions and changes in infill, and classify the standard partition types. Join endpoints to intersecting wall centerlines. The exterior centerline forms a complete perimeter; door openings remain in partition lengths.
- Form continuous floor polygons from the wall faces, closing across door openings. Trace finish boundaries and classify their hatches. Fill annotation, column, and fixture interruptions from the surrounding finish. This includes floor underneath fixtures and excludes doorway thresholds.
- Match fixture symbols at the calibrated physical size, jointly distinguish 3-foot and 4-foot gondola bays, and count a double-sided bay once. Find swing leaves and their circular arcs, group paired leaves, and recognize sliding arrows and coiling-door linework.

Recognition uses the drawing conventions shown in the examples, including their rectangular building shell, orthogonal partitions, finish hatches, wall types, and fixture symbol sizes. Symbol references in `assets/` were cropped from the supplied drawings. There are no store identifiers, stored takeoff totals, or example lookups in the runtime.

## Audit images

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json --debug audit
```

This also writes:

- `normalized-plan.png`: deskewed plan at 24 pixels per foot.
- `floor-finishes.png`: finish segmentation, colors, and totals.
- `recognition.png`: detected walls, doors, and fixtures.
- `measurements.json`: calibration, wall geometry, hatch features, and detections.

## Verification

```sh
python3 /app/takeoff/verify_examples.py /app/examples --unprivileged
```

The verifier checks the exact output schema, all counts, and the requested area/length tolerances. It restricts execution to two CPUs and, when started by root with `--unprivileged`, runs the CLI as uid/gid 65534. Its temporary outputs are removed afterwards.

Validation in this environment: all three supplied examples passed, every count and reported partition length matched, and the largest floor-area error was 7.1 square feet (largest relative error 0.203%). Execution as uid/gid 65534 with two CPUs took about 14 seconds per original scan.

Twenty additional scan variants passed the requested tolerances, covering resolution changes, rotation and page orientation, fading, darkness, blur, JPEG noise, uneven paper shading, and unequal horizontal/vertical stretching. Changed-plan checks exercised fixture additions/removals, 2.5–4-foot swing leaves, freestanding cooler banks, and adjacent coiling doors. A fresh read-only installation also passed with an arbitrary input filename, a new output directory, and audit images enabled.
