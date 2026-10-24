# Cedar Valley Market takeoff

Run with the environment's Python and installed packages:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json
```

The output uses the supplied `floor_sf`, `wall_lf`, `doors`, and `fixtures` schema. Areas are written to 0.1 square foot, lengths to 0.01 foot, and counts as integers. The destination is replaced atomically after analysis succeeds.

The runtime reads the input PDF and the two packaged text-recognition template banks. It requires NumPy, OpenCV, SciPy, and PyMuPDF, which are installed in the supplied environment. It runs offline, supports an unprivileged account, and limits native processing threads to two.

## Measurement

* The sheet frame corrects rotation and perspective. Circular drawing symbols identify unequal horizontal and vertical scan scaling.
* The graphic scale's numerical captions establish feet per pixel. The PDF page size and printed scale notation do not establish measurement scale.
* The input legend supplies the finish and partition-poche samples. Boxed finish-code callouts provide an independent material check. The text template banks were made from the supplied scans and locally rendered digit glyphs.
* Partition faces determine wall centerlines. Junctions extend to the centerline of the intersected wall, and door openings remain in wall lengths.
* Floor regions extend to wall faces. Occluded floor is filled within its wall-bounded room, retaining floor beneath fixtures and columns and excluding openings through walls.
* Fixture recognition uses the chain blocks' geometry and repeated module spacing. A double-sided gondola module contributes one section. Three-foot and four-foot modules are measured and counted separately.

## Diagnostic images

Set `TAKEOFF_DEBUG` to a writable directory:

```sh
TAKEOFF_DEBUG=/tmp/takeoff-review \
  python3 /app/takeoff/takeoff.py scan.pdf takeoff.json
```

This writes `recognized.png`, `finishes.png`, `calibration.json`, and a copy of the measured quantities. In the recognition overlay, exterior walls are red, P1 green, P2 blue, P3 orange, and fixtures magenta.

## Validation

Run the independent comparison against the supplied estimator takeoffs:

```sh
python3 /app/takeoff/validate.py /app/examples
```

The comparison checks the complete output schema, exact counts, the greater of 2% or 8 square feet for each finish, the greater of 1.5% or 2 feet for each partition type, and the five-minute runtime limit. Each input runs in a separate process.
