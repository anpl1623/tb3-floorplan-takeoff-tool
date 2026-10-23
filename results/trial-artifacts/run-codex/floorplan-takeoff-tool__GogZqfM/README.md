Run with the installed Python packages; no installation or network access is needed:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json
```

The output has the same keys and units as the supplied hand takeoffs. The tool writes the destination atomically and needs write permission only in the output directory.

For an audit image and detailed measurements:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json --debug-dir audit
```

The audit directory contains colored floor regions, wall centerlines, marked fixtures, and `measurements.json` with scale calibration, individual wall runs, finish classification votes, and detected openings.

The implementation deskews the raster, calibrates it against the printed 0–32 foot graphic scale, and normalizes its resolution and ink levels. It reconstructs paired wall faces, splits runs at intersections, reads the partition identifiers from the legend, and measures between wall centerlines. Floor regions follow wall faces and finish transition curves. Wall strips span door openings, excluding their thresholds from floor quantities. Fixtures, equipment and columns do not remove floor area.

Finish hatches are recognized by phase-independent spatial frequency features. Fixture counts use the symbols' physical dimensions, contours and repeated door fronts. Gondola module faces are paired before counting double-sided sections. Swing doors require a matching quarter-circle arc; sliding doors use their opposing arrow symbols.

This implementation targets the supplied architect's A101 standard: orthogonal plans, the eight illustrated finish hatches, the P1/P2/P3 partition legend, and the fixture symbols shown in the examples. It does not use PDF filenames, store identifiers, reference takeoff quantities, online services, or the examples directory at runtime.

`assets/` contains small raster templates for printed identifiers and symbols, plus finish hatch features learned from the supplied examples. These assets contain no store identifiers or quantity tables. Runtime dependencies are PyMuPDF, OpenCV, NumPy and SciPy, all present in the supplied environment.

Validate against the hand takeoffs:

```sh
python3 /app/takeoff/validate.py /app/examples
```

All three supplied examples pass the requested area and length tolerances, with exact counts. The largest observed area error is 1.25%; the largest length error is 0.09%. Runtime is approximately four seconds per scan on two CPUs. Additional checks used six recompressed copies with resizing, rotation, brightness changes and blur; all passed. A separate copy of the tool was also checked with a renamed input file, disabled network connections, a non-root user, and two CPUs. Validation on these examples does not establish an error bound for unexamined scans.
