Cedar Valley Market takeoff is an offline raster measurement tool. Run it with the two required paths:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json
```

It writes the example JSON schema: `floor_sf`, `wall_lf`, `doors`, and `fixtures`. Areas are rounded to 0.1 square foot, lengths to 0.01 foot, and counts are integers. Output is written atomically after measurement completes.

The tool uses the environment's installed Python packages: NumPy, SciPy, OpenCV, PyMuPDF, and Shapely. There are no downloads, external executables, service calls, or installation steps. Everything beyond those installed packages is in this directory. Numerical libraries are limited to two threads. The package can be read-only and the process can run as an ordinary user.

For a visual audit, add a diagnostic directory:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json --debug audit
```

The audit contains finish-region, typed-wall, door, and fixture overlays; the measured building footprint; recognized finish annotations; and JSON geometry for individual measurements. Diagnostic coordinates use the corrected plan raster at 16 pixels per foot, with a five-foot margin. The diagnostic directory also contains intermediate raster arrays, so it uses considerably more disk space than the takeoff JSON.

The measurement pipeline:

- Extracts the scan, compensates for paper color and fading, and corrects rotation and skew.
- Finds the graphic scale, reads its numeric divisions, and fits its segment positions to reduce pixel rounding error.
- Reads the eight finish swatches and four partition swatches in the current sheet's legend. Finish and poche recognition use those swatches, with finish-code annotations as a second source of evidence.
- Recognizes the chain's fixture blocks, removes their strokes from architectural line detection, and counts sections individually. A double-sided gondola section produces one count.
- Reconstructs partition centerlines and junctions, continuing through door openings. Floor polygons use wall faces and exclude the opening strips. Fixture and column footprints remain in the floor polygons.
- Measures rectangular finish changes and rounded circulation boundaries, including loops opening toward any side. It can trace rectilinear exterior recesses from the closed wall network.
- Checks door swings for curved stroke support and wall openings, groups paired leaves, and recognizes sliding leaves and coiling openings separately.

The geometry model covers orthogonal partitions, rectilinear building outlines, rectangular finish insets, and circulation fields with circular corner arcs. The standard ordered legend and a readable graphic scale must be present. The supplied reference assets contain only finish-label glyphs and numeric glyphs. Store identifiers and example takeoff quantities are not used by the executable.

Run the end-to-end example check with:

```sh
python3 /app/takeoff/validate.py /app/examples --unprivileged
```

The validator starts a new process for each PDF, restricts it to two CPUs, compares every quantity with the requested tolerance, and requires exact integer counts. When invoked as root with `--unprivileged`, it runs the tool as uid/gid 65534. The production command does not need the examples directory or the validator.

Recorded accuracy, robustness, and isolated deployment checks are in [VALIDATION.md](VALIDATION.md).
