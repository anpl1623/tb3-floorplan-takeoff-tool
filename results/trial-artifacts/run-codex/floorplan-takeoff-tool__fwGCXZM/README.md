# Cedar Valley Market A101 takeoff

Run with the environment's Python:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json
```

The JSON has the same keys, units and count categories as the supplied estimator
files. The output is replaced atomically after recognition finishes. The tool
uses no network, external commands, downloaded models, example-file lookups or
store-number lookup tables. Its runtime dependencies are the already installed
PyMuPDF, OpenCV, NumPy, SciPy and Shapely packages. All tool code is in this
directory; it can run from any working directory as a non-root user.

For annotated images and intermediate measurements:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json --debug diagnostics
```

## Measurement method

The tool reads the scan pixels, corrects the two drawing axes, and calibrates
distance from the graphic scale. Square VCT grid lines provide a check for
nonuniform print or scanner stretch. It normalizes the paper brightness and
works at a common physical resolution.

Paired wall faces identify the partitions. Line weight and wall-body hatching
separate the partition types. Centerlines extend through door openings and meet
the centerlines of intersecting walls. The exterior follows its centerline.

Wall faces bound the floor geometry. Finish outlines and the legend's hatch
conventions distinguish the eight finishes, including the rounded sales-floor
boundaries. Fixture symbols and their white occlusions are restored to the
surrounding finish; columns also count toward floor area. Partition strips close
door thresholds so opening interiors do not contribute floor area.

Fixture recognition uses physical-size CAD symbol templates, orientation checks,
suppression of duplicate sections, and cooler-bank context. Swing doors require
a straight leaf and its quarter-circle arc; opposing leaves form a pair. Sliding
and coiling doors use their panel strokes. Recognition follows the drawing
conventions shown in these A101 examples rather than offering general-purpose
interpretation of arbitrary architectural sheets.

## Validation

```sh
python3 /app/takeoff/validate.py /app/examples
python3 /app/takeoff/validate.py /app/examples --stress --report validation.json
```

The validator checks every quantity against the requested tolerances and
requires integer counts to match exactly. Stress tests reproduce reduction,
rotation, nonuniform stretch, brightness changes, blur, JPEG noise, and alternate
page orientations. These are checks against the three provided plans; they do
not establish ground truth for unseen stores.
