# Cedar Valley A101 takeoff

Run one sheet at a time:

```sh
python3 /app/takeoff/takeoff.py scan.pdf takeoff.json
```

The output has exactly the `floor_sf`, `wall_lf`, `doors`, and `fixtures` fields
and quantity keys shown in the supplied examples. Areas are rounded to 0.1
square foot, lengths to 0.01 foot, and counts are integers. The destination is
replaced only after all quantities have been calculated successfully. Errors
are reported on stderr with a nonzero exit status.

Everything required at runtime is in this directory, in addition to the
Python packages already installed in the environment: NumPy, OpenCV,
PyMuPDF, SciPy, and Shapely. No network, OCR executable, package installation,
root privileges, example directory, or writable installation directory is
required. The code limits OpenCV and numerical-library threads to two.

## Measurement method

- Extract the embedded scan, orient and straighten the CAD axes, and locate
  the standard 0–32 foot graphic scale. The printed sheet scale and PDF page
  dimensions are not used for measurements. The solid scale intervals also
  support scanner contrast correction; the legend's 12-inch VCT grid checks
  for unequal horizontal and vertical scanner scaling.
- Trace enclosed floor regions at wall faces and finish transitions. Fill
  text, column, equipment and furniture holes back into their enclosing floor
  region, while preserving separately enclosed finish regions. Classify the
  eight standard hatches using scale-normalized texture spectra. Door-opening
  strips are excluded from the room regions.
- Pair wall-face rules, distinguish light stud, solid stud and hatched CMU
  styles, bridge door gaps, and extend wall endpoints to intersecting
  centerlines. Trace the exterior as a closed perimeter, including exterior
  jogs, without deducting openings.
- Locate the numbered circular door references. Associate each reference with
  its wall opening and distinguish swing, pair, coiling and vestibule sliding
  frames by opening and leaf geometry.
- Match the standard fixture blocks at their measured physical size in both
  plan directions. Resolve overlapping 3-foot and 4-foot gondola matches
  together so that a double-sided section is counted once.

The tool targets the drawing standard represented by the supplied archive
samples: its standard scale bar, finish hatches, wall fills, numbered door
references, and fixture blocks. The bundled references describe symbols and
textures independently of store identities, filenames, takeoff totals, or
reference floor geometry.

## Validation

```sh
python3 /app/takeoff/validate.py /app/examples
```

All three supplied examples pass every specified tolerance. Every door and
fixture count matches exactly. The largest observed finish-area error is
0.44%; the largest observed partition/perimeter error is 0.09%. Normal runs
complete in about four seconds in this environment, well below five minutes.
Execution was also checked as UID 65534 with CPU affinity limited to two CPUs.

Additional development checks covered altered scan rotation, resolution,
contrast, a 4% vertical stretch, rotated and relocated fixtures, extra 3-foot
gondolas and cooler doors, and an added exterior notch. Validation results apply to
the supplied scans and these described perturbations.

## Optional audit output

```sh
TAKEOFF_DEBUG=/tmp/takeoff-audit \
  python3 /app/takeoff/takeoff.py scan.pdf takeoff.json
```

This writes the rectified sheet and plan, finish-region and wall-centerline overlays, fixture and
door detections, and an `audit.json` containing the calibration, per-region
areas, centerline segments, detection coordinates and quantities. Audit output
is opt-in and is written only to the directory specified by the caller.
