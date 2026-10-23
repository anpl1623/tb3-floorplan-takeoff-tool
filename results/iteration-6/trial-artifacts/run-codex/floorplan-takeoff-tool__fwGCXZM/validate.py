#!/usr/bin/env python3
"""Check takeoffs against paired PDFs and estimator JSON files, offline."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def compare(actual, expected):
    failures = []
    errors = {"floor_sf": 0., "wall_lf": 0.}
    for category, quantities in expected.items():
        if category not in actual or set(actual[category]) != set(quantities):
            failures.append(f"{category}: missing or extra quantities")
            continue
        for code, target in quantities.items():
            value = actual[category][code]
            error = abs(value-target)
            if category == "floor_sf":
                tolerance = max(8., .02*target)
            elif category == "wall_lf":
                tolerance = max(2., .015*target)
            else:
                tolerance = 0.
                if type(value) is not int:
                    failures.append(f"{category}.{code}: count is not an integer")
            if category in errors:
                errors[category] = max(errors[category], error)
            if error > tolerance:
                failures.append(f"{category}.{code}: got {value}, expected {target}, tolerance {tolerance:g}")
    return failures, errors


def transformed_pdf(source, destination, variant):
    import cv2
    import fitz
    import numpy as np
    from takeoff import image_from_pdf

    a = image_from_pdf(source)
    if variant == "upside_down":
        a = np.rot90(a, 2).copy()
    elif variant == "portrait":
        a = np.rot90(a).copy()
    else:
        parameters = {
            "reduced_rotated": (.8, .8, 1.7, .92, 8., .35, 2., 80),
            "stretched": (.96, 1.04, -1.1, 1., 0., 0., 0., 85),
            "degraded": (.7, .7, -.9, .9, 12., .6, 3., 60)}
        sx, sy, angle, contrast, offset, blur, noise, quality = parameters[variant]
        h, w = a.shape
        center = np.array([w/2, h/2])
        matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
        matrix[:, :2] = matrix[:, :2]@np.diag([sx, sy])
        matrix[:, 2] = center-matrix[:, :2]@center
        a = cv2.warpAffine(a, matrix, (w, h), borderValue=255)
        if blur:
            a = cv2.GaussianBlur(a, (0, 0), blur)
        if contrast != 1 or offset or noise:
            a = a.astype(np.float32)*contrast+offset
            if noise:
                a += np.random.default_rng(47).normal(0, noise, a.shape).astype(np.float32)
            a = np.clip(a, 0, 255).astype(np.uint8)
    quality = 90 if variant in ["upside_down", "portrait"] else quality
    raw = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    with fitz.open() as doc:
        page = doc.new_page(width=a.shape[1]*.36, height=a.shape[0]*.36)
        page.insert_image(page.rect, stream=raw)
        doc.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path, help="Directory containing *.pdf and *.takeoff.json pairs")
    parser.add_argument("--stress", action="store_true", help="Also test reduction, rotation, stretch, noise and page orientation")
    parser.add_argument("--report", type=Path, help="Write detailed validation results as JSON")
    args = parser.parse_args()
    references = sorted(args.examples.glob("*.takeoff.json"))
    if not references:
        parser.error("No estimator takeoffs found")
    program = Path(__file__).with_name("takeoff.py")
    records = []
    with tempfile.TemporaryDirectory(prefix="takeoff-validation-") as temp:
        work = Path(temp)
        for reference in references:
            source = reference.with_name(reference.name.removesuffix(".takeoff.json")+".pdf")
            expected = json.loads(reference.read_text())
            variants = ["original"]
            if args.stress:
                variants += ["reduced_rotated", "stretched", "degraded", "upside_down", "portrait"]
            for variant in variants:
                scan = source
                if variant != "original":
                    scan = work / f"{source.stem}-{variant}.pdf"
                    transformed_pdf(source, scan, variant)
                output = work / "takeoff.json"
                start = time.monotonic()
                run = subprocess.run([sys.executable, str(program), str(scan), str(output)],
                                     capture_output=True, text=True, timeout=300)
                seconds = time.monotonic()-start
                if run.returncode:
                    failures, errors = [run.stderr.strip()], {}
                else:
                    actual = json.loads(output.read_text())
                    failures, errors = compare(actual, expected)
                record = dict(scan=source.name, variant=variant, passed=not failures,
                              seconds=round(seconds, 3), max_absolute_error=errors, failures=failures)
                records.append(record)
                print(f"{'PASS' if not failures else 'FAIL'} {source.name} {variant}: {seconds:.2f}s", flush=True)
                for failure in failures:
                    print("  "+failure, flush=True)
    if args.report:
        args.report.write_text(json.dumps(records, indent=2)+"\n")
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
