"""Drafting standards of the architect's offices that drew the chain's stores.

Fixtures and doors are the owner's CAD blocks and look the same on every sheet.
Floor-finish hatches and partition poche follow each office's own standard,
and every sheet shows its office's swatches in the legend, in the owner's
standard order (finishes LVT-1, PC-1, SC-1, EP-1, VCT-1, CT-1, CPT-1, WOM-1;
partitions EXT, P3, P2, P1, where EXT and P3 share the CMU poche).

Pattern specs (all lengths in plan feet, angles in degrees, greys 0 black - 1 white):
  ("planks", width, length, angle)    staggered plank joints
  ("brick", course, length)           running bond, half-length offset
  ("lines", angle, spacing)
  ("dashed", angle, spacing, dash, gap)
  ("double", angle, spacing, gap)     pairs of lines
  ("cross", angle1, angle2, spacing)
  ("dots", spacing, radius)           square grid of dots
  ("stipple", density)                random dots and small triangles per square foot
  ("joints", spacing)                 sparse sawcut grid
  ("tint", grey)                      flat tone (no darker than 0.62, so tints never read as linework)
"""

FINISH_ORDER = ["LVT-1", "PC-1", "SC-1", "EP-1", "VCT-1", "CT-1", "CPT-1", "WOM-1"]


def office(name, finishes, cmu, p2, p1, hatch_grey=0.45, hatch_lw=0.3):
    return dict(name=name, finishes=dict(zip(FINISH_ORDER, finishes)), walls={"CMU": cmu, "P2": p2, "P1": p1},
                hatch_grey=hatch_grey, hatch_lw=hatch_lw)


OFFICES = {
    # the three offices the example sheets come from
    "NORTHLINE": office("NORTHLINE RETAIL ARCHITECTS", [
        ("planks", 7 / 12, 4.0, 0), ("joints", 10.0), ("stipple", 0.9), ("lines", 45, 1.0),
        ("cross", 0, 90, 1.0), ("cross", 45, 135, 0.75), ("lines", 135, 0.5), ("lines", 0, 0.25)],
        cmu=("lines", 45, 0.32), p2=("tint", 0.62), p1=("tint", 0.88)),
    "HARGROVE": office("HARGROVE + WEISS ARCHITECTS", [
        ("brick", 0.5, 2.0), ("joints", 8.0), ("dots", 1.0, 0.07), ("cross", 45, 135, 1.25),
        ("cross", 0, 90, 1.0), ("dots", 0.5, 0.05), ("tint", 0.75), ("lines", 90, 0.25)],
        cmu=("cross", 45, 135, 0.4), p2=("lines", 45, 0.18), p1=("tint", 0.8), hatch_grey=0.4, hatch_lw=0.35),
    "MERIDIAN": office("MERIDIAN DESIGN GROUP", [
        ("dashed", 0, 0.6, 1.5, 0.5), ("joints", 12.0), ("stipple", 1.4), ("lines", 135, 0.75),
        ("cross", 0, 90, 1.5), ("double", 45, 1.0, 0.12), ("dots", 0.75, 0.07), ("tint", 0.66)],
        cmu=("tint", 0.66), p2=("tint", 0.84), p1=("lines", 135, 0.25), hatch_grey=0.5, hatch_lw=0.3),
    # offices whose stores are only in the hidden part of the archive
    "CALLOWAY": office("CALLOWAY ARCHITECTURE", [
        ("planks", 0.5, 3.0, 90), ("joints", 10.0), ("dots", 2.0, 0.09), ("lines", 45, 1.0),
        ("cross", 0, 90, 0.66), ("lines", 45, 0.66), ("tint", 0.78), ("lines", 90, 0.2)],
        cmu=("lines", 135, 0.3), p2=("tint", 0.66), p1=("tint", 0.85), hatch_grey=0.42, hatch_lw=0.32),
    "BRANDT": office("BRANDT & OKAFOR", [
        ("lines", 0, 0.6), ("joints", 10.0), ("stipple", 0.7), ("cross", 45, 135, 1.0),
        ("cross", 0, 90, 1.0), ("dots", 0.5, 0.06), ("tint", 0.82), ("tint", 0.66)],
        cmu=("cross", 0, 90, 0.33), p2=("lines", 45, 0.2), p1=("tint", 0.82), hatch_grey=0.48, hatch_lw=0.28),
    "STILLWATER": office("STILLWATER PARTNERS", [
        ("double", 0, 0.75, 0.1), ("joints", 9.0), ("dots", 1.25, 0.07), ("dashed", 45, 0.8, 1.0, 0.4),
        ("cross", 0, 90, 1.25), ("cross", 45, 135, 0.6), ("lines", 135, 0.4), ("lines", 0, 0.2)],
        cmu=("lines", 45, 0.4), p2=("cross", 45, 135, 0.25), p1=("tint", 0.8), hatch_grey=0.44, hatch_lw=0.3),
    "KESSLER": office("KESSLER LYONS STUDIO", [
        ("brick", 0.4, 1.6), ("joints", 10.0), ("dots", 0.8, 0.05), ("lines", 135, 1.0),
        ("cross", 0, 90, 1.0), ("dots", 0.5, 0.05), ("lines", 45, 0.4), ("tint", 0.66)],
        cmu=("tint", 0.66), p2=("tint", 0.84), p1=("lines", 45, 0.3), hatch_grey=0.46, hatch_lw=0.3),
    "ARDEN": office("ARDEN & FIELD ARCHITECTS", [
        ("planks", 0.6, 4.0, 0), ("joints", 10.0), ("stipple", 1.1), ("lines", 45, 0.8),
        ("cross", 0, 90, 0.8), ("cross", 45, 135, 0.8), ("tint", 0.7), ("lines", 0, 0.3)],
        cmu=("cross", 45, 135, 0.35), p2=("lines", 135, 0.18), p1=("tint", 0.75), hatch_grey=0.4, hatch_lw=0.34),
}
EXAMPLE_OFFICES = ["NORTHLINE", "HARGROVE", "MERIDIAN"]
HIDDEN_OFFICES = ["CALLOWAY", "BRANDT", "STILLWATER", "KESSLER", "ARDEN"]
