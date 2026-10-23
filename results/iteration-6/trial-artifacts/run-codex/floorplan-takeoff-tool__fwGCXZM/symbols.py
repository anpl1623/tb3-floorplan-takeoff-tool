"""Scale-normalized CAD symbol recognition."""
import cv2
import numpy as np


PPF = 24
PAD = 4
UPSAMPLE = 4


def symbol_template(kind):
    dims = {"self_checkouts": (3, 3), "produce_tables": (5, 4),
            "checkouts": (3.7, 12), "cooler_doors": (2.3, .6),
            "endcaps": (4, 1.5), "gondola_sections_4ft": (4, 4),
            "gondola_sections_3ft": (4, 3)}
    w, h = dims[kind]
    scale = PPF*UPSAMPLE
    pad = PAD*UPSAMPLE
    a = np.full((round(h*scale)+2*pad+1, round(w*scale)+2*pad+1), 255, np.uint8)
    def pt(x, y):
        return round(x*scale)+pad, round(y*scale)+pad
    def line(x0, y0, x1, y1):
        cv2.line(a, pt(x0, y0), pt(x1, y1), 0, 4, cv2.LINE_AA)
    def rect(x0, y0, x1, y1):
        cv2.rectangle(a, pt(x0, y0), pt(x1, y1), 0, 4, cv2.LINE_AA)
    if kind == "self_checkouts":
        rect(0, 0, w, h)
        cv2.circle(a, pt(w/2, h/2), round(.8*scale), 0, 4, cv2.LINE_AA)
    elif kind == "produce_tables":
        rect(0, 0, w, h)
        line(0, 0, w, h)
        line(0, h, w, 0)
    elif kind == "checkouts":
        rect(0, 0, 2.5, 12)
        line(0, 9, 2.5, 9)
        line(0, 9, 2.5, 12)
        line(0, 12, 2.5, 9)
        line(1.25, .4, 1.25, 8.6)
        rect(2.5, 4.8, 3.7, 6)
    elif kind == "cooler_doors":
        rect(0, 0, w, h)
        line(0, 0, w, h)
    elif kind == "endcaps":
        rect(0, 0, w, h)
        line(0, h, w, 0)
    else:
        rect(0, 0, w, h)
        line(w/2, 0, w/2, h)
    a = cv2.resize(a, (round(w*PPF)+2*PAD+1, round(h*PPF)+2*PAD+1), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (5, 5), .65)
    return a


def peaks(score, template, threshold=.5):
    result = []
    h, w = template.shape
    work = score.copy()
    for _ in range(1000):
        _, best, _, (x, y) = cv2.minMaxLoc(work)
        if best < threshold:
            break
        result.append({"x": x, "y": y, "w": w, "h": h, "score": float(best)})
        radius_x, radius_y = max(8, w//3), max(8, h//3)
        work[max(0, y-radius_y):y+radius_y+1, max(0, x-radius_x):x+radius_x+1] = -1
    return result


def cooler_body_side(a, m):
    """A cooler's unjointed white body distinguishes it from floor hatching."""
    x, y = round(m["x"]+m["w"]/2), round(m["y"]+m["h"]/2)
    if m["w"] > m["h"]:
        patches = [a[max(0, y-52):max(1, y-20), max(0, x-18):x+19],
                   a[y+20:y+52, max(0, x-18):x+19]]
    else:
        patches = [a[max(0, y-18):y+19, max(0, x-52):max(1, x-20)],
                   a[max(0, y-18):y+19, x+20:x+52]]
    candidates = [(float(p.std()), side) for p, side in zip(patches, [-1, 1])
                  if p.size > 400 and p.mean() > 240]
    if not candidates:
        return None
    spread, side = min(candidates)
    return side if spread < 9 else None


def section_outline(a, m):
    # Both section ends must be present at the measured three- or four-foot
    # spacing. This rejects a shorter window cropped from a longer section.
    l, t = m["x"]+PAD, m["y"]+PAD
    r, b = m["x"]+m["w"]-PAD-1, m["y"]+m["h"]-PAD-1
    profiles = [a[t-2:t+3, l+5:r-4].min(0), a[b-2:b+3, l+5:r-4].min(0),
                a[t+5:b-4, l-2:l+3].min(1), a[t+5:b-4, r-2:r+3].min(1)]
    return all(p.size and (p < 225).mean() > .8 for p in profiles)


def recognize_fixtures(a, debug=None, threshold=None):
    results = {}
    smooth = cv2.GaussianBlur(a, (3, 3), .4)
    cutoffs = {"self_checkouts": .8, "produce_tables": .76, "checkouts": .78,
               "cooler_doors": .68, "endcaps": .72,
               "gondola_sections_4ft": .76, "gondola_sections_3ft": .76}
    for kind in ["self_checkouts", "produce_tables", "checkouts", "cooler_doors", "endcaps",
                 "gondola_sections_4ft", "gondola_sections_3ft"]:
        template = symbol_template(kind)
        matches = []
        rotations = range(4) if kind in ["checkouts", "cooler_doors", "endcaps"] else range(2)
        flips = [False, True] if kind in ["checkouts", "cooler_doors", "endcaps"] else [False]
        for flipped in flips:
            for rotation in rotations:
                t = np.ascontiguousarray(np.rot90(np.fliplr(template) if flipped else template, rotation))
                score = cv2.matchTemplate(smooth, t, cv2.TM_CCOEFF_NORMED)
                cutoff = threshold if threshold is not None else cutoffs[kind]
                found = peaks(score, t, cutoff)
                for f in found:
                    f["rotation"] = rotation
                    f["flipped"] = flipped
                if kind == "cooler_doors":
                    for f in found:
                        f["body_side"] = cooler_body_side(a, f)
                    found = [f for f in found if f["body_side"] is not None]
                if kind.startswith("gondola_sections"):
                    found = [f for f in found if section_outline(a, f)]
                matches.extend(found)
        matches.sort(key=lambda v: -v["score"])
        unique = []
        for m in matches:
            cx, cy = m["x"]+m["w"]/2, m["y"]+m["h"]/2
            def duplicate(o):
                if np.hypot(cx-o["x"]-o["w"]/2, cy-o["y"]-o["h"]/2) < 12:
                    return True
                dx = max(0, min(m["x"]+m["w"], o["x"]+o["w"])-max(m["x"], o["x"]))
                dy = max(0, min(m["y"]+m["h"], o["y"]+o["h"])-max(m["y"], o["y"]))
                return dx*dy > .55*min(m["w"]*m["h"], o["w"]*o["h"])
            if not any(duplicate(o) for o in unique):
                unique.append(m)
        if kind == "cooler_doors":
            strong = [m for m in unique if m["score"] >= .72]
            accepted = strong.copy()
            for m in unique:
                if m["score"] >= .72:
                    continue
                horizontal = m["w"] > m["h"]
                c = np.array([m["x"]+m["w"]/2, m["y"]+m["h"]/2])
                for neighbor in strong:
                    if horizontal != (neighbor["w"] > neighbor["h"]) or m["body_side"] != neighbor["body_side"]:
                        continue
                    d = abs(c-np.array([neighbor["x"]+neighbor["w"]/2, neighbor["y"]+neighbor["h"]/2]))
                    along, across = d if horizontal else d[::-1]
                    if across < 3 and abs(along-60) < 4:
                        accepted.append(m)
                        break
            unique = accepted
        if kind == "gondola_sections_3ft":
            # A cropped four-foot section can resemble a three-foot section.
            # Assign the full section once, using the stronger complete outline.
            retained = []
            for m in unique:
                area = m["w"]*m["h"]
                for other in results["gondola_sections_4ft"]:
                    dx = max(0, min(m["x"]+m["w"], other["x"]+other["w"])-max(m["x"], other["x"]))
                    dy = max(0, min(m["y"]+m["h"], other["y"]+other["h"])-max(m["y"], other["y"]))
                    if dx*dy > area*.6:
                        break
                else:
                    retained.append(m)
            unique = retained
        results[kind] = unique
        if debug is not None:
            cv2.imwrite(str(debug / ("template_"+kind+".png")), template)
            vis = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            for i, m in enumerate(unique):
                x, y, w, h = [m[k] for k in ["x", "y", "w", "h"]]
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis, f"{i}:{m['score']:.2f}", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            cv2.imwrite(str(debug / (kind+".png")), vis)
    return results


def fixture_exclusion(a, fixtures):
    """Mask the symbol and white floor occlusion, including cooler-bank bodies."""
    mask = np.zeros(a.shape, np.uint8)
    height, width = a.shape
    def rectangle(l, t, r, b):
        cv2.rectangle(mask, (max(0, round(l)), max(0, round(t))),
                      (min(width-1, round(r)), min(height-1, round(b))), 1, -1)
    for kind, matches in fixtures.items():
        for m in matches:
            l, t, r, b = m["x"], m["y"], m["x"]+m["w"], m["y"]+m["h"]
            if kind != "cooler_doors":
                rectangle(l, t, r, b)
                continue
            # Door-panel pitch is 2.5 ft, with a 0.1 ft inset on each side.
            # The two-foot-four-inch body is white and has no internal joints.
            if m["w"] > m["h"]:
                if m["body_side"] == -1:
                    rectangle(l-2, t-58, r+2, b)
                else:
                    rectangle(l-2, t, r+2, b+58)
            else:
                if m["body_side"] == -1:
                    rectangle(l-58, t-2, r, b+2)
                else:
                    rectangle(l, t-2, r+58, b+2)
    return mask
