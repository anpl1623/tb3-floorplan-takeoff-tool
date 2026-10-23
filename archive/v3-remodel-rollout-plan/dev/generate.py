#!/usr/bin/env python3
"""Generate the Cedar Valley Market 2027 remodel program instance.

Writes the CSV files the agent gets under environment/data/:
  stores.csv     store, city, district, weeks, permit_week, disruption, blackout_weeks
  sales.csv      store, week, sales  (forecast net sales for each fiscal week 1-52)
  crews.csv      crew, home, weekly_rate
  miles.csv      from, to, miles  (road miles between every pair of stores and crew homes)

Everything is seeded, so the files are reproducible. The same files are copied
to tests/data for the verifier, along with dev/rollout.py (the program rules and
cost model) as tests/rollout.py and solution/rollout.py.

Usage: python dev/generate.py
"""

import csv
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "remodel-rollout-plan"
DATA = TASK / "environment" / "data"

# (city, lat, lon, district, profile, size) ; size 1 small rural .. 3 large metro
TOWNS = [
    # D1 Central Iowa
    ("Des Moines", 41.59, -93.62, "D1", "metro", 3), ("Ankeny", 41.73, -93.60, "D1", "metro", 3),
    ("Ames", 42.03, -93.62, "D1", "college", 2), ("Newton", 41.70, -93.05, "D1", "rural", 1),
    ("Pella", 41.41, -92.92, "D1", "rural", 1), ("Indianola", 41.36, -93.56, "D1", "rural", 1),
    ("Marshalltown", 42.05, -92.91, "D1", "rural", 2), ("Grinnell", 41.74, -92.72, "D1", "college", 1),
    ("Ottumwa", 41.02, -92.41, "D1", "rural", 2), ("Carroll", 42.07, -94.87, "D1", "rural", 1),
    ("West Des Moines", 41.57, -93.75, "D1", "metro", 3), ("Urbandale", 41.63, -93.71, "D1", "metro", 2),
    # D2 Eastern Iowa
    ("Cedar Rapids", 41.98, -91.67, "D2", "metro", 3), ("Marion", 42.03, -91.60, "D2", "metro", 2),
    ("Iowa City", 41.66, -91.53, "D2", "college", 3), ("Coralville", 41.68, -91.58, "D2", "college", 2),
    ("Waterloo", 42.49, -92.34, "D2", "metro", 2), ("Cedar Falls", 42.53, -92.45, "D2", "college", 2),
    ("Dubuque", 42.50, -90.66, "D2", "metro", 2), ("Decorah", 43.30, -91.79, "D2", "college", 1),
    ("Muscatine", 41.42, -91.04, "D2", "rural", 2), ("Burlington", 40.81, -91.11, "D2", "rural", 2),
    ("Charles City", 43.07, -92.67, "D2", "rural", 1), ("Mount Vernon", 41.92, -91.42, "D2", "college", 1),
    # D3 Northern and Western Iowa
    ("Mason City", 43.15, -93.20, "D3", "rural", 2), ("Clear Lake", 43.14, -93.38, "D3", "resort", 1),
    ("Fort Dodge", 42.50, -94.17, "D3", "rural", 2), ("Spencer", 43.14, -95.14, "D3", "rural", 1),
    ("Spirit Lake", 43.42, -95.10, "D3", "resort", 1), ("Storm Lake", 42.64, -95.21, "D3", "rural", 1),
    ("Sioux City", 42.50, -96.40, "D3", "metro", 3), ("Council Bluffs", 41.26, -95.86, "D3", "metro", 2),
    ("Le Mars", 42.79, -96.17, "D3", "rural", 1), ("Algona", 43.07, -94.23, "D3", "rural", 1),
    ("Okoboji", 43.39, -95.14, "D3", "resort", 1), ("Denison", 42.02, -95.36, "D3", "rural", 1),
    # D4 Quad Cities and Illinois
    ("Davenport", 41.52, -90.58, "D4", "metro", 3), ("Bettendorf", 41.52, -90.52, "D4", "metro", 2),
    ("Moline", 41.51, -90.52, "D4", "metro", 2), ("Galesburg", 40.95, -90.37, "D4", "rural", 2),
    ("Peoria", 40.69, -89.59, "D4", "metro", 3), ("Bloomington", 40.48, -88.99, "D4", "metro", 3),
    ("Champaign", 40.12, -88.24, "D4", "college", 3), ("Macomb", 40.46, -90.67, "D4", "college", 1),
    ("Quincy", 39.94, -91.41, "D4", "rural", 2), ("Kewanee", 41.25, -89.92, "D4", "rural", 1),
    ("Ottawa", 41.35, -88.84, "D4", "rural", 1), ("Galena", 42.42, -90.43, "D4", "resort", 1),
    # D5 Southern Wisconsin and Northern Illinois
    ("Madison", 43.07, -89.40, "D5", "metro", 3), ("Middleton", 43.10, -89.50, "D5", "metro", 2),
    ("Janesville", 42.68, -89.02, "D5", "metro", 2), ("Beloit", 42.51, -89.03, "D5", "rural", 2),
    ("Rockford", 42.27, -89.09, "D5", "metro", 3), ("Freeport", 42.30, -89.62, "D5", "rural", 1),
    ("DeKalb", 41.93, -88.75, "D5", "college", 2), ("Dixon", 41.84, -89.48, "D5", "rural", 1),
    ("Platteville", 42.73, -90.48, "D5", "college", 1), ("Lake Geneva", 42.59, -88.43, "D5", "resort", 2),
    ("Wisconsin Dells", 43.63, -89.77, "D5", "resort", 1), ("Baraboo", 43.47, -89.74, "D5", "resort", 1),
    # D6 Western Wisconsin and Southern Minnesota
    ("La Crosse", 43.81, -91.24, "D6", "college", 2), ("Eau Claire", 44.81, -91.50, "D6", "college", 2),
    ("Menomonie", 44.88, -91.92, "D6", "college", 1), ("Tomah", 43.98, -90.50, "D6", "rural", 1),
    ("Rochester", 44.02, -92.47, "D6", "metro", 3), ("Winona", 44.05, -91.64, "D6", "college", 1),
    ("Mankato", 44.16, -94.00, "D6", "college", 2), ("Albert Lea", 43.65, -93.37, "D6", "rural", 1),
    ("Austin", 43.67, -92.97, "D6", "rural", 1), ("Owatonna", 44.08, -93.23, "D6", "rural", 2),
    ("Red Wing", 44.56, -92.53, "D6", "resort", 1), ("Stevens Point", 44.52, -89.57, "D6", "college", 2),
]

CREWS = [  # crew, home, weekly rate (labor, supervision, per diem, equipment)
    ("C1", "Des Moines", 19400), ("C2", "Cedar Rapids", 18900), ("C3", "Davenport", 19100),
    ("C4", "Madison", 20600), ("C5", "Rochester", 19800), ("C6", "Peoria", 18700),
    ("C7", "Sioux City", 18200), ("C8", "Eau Claire", 19300), ("C9", "Rockford", 20100),
]
HOMES = {"Des Moines": (41.59, -93.62), "Cedar Rapids": (41.98, -91.67), "Davenport": (41.52, -90.58),
         "Madison": (43.07, -89.40), "Rochester": (44.02, -92.47), "Peoria": (40.69, -89.59),
         "Sioux City": (42.50, -96.40), "Eau Claire": (44.81, -91.50), "Rockford": (42.27, -89.09)}

# Local events and lease terms that close a store to construction for specific weeks.
EVENT_BLACKOUTS = {
    "Des Moines": [32, 33], "Ankeny": [32, 33], "West Des Moines": [32, 33], "Urbandale": [32, 33],
    "Pella": [18, 19], "Ames": [33, 34], "Iowa City": [33, 34], "Coralville": [33, 34],
    "Cedar Falls": [34], "Champaign": [33, 34], "Madison": [35, 36], "La Crosse": [39, 40],
    "Galena": [40, 41], "Spirit Lake": [27], "Okoboji": [27, 28], "Clear Lake": [27],
    "Wisconsin Dells": [26, 27], "Lake Geneva": [26, 27], "Decorah": [30], "Le Mars": [24],
    "Dubuque": [25], "Rochester": [21, 22], "Mankato": [34], "Quincy": [24, 25],
}
LATE_PERMITS = {"Sioux City": 24, "Peoria": 20, "Rockford": 26, "Dubuque": 18, "Eau Claire": 22,
                "Bloomington": 16, "Waterloo": 19, "Madison": 14, "Council Bluffs": 28, "Mankato": 21}


def road_miles(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h)) * 1.19       # circuity factor for rural highway networks


def profile(kind, rng, amp=1.0):
    w = np.arange(1, 53)
    if kind == "metro":
        f = 1 + 0.06 * np.sin((w - 14) / 52 * 2 * np.pi) + 0.10 * np.isin(w, [13, 21, 27, 36, 47])
        f -= 0.08 * (w <= 8)
    elif kind == "college":
        f = 1 + 0.10 * ((w >= 35) & (w <= 46)) + 0.06 * ((w >= 2) & (w <= 18))
        f -= 0.30 * ((w >= 20) & (w <= 33)) + 0.18 * (w == 11)
    elif kind == "resort":
        f = 1 + 0.55 * np.exp(-((w - 28) / 6.5) ** 2) - 0.15 * ((w <= 16) | ((w >= 40) & (w <= 46)))
        f += 0.10 * np.isin(w, [21, 36])
    else:  # rural / agricultural
        f = 1 + 0.05 * np.sin((w - 10) / 52 * 2 * np.pi) + 0.07 * ((w >= 38) & (w <= 43)) - 0.06 * (w <= 7)
        f += 0.08 * np.isin(w, [13, 27, 36])
    f = 1 + amp * (f - 1)
    f = f * (1 + rng.normal(0, 0.025, 52))
    return f


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--crews", type=int, default=len(CREWS))
    ap.add_argument("--season-amp", type=float, default=1.8)
    ap.add_argument("--out", help="write only the data files, to this directory (calibration variants)")
    ap.add_argument("--scale", type=int, default=1,
                    help="calibration only: k stores per town (jittered) and k crews per home, k-times the districts")
    args = ap.parse_args()
    crews = CREWS[:args.crews]
    data = Path(args.out) if args.out else DATA
    rng = np.random.default_rng(20270104)
    data.mkdir(parents=True, exist_ok=True)
    coords = {}
    stores = []
    for n, (city, lat, lon, district, kind, size) in enumerate(TOWNS, start=1):
        sid = f"S{n:03d}"
        coords[sid] = (lat, lon)
        base = {1: 290_000, 2: 520_000, 3: 860_000}[size] * rng.uniform(0.85, 1.15)
        weeks = int(rng.choice({1: [3, 3, 4], 2: [4, 4, 5], 3: [5, 6, 6]}[size]))
        disruption = round(float(rng.uniform(0.07, 0.13) + 0.01 * size), 3)
        permit = LATE_PERMITS.get(city, int(rng.integers(1, 13)))
        blackout = EVENT_BLACKOUTS.get(city, [])
        sales = np.round(base * profile(kind, rng, args.season_amp), -2)
        stores.append(dict(store=sid, city=city, district=district, weeks=weeks, permit_week=permit,
                           disruption=disruption, blackout_weeks=" ".join(map(str, blackout)), sales=sales))
    for copy in range(2, args.scale + 1):
        for (city, lat, lon, district, kind, size) in TOWNS:
            sid = f"S{len(stores) + 1:03d}"
            coords[sid] = (lat + rng.uniform(-0.25, 0.25), lon + rng.uniform(-0.3, 0.3))
            size = int(rng.integers(1, 4))
            base = {1: 290_000, 2: 520_000, 3: 860_000}[size] * rng.uniform(0.85, 1.15)
            weeks = int(rng.choice({1: [3, 3, 4], 2: [4, 4, 5], 3: [5, 6, 6]}[size]))
            disruption = round(float(rng.uniform(0.07, 0.13) + 0.01 * size), 3)
            stores.append(dict(store=sid, city=f"{city} area {copy}", district=f"{district}{'abcd'[copy - 1]}",
                               weeks=weeks, permit_week=int(rng.integers(1, 13)), disruption=disruption,
                               blackout_weeks=" ".join(map(str, EVENT_BLACKOUTS.get(city, []))),
                               sales=np.round(base * profile(kind, rng, args.season_amp), -2)))
        crews = crews + [(f"C{len(crews) + n + 1}", home, rate + int(rng.integers(-6, 7)) * 100)
                         for n, (_, home, rate) in enumerate(CREWS[:args.crews])]
    for crew, home, _ in crews:
        coords[home] = HOMES[home]

    with open(data / "stores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["store", "city", "district", "weeks", "permit_week", "disruption", "blackout_weeks"])
        for s in stores:
            w.writerow([s["store"], s["city"], s["district"], s["weeks"], s["permit_week"], s["disruption"],
                        s["blackout_weeks"]])
    with open(data / "sales.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["store", "week", "sales"])
        for s in stores:
            for wk, v in enumerate(s["sales"], start=1):
                w.writerow([s["store"], wk, int(v)])
    with open(data / "crews.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crew", "home", "weekly_rate"])
        w.writerows(crews)
    places = [s["store"] for s in stores] + [home for _, home, _ in crews]
    with open(data / "miles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from", "to", "miles"])
        for a in places:
            for b in places:
                if a != b:
                    w.writerow([a, b, round(road_miles(coords[a], coords[b]))])
    if args.out:
        return
    shutil.copytree(DATA, TASK / "tests" / "data", dirs_exist_ok=True)
    for dest in ("tests", "solution"):
        shutil.copy(ROOT / "dev" / "rollout.py", TASK / dest / "rollout.py")
    print(f"{len(stores)} stores, {len(crews)} crews, "
          f"{sum(s['weeks'] for s in stores)} remodel weeks -> {data}")


if __name__ == "__main__":
    main()
