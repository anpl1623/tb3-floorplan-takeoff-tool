#!/usr/bin/env python3
"""Generate the synthetic Cedar Valley Market data set for pos-promo-parity.

Writes the agent-visible files into tasks/pos-promo-parity/environment/data
and byte-identical copies of the October inputs into tasks/pos-promo-parity/tests.
History receipts are produced by the reference engine (solution/engine.py),
which plays the role of the decommissioned legacy POS.

Usage: python3 dev/generate.py
"""

import csv
import datetime as dt
import importlib.util
import json
import math
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "pos-promo-parity"
DATA = TASK / "environment" / "data"
TESTS = TASK / "tests"

spec = importlib.util.spec_from_file_location("engine", TASK / "solution" / "engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

SEED = 20261021
SEARCH_CAP = 20000          # max deal-assignment leaves per department
HISTORY_BASKETS = 9000
PENDING_BASKETS = 3000

STORES = {
    "101": {"name": "Cedar Valley Market #101 Riverside", "rates": {"F": 0, "T": 825, "P": 1025}},
    "102": {"name": "Cedar Valley Market #102 Hillcrest", "rates": {"F": 0, "T": 775, "P": 975}},
    "103": {"name": "Cedar Valley Market #103 Lakeport", "rates": {"F": 150, "T": 900, "P": 1100}},
}
DEPT_CLASS = {
    "PRODUCE": "F", "MEAT": "F", "DAIRY": "F", "BAKERY": "F", "FROZEN": "F", "PANTRY": "F",
    "BEVERAGE": "T", "SNACKS": "T", "HOUSEHOLD": "T", "HBC": "T", "DELI": "P",
}

# family key -> (dept, sold_by, [(description, price, member_price)])
FAMILIES = {
    "fizz2l": ("BEVERAGE", "each", [
        ("FIZZ COLA 2L", 249, None), ("FIZZ DIET COLA 2L", 249, None),
        ("FIZZ CHERRY COLA 2L", 249, None), ("FIZZ LEMON LIME 2L", 249, None)]),
    "fizz12": ("BEVERAGE", "each", [
        ("FIZZ COLA 12PK CAN", 699, 599), ("FIZZ DIET COLA 12PK CAN", 699, 599),
        ("FIZZ ZERO 12PK CAN", 699, 599)]),
    "peak": ("BEVERAGE", "each", [
        ("PEAK SPARKLING LIME 8PK", 499, None), ("PEAK SPARKLING BERRY 8PK", 499, None),
        ("PEAK SPARKLING PLAIN 8PK", 449, None), ("PEAK SPARKLING GRAPEFRUIT 8PK", 499, None)]),
    "volt": ("BEVERAGE", "each", [
        ("VOLT ENERGY ORIGINAL 16OZ", 299, None), ("VOLT ENERGY SUGAR FREE 16OZ", 299, None),
        ("VOLT ENERGY TROPICAL 16OZ", 329, None)]),
    "brookside": ("BEVERAGE", "each", [
        ("BROOKSIDE ICED TEA LEMON 1.5L", 229, None), ("BROOKSIDE ICED TEA PEACH 1.5L", 229, None)]),
    "crunchos": ("SNACKS", "each", [
        ("CRUNCHOS TORTILLA CHIPS 13OZ", 499, None), ("CRUNCHOS LIME TORTILLA 13OZ", 499, None),
        ("CRUNCHOS FAMILY SIZE 20OZ", 699, None), ("CRUNCHOS SCOOPS 10OZ", 449, None)]),
    "kettle": ("SNACKS", "each", [
        ("GOLDEN KETTLE SEA SALT 8OZ", 429, None), ("GOLDEN KETTLE BBQ 8OZ", 429, None),
        ("GOLDEN KETTLE SALT VINEGAR 8OZ", 429, None), ("GOLDEN KETTLE JALAPENO 8OZ", 449, None)]),
    "choco": ("SNACKS", "each", [
        ("CHOCO BAR MILK 1.5OZ", 149, None), ("CHOCO BAR DARK 1.5OZ", 149, None),
        ("CHOCO BAR ALMOND 1.5OZ", 169, None)]),
    "salsa": ("SNACKS", "each", [
        ("EL CAMINO SALSA MILD 16OZ", 379, 329), ("EL CAMINO SALSA HOT 16OZ", 379, 329)]),
    "yogurt": ("DAIRY", "each", [
        ("ALPINE GREEK YOGURT PLAIN 5.3OZ", 129, None), ("ALPINE GREEK YOGURT STRAWBERRY 5.3OZ", 129, None),
        ("ALPINE GREEK YOGURT BLUEBERRY 5.3OZ", 129, None), ("ALPINE GREEK YOGURT VANILLA 5.3OZ", 129, None),
        ("ALPINE GREEK YOGURT PEACH 5.3OZ", 139, None)]),
    "cheese": ("DAIRY", "each", [
        ("MEADOW SHREDDED CHEDDAR 8OZ", 399, 299), ("MEADOW SHREDDED MOZZARELLA 8OZ", 399, 299),
        ("MEADOW SHREDDED MEXICAN 8OZ", 399, 299)]),
    "pasta": ("PANTRY", "each", [
        ("BELLA SPAGHETTI 1LB", 189, None), ("BELLA PENNE 1LB", 189, None),
        ("BELLA ROTINI 1LB", 189, None), ("BELLA FARFALLE 1LB", 199, None)]),
    "sauce": ("PANTRY", "each", [
        ("BELLA MARINARA 24OZ", 349, None), ("BELLA TOMATO BASIL 24OZ", 349, None),
        ("BELLA FOUR CHEESE 24OZ", 379, None)]),
    "cereal": ("PANTRY", "each", [
        ("SUNNY OS CEREAL 18OZ", 499, 449), ("HONEY CLUSTERS CEREAL 14OZ", 529, 449),
        ("BRAN FLAKES CEREAL 16OZ", 449, None)]),
    "pizza": ("FROZEN", "each", [
        ("POLAR PIZZA PEPPERONI", 699, None), ("POLAR PIZZA CHEESE", 649, None),
        ("POLAR PIZZA SUPREME", 749, None)]),
    "icecream": ("FROZEN", "each", [
        ("FROSTY ICE CREAM VANILLA 1.5QT", 549, 449), ("FROSTY ICE CREAM CHOCOLATE 1.5QT", 549, 449),
        ("FROSTY COOKIES N CREAM 1.5QT", 579, 479), ("FROSTY MINT CHIP 1.5QT", 549, 449)]),
    "frozenveg": ("FROZEN", "each", [
        ("FIELD FROZEN PEAS 12OZ", 199, None), ("FIELD FROZEN CORN 12OZ", 199, None),
        ("FIELD FROZEN BROCCOLI 12OZ", 229, None)]),
    "apples": ("PRODUCE", "lb", [
        ("APPLES GALA", 199, None), ("APPLES HONEYCRISP", 299, None), ("APPLES GRANNY SMITH", 219, None)]),
    "grapes": ("PRODUCE", "lb", [("GRAPES RED SEEDLESS", 349, None), ("GRAPES GREEN SEEDLESS", 349, None)]),
    "berries": ("PRODUCE", "each", [
        ("STRAWBERRIES 1LB", 449, 399), ("BLUEBERRIES PINT", 399, None), ("RASPBERRIES 6OZ", 449, None)]),
    "avocado": ("PRODUCE", "each", [("AVOCADO HASS", 149, None), ("AVOCADO HASS BAG 4CT", 549, None)]),
    "salmon": ("MEAT", "lb", [("ATLANTIC SALMON FILLET", 1199, None), ("SOCKEYE SALMON FILLET", 1499, None)]),
    "chicken": ("MEAT", "lb", [("CHICKEN BREAST BONELESS", 499, None), ("CHICKEN THIGHS BONELESS", 399, None)]),
    "bacon": ("MEAT", "each", [("HICKORY BACON 12OZ", 699, 599), ("THICK CUT BACON 16OZ", 899, 799)]),
    "towels": ("HOUSEHOLD", "each", [
        ("SPARKLE PAPER TOWELS 6 ROLL", 1199, None), ("SPARKLE PAPER TOWELS 12 ROLL", 2099, None)]),
    "pods": ("HOUSEHOLD", "each", [
        ("FRESH LAUNDRY PODS 42CT", 1399, None), ("FRESH LAUNDRY PODS 81CT", 2299, None)]),
    "dish": ("HOUSEHOLD", "each", [("GLEAM DISH SOAP 19OZ", 349, None), ("GLEAM DISH SOAP LEMON 19OZ", 349, None)]),
    "toothpaste": ("HBC", "each", [("BRIGHT TOOTHPASTE MINT", 399, None), ("BRIGHT TOOTHPASTE WHITENING", 449, None)]),
    "shampoo": ("HBC", "each", [
        ("SILK SHAMPOO 12OZ", 699, None), ("SILK CONDITIONER 12OZ", 699, None), ("SILK 2IN1 12OZ", 749, None)]),
    "delisalad": ("DELI", "lb", [("DELI POTATO SALAD", 499, None), ("DELI CHICKEN SALAD", 899, None)]),
    "sandwich": ("DELI", "each", [
        ("DELI TURKEY SANDWICH", 699, None), ("DELI ITALIAN SANDWICH", 749, None), ("DELI VEGGIE WRAP", 649, None)]),
}

FILLER = {
    "PRODUCE": ("lb", ["BANANAS", "ORGANIC BANANAS", "ROMA TOMATOES", "YELLOW ONIONS", "RUSSET POTATOES",
                       "CARROTS", "RED PEPPERS", "CUCUMBERS", "ZUCCHINI", "PEARS BARTLETT", "NAVEL ORANGES",
                       "LIMES", "GINGER ROOT", "SWEET POTATOES"], (59, 399)),
    "PRODUCE_EACH": ("each", ["BAGGED SALAD SPRING MIX", "CELERY HEARTS", "ROMAINE HEARTS 3CT", "GARLIC 3CT",
                              "CILANTRO BUNCH", "BABY CARROTS 1LB", "MUSHROOMS 8OZ", "LEMONS EACH"], (79, 449)),
    "MEAT": ("lb", ["GROUND BEEF 80/20", "PORK CHOPS", "BEEF STEW MEAT", "TURKEY BREAST", "RIBEYE STEAK"],
             (349, 1599)),
    "DAIRY": ("each", ["MEADOW MILK GALLON", "MEADOW 2% MILK HALF GAL", "LARGE EGGS DOZEN", "SALTED BUTTER 1LB",
                       "SOUR CREAM 16OZ", "CREAM CHEESE 8OZ", "COTTAGE CHEESE 16OZ", "OAT MILK 64OZ",
                       "STRING CHEESE 12CT", "HEAVY CREAM PINT"], (199, 649)),
    "BAKERY": ("each", ["SOURDOUGH LOAF", "WHOLE WHEAT BREAD", "PLAIN BAGELS 6CT", "BLUEBERRY MUFFINS 4CT",
                        "HAMBURGER BUNS 8CT", "FLOUR TORTILLAS 10CT", "CROISSANTS 4CT"], (249, 599)),
    "FROZEN": ("each", ["FROZEN WAFFLES 10CT", "FISH STICKS 24OZ", "FROZEN BURRITO", "FROZEN BERRY MEDLEY 16OZ",
                        "VEGGIE BURGERS 4CT", "FROZEN LASAGNA 38OZ"], (129, 1099)),
    "PANTRY": ("each", ["LONG GRAIN RICE 2LB", "BLACK BEANS 15OZ", "CHICKPEAS 15OZ", "PEANUT BUTTER 16OZ",
                        "STRAWBERRY JAM 18OZ", "CHICKEN BROTH 32OZ", "OLIVE OIL 17OZ", "ROLLED OATS 42OZ",
                        "ALL PURPOSE FLOUR 5LB", "GRANULATED SUGAR 4LB", "DICED TOMATOES 28OZ",
                        "TUNA CHUNK LIGHT 5OZ", "MAC AND CHEESE 7OZ", "HONEY 12OZ"], (109, 1099)),
    "BEVERAGE": ("each", ["SPRING WATER 24PK", "ORANGE JUICE 52OZ", "APPLE JUICE 64OZ", "COLD BREW COFFEE 48OZ",
                          "SPORTS DRINK 28OZ", "GINGER ALE 2L"], (149, 699)),
    "SNACKS": ("each", ["PRETZEL TWISTS 16OZ", "MICROWAVE POPCORN 3CT", "ANIMAL CRACKERS 13OZ", "GUMMY BEARS 5OZ",
                        "ROASTED PEANUTS 16OZ", "GRANOLA BARS 6CT"], (199, 549)),
    "HOUSEHOLD": ("each", ["TRASH BAGS 13GAL 40CT", "BATH TISSUE 12 ROLL", "ALUMINUM FOIL 75SQFT", "SPONGES 6CT",
                           "ALL PURPOSE CLEANER 32OZ", "ZIPPER BAGS GALLON 30CT"], (299, 1499)),
    "HBC": ("each", ["BAR SOAP 4CT", "DEODORANT 2.6OZ", "IBUPROFEN 100CT", "COTTON SWABS 300CT",
                     "BODY WASH 18OZ", "MOUTHWASH 1L"], (299, 999)),
    "DELI": ("each", ["ROTISSERIE CHICKEN", "DELI SUSHI COMBO", "DELI MAC SALAD CUP", "DELI SOUP 16OZ"],
             (399, 999)),
}


def upc(rng):
    digits = [rng.randrange(10) for _ in range(11)]
    odd = sum(digits[0::2])
    even = sum(digits[1::2])
    check = (10 - (odd * 3 + even) % 10) % 10
    return "".join(map(str, digits)) + str(check)


def build_catalog(rng):
    catalog, families, plus = [], {}, iter(range(3000, 4999, 7))
    seen = set()

    def new_code(sold_by):
        while True:
            code = str(next(plus)) if sold_by == "lb" else upc(rng)
            if code not in seen:
                seen.add(code)
                return code

    for key, (dept, sold_by, items) in FAMILIES.items():
        families[key] = []
        for desc, price, member in items:
            sku = new_code(sold_by)
            catalog.append({"sku": sku, "description": desc, "dept": dept, "tax_class": DEPT_CLASS[dept],
                            "price": price, "member_price": member, "sold_by": sold_by})
            families[key].append(sku)
    for group, (sold_by, names, (lo, hi)) in FILLER.items():
        dept = group.split("_")[0]
        for name in names:
            price = rng.randrange(lo, hi + 1, 10) + 9
            member = price - rng.choice([20, 30, 50, 100]) if rng.random() < 0.15 and sold_by == "each" else None
            if member is not None and member <= 0:
                member = None
            catalog.append({"sku": new_code(sold_by), "description": name, "dept": dept,
                            "tax_class": DEPT_CLASS[dept], "price": price, "member_price": member,
                            "sold_by": sold_by})
    rng.shuffle(catalog)
    return catalog, families


ALL_STORES = [101, 102, 103]


VENDOR_FUNDED = {
    "S1103", "S1105", "S1106", "S1110", "S1113", "S1116", "S1120", "S1122", "S1126", "S1127",
    "D2402", "D2404", "D2405", "D2407", "D2408", "D2411", "D2415", "D2416", "D2418", "D2421", "D2422",
    "D2432", "D2434", "D2436", "D2440", "D2441", "D2442", "D2443",
}


def deal(id_, type_, desc, skus, priority, start, end, stores=ALL_STORES, member_only=False, **kw):
    d = {"id": id_, "type": type_, "description": desc,
         "funding": "VENDOR" if id_ in VENDOR_FUNDED else "STORE"}
    if skus is not None:
        d["skus"] = skus
    d.update(kw)
    d.update({"priority": priority, "member_only": member_only, "stores": stores, "start": start, "end": end})
    return d


def coupon(code, desc, skus, qty, save, start, end, stores=ALL_STORES):
    return {"code": code, "description": desc, "skus": skus, "qty": qty, "save": save,
            "stores": stores, "start": start, "end": end}


def september(f):
    a, b, e = "2026-09-01", "2026-09-14", "2026-09-30"
    m = "2026-09-15"
    deals = [
        deal("S1101", "MULTI", "FIZZ 2L 2/$4.00", f["fizz2l"], 61, a, e, qty=2, price=400, limit=4),
        deal("S1102", "PCT", "FIZZ CHERRY 2L 20% OFF MEMBER", f["fizz2l"][2:3], 34, a, b, member_only=True,
             pct=20),
        deal("S1103", "MULTI", "FIZZ 12PK 3/$16.00", f["fizz12"], 55, m, e, qty=3, price=1600, limit=2),
        deal("S1104", "PCT", "PEAK SPARKLING 25% OFF", f["peak"], 47, a, e, pct=25),
        deal("S1105", "BOGO", "VOLT ENERGY BUY 1 GET 1 FREE", f["volt"], 70, a, b, buy=1, get=1, pct=100,
             limit=3),
        deal("S1106", "BOGO", "CRUNCHOS BUY 1 GET 1 FREE", f["crunchos"][:2] + f["crunchos"][3:], 66, a, e,
             buy=1, get=1, pct=100, limit=None),
        deal("S1107", "MULTI", "GOLDEN KETTLE 2/$7.00", f["kettle"], 58, a, e, qty=2, price=700, limit=None),
        deal("S1108", "PCT", "GOLDEN KETTLE BBQ 30% OFF MEMBER", f["kettle"][1:2], 49, m, e, member_only=True,
             pct=30),
        deal("S1109", "MULTI", "CHOCO BAR 3/$4.00", f["choco"], 40, a, e, qty=3, price=400, limit=3),
        deal("S1110", "MULTI", "ALPINE YOGURT 5/$5.00", f["yogurt"], 52, a, e, qty=5, price=500, limit=2),
        deal("S1111", "PCT", "MEADOW SHREDDED CHEESE 20% OFF", f["cheese"], 36, a, b, pct=20),
        deal("S1112", "MULTI", "BELLA PASTA 4/$6.00", f["pasta"], 45, a, e, qty=4, price=600, limit=2),
        deal("S1113", "BOGO", "BELLA SAUCE BUY 1 GET 1 FREE MEMBER", f["sauce"], 68, m, e, member_only=True,
             buy=1, get=1, pct=100, limit=2),
        deal("S1114", "PCT", "CEREAL 15% OFF", f["cereal"], 31, a, e, pct=15),
        deal("S1115", "MULTI", "POLAR PIZZA 2/$11.00", f["pizza"], 57, a, e, qty=2, price=1100, limit=3),
        deal("S1116", "BOGO", "FROSTY ICE CREAM BUY 1 GET 1 FREE MEMBER", f["icecream"], 72, a, b,
             member_only=True, buy=1, get=1, pct=100, limit=2),
        deal("S1117", "PCT", "HONEYCRISP APPLES 25% OFF", f["apples"][1:2], 44, a, e, pct=25),
        deal("S1118", "PCT", "SALMON FILLETS 15% OFF", f["salmon"], 39, m, e, pct=15),
        deal("S1119", "MULTI", "AVOCADO HASS 4/$5.00", f["avocado"][:1], 50, a, e, qty=4, price=500, limit=2),
        deal("S1120", "PCT", "FRESH LAUNDRY PODS 30% OFF", f["pods"], 42, a, e, pct=30),
        deal("S1121", "MULTI", "SPARKLE TOWELS 2/$20.00", f["towels"][:1], 48, a, e, qty=2, price=2000, limit=2),
        deal("S1122", "PCT", "SILK HAIR CARE 25% OFF MEMBER", f["shampoo"], 38, a, e, member_only=True, pct=25),
        deal("S1123", "MULTI", "DELI SANDWICH 2/$12.00", f["sandwich"], 54, a, e, qty=2, price=1200, limit=None),
        deal("S1124", "PCT", "BERRIES 20% OFF", f["berries"], 33, m, e, pct=20),
        deal("S1125", "MULTI", "BACON 2/$11.00", f["bacon"][:1], 46, a, e, qty=2, price=1100, limit=2),
        deal("S1126", "MULTI", "FIZZ 2L 2/$4.00 VENDOR TPR", f["fizz2l"], 66, m, e, qty=2, price=400, limit=1),
        deal("S1127", "PCT", "PEAK SPARKLING 25% OFF VENDOR TPR", f["peak"][:2], 51, a, b, pct=25),
        deal("S1190", "SPEND", "SPEND $25 ON PRODUCE SAVE $4", None, 80, a, e, dept="PRODUCE",
             threshold=2500, save=400),
        deal("S1191", "SPEND", "SPEND $20 ON SNACKS SAVE $3", None, 76, a, e, dept="SNACKS",
             threshold=2000, save=300),
        deal("S1192", "SPEND", "SPEND $30 ON HOUSEHOLD SAVE $5 MEMBER", None, 74, m, e, dept="HOUSEHOLD",
             threshold=3000, save=500, member_only=True),
    ]
    coupons = [
        coupon("MC-20417", "$1.00 OFF 2 GOLDEN KETTLE", f["kettle"], 2, 100, a, e),
        coupon("MC-20418", "$2.00 OFF FRESH LAUNDRY PODS", f["pods"], 1, 200, a, e),
        coupon("MC-20419", "$0.75 OFF BELLA SAUCE", f["sauce"], 1, 75, a, e),
        coupon("MC-20420", "$1.50 OFF 2 POLAR PIZZA", f["pizza"], 2, 150, a, b),
        coupon("MC-20421", "$1.00 OFF CRUNCHOS", f["crunchos"], 1, 100, m, e),
        coupon("MC-20422", "$0.50 OFF ALPINE YOGURT 4", f["yogurt"], 4, 50, a, e),
    ]
    return {"period": "2026-09", "deals": deals, "coupons": coupons}


def october(f):
    a, b, e = "2026-10-01", "2026-10-14", "2026-10-31"
    m = "2026-10-15"
    deals = [
        deal("D2401", "MULTI", "FIZZ 2L 3/$6.00", f["fizz2l"], 63, a, e, qty=3, price=600, limit=3),
        deal("D2402", "BOGO", "FIZZ 2L BUY 2 GET 1 FREE MEMBER", f["fizz2l"], 71, m, e, member_only=True,
             buy=2, get=1, pct=100, limit=2),
        deal("D2403", "PCT", "FIZZ DIET 2L 10% OFF", f["fizz2l"][1:2], 29, a, e, pct=10),
        deal("D2404", "MULTI", "FIZZ 12PK 2/$11.00", f["fizz12"], 56, a, b, qty=2, price=1100, limit=2),
        deal("D2405", "BOGO", "PEAK SPARKLING BUY 2 GET 1 50% OFF", f["peak"], 64, a, e, buy=2, get=1, pct=50,
             limit=None),
        deal("D2406", "PCT", "PEAK SPARKLING LIME 20% OFF MEMBER", f["peak"][:1], 41, a, e, member_only=True,
             pct=20),
        deal("D2407", "MULTI", "VOLT ENERGY 4/$10.00", f["volt"], 53, a, e, qty=4, price=1000, limit=2),
        deal("D2408", "BOGO", "CRUNCHOS BUY 2 GET 1 50% OFF", f["crunchos"], 67, a, e, buy=2, get=1, pct=50,
             limit=None),
        deal("D2409", "MULTI", "CRUNCHOS 13OZ 2/$8.00", f["crunchos"][:2], 59, a, e, qty=2, price=800,
             limit=None),
        deal("D2410", "MULTI", "GOLDEN KETTLE 3/$10.00", f["kettle"], 58, a, e, qty=3, price=1000, limit=2),
        deal("D2411", "BOGO", "GOLDEN KETTLE BUY 1 GET 1 50% OFF MEMBER", f["kettle"], 62, m, e,
             member_only=True, buy=1, get=1, pct=50, limit=2),
        deal("D2412", "MULTI", "CHOCO BAR 4/$5.00", f["choco"], 37, a, e, qty=4, price=500, limit=2),
        deal("D2413", "PCT", "EL CAMINO SALSA 25% OFF", f["salsa"], 35, a, e, pct=25),
        deal("D2414", "MULTI", "ALPINE YOGURT 6/$6.00", f["yogurt"], 52, a, e, qty=6, price=600, limit=2),
        deal("D2415", "PCT", "ALPINE YOGURT FRUIT 30% OFF MEMBER", f["yogurt"][1:3] + f["yogurt"][4:], 43,
             a, e, member_only=True, pct=30),
        deal("D2416", "BOGO", "MEADOW CHEESE BUY 1 GET 1 FREE", f["cheese"], 69, a, b, buy=1, get=1, pct=100,
             limit=1),
        deal("D2417", "MULTI", "BELLA PASTA 5/$7.00", f["pasta"], 44, a, e, qty=5, price=700, limit=2),
        deal("D2418", "MULTI", "BELLA SAUCE 2/$6.00", f["sauce"], 51, a, e, qty=2, price=600, limit=2),
        deal("D2419", "PCT", "CEREAL 20% OFF MEMBER", f["cereal"], 32, a, e, member_only=True, pct=20),
        deal("D2420", "MULTI", "POLAR PIZZA 3/$17.00", f["pizza"], 57, a, e, qty=3, price=1700, limit=2),
        deal("D2421", "BOGO", "POLAR PIZZA BUY 1 GET 1 50% OFF", f["pizza"], 65, m, e, buy=1, get=1, pct=50,
             limit=2),
        deal("D2422", "BOGO", "FROSTY ICE CREAM BUY 1 GET 1 FREE", f["icecream"], 73, a, b, buy=1, get=1,
             pct=100, limit=2),
        deal("D2423", "MULTI", "FROSTY ICE CREAM 2/$8.00 MEMBER", f["icecream"], 60, a, e, member_only=True,
             qty=2, price=800, limit=3),
        deal("D2424", "MULTI", "FIELD FROZEN VEG 5/$8.00", f["frozenveg"], 39, a, e, qty=5, price=800, limit=2),
        deal("D2425", "PCT", "APPLES 20% OFF", f["apples"], 46, a, e, pct=20),
        deal("D2426", "PCT", "GRAPES 30% OFF MEMBER", f["grapes"], 48, m, e, member_only=True, pct=30),
        deal("D2427", "MULTI", "BERRIES 2/$7.00", f["berries"], 50, a, e, qty=2, price=700, limit=3),
        deal("D2428", "PCT", "STRAWBERRIES 25% OFF MEMBER", f["berries"][:1], 42, a, b, member_only=True,
             pct=25),
        deal("D2429", "MULTI", "AVOCADO HASS 5/$6.00", f["avocado"][:1], 47, a, e, qty=5, price=600, limit=2),
        deal("D2430", "PCT", "SALMON FILLETS 20% OFF", f["salmon"], 40, a, e, pct=20),
        deal("D2431", "PCT", "CHICKEN BREAST 25% OFF MEMBER", f["chicken"][:1], 36, a, e, member_only=True,
             pct=25),
        deal("D2432", "BOGO", "BACON BUY 1 GET 1 50% OFF", f["bacon"], 55, a, e, buy=1, get=1, pct=50, limit=2),
        deal("D2433", "MULTI", "SPARKLE TOWELS 2/$19.00", f["towels"][:1], 49, a, e, qty=2, price=1900, limit=2),
        deal("D2434", "PCT", "FRESH LAUNDRY PODS 25% OFF", f["pods"], 45, a, e, pct=25),
        deal("D2435", "MULTI", "GLEAM DISH SOAP 2/$6.00", f["dish"], 38, a, e, qty=2, price=600, limit=2),
        deal("D2436", "BOGO", "SILK HAIR CARE BUY 2 GET 1 FREE", f["shampoo"], 54, a, e, buy=2, get=1, pct=100,
             limit=1),
        deal("D2437", "MULTI", "BRIGHT TOOTHPASTE 2/$7.00", f["toothpaste"], 34, a, e, qty=2, price=700,
             limit=None),
        deal("D2438", "MULTI", "DELI SANDWICH 2/$12.50", f["sandwich"], 51, a, e, qty=2, price=1250, limit=None),
        deal("D2439", "PCT", "DELI SALADS 15% OFF", f["delisalad"], 30, a, e, pct=15),
        deal("D2440", "MULTI", "GOLDEN KETTLE 3/$10.00 VENDOR TPR", f["kettle"], 61, a, e, qty=3, price=1000,
             limit=1),
        deal("D2441", "MULTI", "ALPINE YOGURT 6/$6.00 VENDOR TPR", f["yogurt"], 54, m, e, qty=6, price=600, limit=1),
        deal("D2442", "MULTI", "FIZZ 2L 3/$6.00 VENDOR TPR", f["fizz2l"], 68, a, b, qty=3, price=600, limit=1),
        deal("D2443", "PCT", "SALMON FILLETS 20% OFF VENDOR TPR", f["salmon"][:1], 43, m, e, pct=20),
        deal("D2490", "SPEND", "SPEND $25 ON PRODUCE SAVE $5", None, 82, a, e, dept="PRODUCE",
             threshold=2500, save=500),
        deal("D2491", "SPEND", "SPEND $20 ON SNACKS SAVE $4", None, 78, a, e, dept="SNACKS",
             threshold=2000, save=400),
        deal("D2492", "SPEND", "SPEND $30 ON FROZEN SAVE $6 MEMBER", None, 77, a, e, dept="FROZEN",
             threshold=3000, save=600, member_only=True),
        deal("D2493", "SPEND", "SPEND $25 ON BEVERAGE SAVE $5", None, 75, m, e, dept="BEVERAGE",
             threshold=2500, save=500),
        deal("D2494", "SPEND", "SPEND $40 ON HOUSEHOLD SAVE $8", None, 74, a, e, dept="HOUSEHOLD",
             threshold=4000, save=800),
    ]
    coupons = [
        coupon("MC-31201", "$1.00 OFF 2 CRUNCHOS", f["crunchos"], 2, 100, a, e),
        coupon("MC-31202", "$1.50 OFF FIZZ 12PK", f["fizz12"], 1, 150, a, e),
        coupon("MC-31203", "$0.50 OFF ALPINE YOGURT 3", f["yogurt"], 3, 50, a, e),
        coupon("MC-31204", "$2.00 OFF FRESH LAUNDRY PODS", f["pods"], 1, 200, a, e),
        coupon("MC-31205", "$1.00 OFF POLAR PIZZA", f["pizza"], 1, 100, m, e),
        coupon("MC-31206", "$0.75 OFF 2 BELLA PASTA", f["pasta"], 2, 75, a, e),
        coupon("MC-31207", "$1.00 OFF FROSTY ICE CREAM", f["icecream"], 1, 100, a, b),
        coupon("MC-31208", "$1.00 OFF SILK HAIR CARE", f["shampoo"], 1, 100, a, e),
    ]
    return {"period": "2026-10", "deals": deals, "coupons": coupons}


def leaf_count(basket, catalog, promos):
    """Upper bound on the deal-assignment search for the largest department."""
    deals = [d for d in promos["deals"] if engine.is_active(d, basket) and d["type"] != "SPEND"]
    per_sku = {}
    for d in deals:
        for s in d["skus"]:
            per_sku[s] = per_sku.get(s, 0) + 1
    qty, weighed = {}, {}
    worst = {}
    for line in basket["lines"]:
        item = catalog[line["sku"]]
        if line["sku"] not in per_sku:
            continue
        if item["sold_by"] == "lb":
            worst[item["dept"]] = worst.get(item["dept"], 1) * 2
        else:
            qty[line["sku"]] = qty.get(line["sku"], 0) + line["qty"]
    for sku, n in qty.items():
        k = per_sku[sku]
        dept = catalog[sku]["dept"]
        worst[dept] = worst.get(dept, 1) * math.comb(n + k, k)
    return max(worst.values(), default=1)


def make_baskets(rng, period, days, n, catalog_rows, families, promos, focus):
    catalog = {r["sku"]: r for r in catalog_rows}
    by_dept = {}
    for r in catalog_rows:
        by_dept.setdefault(r["dept"], []).append(r["sku"])
    family_keys = list(families)
    promo_skus = {s for d in promos["deals"] if d["type"] != "SPEND" for s in d["skus"]}
    spend_depts = [d["dept"] for d in promos["deals"] if d["type"] == "SPEND"]
    baskets, seq = [], {}
    while len(baskets) < n:
        store = rng.choices(ALL_STORES, weights=[5, 3, 2])[0]
        day = rng.choice(days)
        member = f"L{rng.randrange(10000, 99999)}" if rng.random() < 0.62 else None
        lines = []
        size = rng.choice([3, 5, 8, 12, 16, 20, 26])
        size = max(2, int(rng.gauss(size, 2)))
        for _ in range(size):
            r = rng.random()
            if r < focus:
                fam = families[rng.choice(family_keys)]
                skus = [s for s in fam if s in promo_skus] or fam
                picks = rng.choices([1, 2, 3], weights=[5, 3, 2])[0]
                for sku in rng.sample(skus, min(picks, len(skus))):
                    lines.append(make_line(rng, catalog[sku]))
            elif r < focus + 0.12 and spend_depts:
                dept = rng.choice(spend_depts)
                lines.append(make_line(rng, catalog[rng.choice(by_dept[dept])]))
            else:
                dept = rng.choice(list(by_dept))
                lines.append(make_line(rng, catalog[rng.choice(by_dept[dept])]))
        if rng.random() < 0.08 and lines:
            lines.append(dict(rng.choice(lines)))          # re-scan of an earlier item
        coupons = []
        if rng.random() < 0.18:
            for c in rng.sample(promos["coupons"], rng.choice([1, 1, 2, 3])):
                coupons.extend([c["code"]] * rng.choice([1, 1, 1, 2]))
        key = (store, day)
        seq[key] = seq.get(key, 0) + 1
        basket = {"basket_id": f"{store}-{day.replace('-', '')}-{seq[key]:04d}", "store": store, "date": day,
                  "member": member, "lines": lines, "coupons": coupons}
        if leaf_count(basket, catalog, promos) > SEARCH_CAP:
            continue
        baskets.append(basket)
    baskets.sort(key=lambda b: (b["date"], b["store"], b["basket_id"]))
    return baskets


def make_line(rng, item):
    if item["sold_by"] == "lb":
        return {"sku": item["sku"], "weight": round(rng.uniform(0.25, 3.6), 2)}
    return {"sku": item["sku"], "qty": rng.choices([1, 2, 3, 4, 5, 6], weights=[40, 25, 14, 10, 6, 5])[0]}


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def price_all(baskets, catalog_path, promos):
    catalog = engine.load_catalog(catalog_path)
    return [engine.price_basket(b, catalog, STORES[str(b["store"])]["rates"], promos) for b in baskets]


footer = engine.journal_footer


def main():
    rng = random.Random(SEED)
    catalog_rows, families = build_catalog(rng)
    sep, octo = september(families), october(families)

    for sub in ("legacy", "history", "pending"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    catalog_path = DATA / "legacy" / "catalog.csv"
    with open(catalog_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sku", "description", "dept", "tax_class", "price", "member_price",
                                           "sold_by"])
        w.writeheader()
        for r in catalog_rows:
            w.writerow({**r, "member_price": r["member_price"] if r["member_price"] is not None else ""})
    with open(DATA / "legacy" / "stores.json", "w") as fh:
        json.dump({"stores": STORES}, fh, indent=2)
        fh.write("\n")
    for path, promos in ((DATA / "history" / "promotions.json", sep), (DATA / "pending" / "promotions.json", octo)):
        with open(path, "w") as fh:
            json.dump(promos, fh, indent=2)
            fh.write("\n")

    sep_days = [(dt.date(2026, 9, 1) + dt.timedelta(d)).isoformat() for d in range(30)]
    oct_days = [(dt.date(2026, 10, 1) + dt.timedelta(d)).isoformat() for d in range(31)]
    history = make_baskets(rng, "2026-09", sep_days, HISTORY_BASKETS, catalog_rows, families, sep, focus=0.42)
    pending = make_baskets(rng, "2026-10", oct_days, PENDING_BASKETS, catalog_rows, families, octo, focus=0.55)

    write_jsonl(DATA / "history" / "baskets.jsonl", history)
    write_jsonl(DATA / "history" / "journal.jsonl", [footer(r) for r in price_all(history, catalog_path, sep)])
    write_jsonl(DATA / "pending" / "baskets.jsonl", pending)

    shutil.copyfile(TASK / "solution" / "engine.py", TESTS / "legacy_engine.py")
    for rel in ("legacy/catalog.csv", "legacy/stores.json", "pending/promotions.json", "pending/baskets.jsonl"):
        dest = TESTS / "inputs" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / rel, dest)
    print(f"history={len(history)} pending={len(pending)} catalog={len(catalog_rows)}")


if __name__ == "__main__":
    main()
