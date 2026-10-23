RTX-4 Store Systems
Pricing and Promotions Reference, rev. 3.2 (excerpt: chapters 4, 6, 7 and appendix C)

---

4. ITEM PRICING

4.1 Shelf price. Every item in the item file has a shelf price in cents
(`price`). Items sold by the each (`sold_by` = `each`) are rung up as
quantity x unit price.

4.2 Member price. When a loyalty number is entered on the transaction, items
that carry a member price (`member_price`) are rung up at the member price
instead of the shelf price. All further pricing, including deals, starts from
the price the item was rung up at.

4.3 Weighed items. Items with `sold_by` = `lb` are rung up from the scale.
`price` is the price per pound and the scale reports the weight in pounds to
two decimals. The line amount is weight x price per pound, in cents.

4.4 Subtotal. The transaction subtotal is the sum of the line amounts before
any deal or coupon.

---

6. DEALS

6.1 General. Deals are loaded from the promotion calendar. Every deal has:

  id           Deal number. Printed on the receipt and used for vendor billing.
  type         MULTI, BOGO, PCT or SPEND (see 6.2 - 6.5).
  description  Receipt text.
  funding      STORE or VENDOR. Vendor-funded deals are billed back to the
               vendor at the end of the period, deal by deal, from the
               receipt export.
  priority     Whole number. Used only to break ties (see 6.7). No two deals
               in a calendar share a priority.
  member_only  If true the deal applies only when a loyalty number is entered.
  stores       Store numbers where the deal runs.
  start, end   First and last business date the deal runs (both inclusive).

MULTI, BOGO and PCT deals are item deals and list the items they cover in
`skus`. A deal applies to any mix of the listed items.

6.2 MULTI ("n for $X"). Fields: `qty` (n), `price` (X, cents for the group),
`limit` (most groups per transaction, null for no limit). Each group of `qty`
units is sold for `price`. The saving for a group is the amount the units were
rung up at minus `price`.

6.3 BOGO ("buy n get m"). Fields: `buy`, `get`, `pct`, `limit` (most groups
per transaction, null for no limit). Each group is `buy` + `get` units; the
`get` lowest-priced units in the group are taken `pct` percent off (100 =
free).

6.4 PCT (percent off). Field: `pct`. Each unit on the deal is taken `pct`
percent off. Weighed items can take PCT deals; the discount is taken off the
weighed line amount.

6.5 SPEND ("spend $T save $V"). Fields: `dept`, `threshold` (T, cents),
`save` (V, cents). Applies once per transaction when the department total is
at least `threshold`. The department total is the amount of every item in the
department, on deal or not, after item deal savings and before coupons.

6.6 Best deal. The terminal applies deals so that the customer pays the least.
It considers every valid way of applying the active deals to the transaction
and uses the one with the greatest total saving from item deals and spend deals
together. A way of applying deals is valid when:

  a) every unit is in at most one item deal (one MULTI group, one BOGO group or
     one PCT deal); weighed items can only be in PCT deals;
  b) MULTI and BOGO deals are applied in complete groups and no more times
     than their limit;
  c) within each MULTI or BOGO deal, the units given to the deal are grouped
     the way that saves the most for that deal. For BOGO this means the
     highest-priced units are grouped together and the lowest-priced unit(s)
     of each group are discounted.

A deal does not have to be used on every unit it covers. The terminal will
leave units off a deal, or apply a deal fewer times, when that gives the
customer a bigger total saving, for example to keep a department total at or
above a spend threshold.

6.7 Ties. If two or more ways of applying deals give the same total saving,
the terminal compares them deal by deal from the highest priority to the
lowest and uses the one that gives the larger saving to the first deal where
they differ.

---

7. COUPONS

7.1 Manufacturer coupons are scanned after the items. The promotion calendar
lists the accepted codes with `skus`, `qty`, `save`, `stores`, `start` and
`end`. A scanned coupon takes `save` cents off when the transaction has at
least `qty` units of the listed items. A code scanned more than once is
accepted once for every `qty` units. Units count toward a coupon whether or
not they are on a deal. Coupons that do not qualify, have expired or do not
run at the store are rejected and save nothing.

7.2 Coupons are not deals. They are applied after deals and do not affect
which deals are applied.

---

APPENDIX C. EXPORT FORMATS

C.1 Receipt export. One receipt per transaction, written as one JSON object
per line. All amounts are whole cents (JSON integers).

  basket_id   Transaction id, as in the basket file.
  subtotal    Section 4.4.
  deals       Object: deal id -> total saving from that deal.
  coupons     Object: coupon code -> total saving from that code.
  tax         Object: tax class -> tax charged for that class. Rates are in
              the store table (`stores.json`), in basis points.
  total       subtotal - all deal savings - all coupon savings + all tax.

Deals, coupons and tax classes with a zero amount are not exported.

C.2 Electronic journal. The terminal's electronic journal keeps only the
footer of each receipt, one JSON object per line:

  basket_id       As in C.1.
  subtotal        As in C.1.
  deal_savings    Sum of all deal savings on the receipt.
  coupon_savings  Sum of all coupon savings on the receipt.
  tax             As in C.1.
  total           As in C.1.
