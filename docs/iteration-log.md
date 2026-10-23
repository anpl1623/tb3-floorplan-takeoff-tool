# Iteration log

## Iteration 1: POS promotion parity (abandoned)

**Idea.** A grocer is cutting over from a decommissioned legacy POS. The agent had to reproduce
the legacy terminal's October receipts to the cent from a vendor manual, a catalog, store tax
rates and a month of September parallel-run data. The manual specified the deal types and a
"best deal" rule: overlapping item deals plus department spend thresholds, optimized jointly,
with priority tie-breaks. Rounding and the sales-tax base were left out of the manual and had to
be recovered from the data; vendor-funded deals and manufacturer coupons are taxed as third-party
consideration. Code and results are in [`archive/v1-pos-promo-parity`](../archive/v1-pos-promo-parity).

**What was built.** A reference engine, a seeded data generator, a unit-level brute-force
cross-check of the optimizer (about 40,000 real department problems plus random stress cases),
single-rule ablations showing every rule was visible in the September data and caught by the
verifier, and a verifier probe.

**Result.** GPT-6.1-sol (codex, xhigh) passed **3/3 in under 6 minutes**. In each trajectory it
read the manual, wrote one exact optimizer, matched all 9,000 September records on its first run,
brute-force-checked its October allocations and stopped. Before that pilot I had made September
less informative (receipt footers only, no per-deal lines) and added the funding/tax rule. Neither
change mattered.

**Lesson.** "Implement a precise spec, then infer a few missing rules from plentiful clean data" is
exactly what frontier coding agents are best at. Every hypothesis can be checked in seconds
against local data, so obscuring the rules only adds work. It doesn't add a wall. Obvious next
moves (path-dependent deal evaluation, per-store firmware, noisy journals) would have kept that
same property, so I dropped the idea rather than escalate it.

The /cheat trial for codex never started: OpenAI's API rejects the TB3 adversarial prompt at the
first turn ("flagged for possible cybersecurity risk", `AgentSafetyRefusalError`). This happened
twice and is deterministic.

## Iteration 2: store remodel quantity takeoff from a clean scan (superseded)

Iterations 2-4 are archived in [`archive/v2-store-remodel-takeoff`](../archive/v2-store-remodel-takeoff),
with every trial outcome in `results/trials.json`.

**Why this domain.** Difficulty should come from a wall the agent cannot test its way past. A
quantity takeoff from a scanned drawing has that property. Nothing in the environment can
confirm a measured number. The input is pixels, not text, so even the hypotheses about what the
drawing shows have to be checked by eye. The conventions (net areas, centerline partitions,
openings not deducted, verify the scale) are ones estimators learn on the job. It is also squarely
the author's domain: retail construction and store rollouts.

**Design decisions**
- *Synthetic but realistic sheet.* `dev/plan.py` holds a vector model of a 17,600 SF grocery;
  `dev/render.py` plots it the way a CAD pen table would. That means line-weight hierarchy, hatch
  in a lighter pen, poché by partition type, partition/finish/door tags, a dash-dot column grid,
  dimension strings, a legend, general notes and a title block.
- *Scan, not vector PDF.* A vector PDF hands the geometry to anyone who calls `get_drawings()`. The
  sheet is rasterized, reduced to 97.3% (a "not to scale" print), skewed 0.38°, blurred, given
  noise and uneven lamp shading, JPEG-compressed, and wrapped in an image-only PDF.
- *Things that make digitizing rectangles insufficient.* A polished-concrete racetrack aisle with
  radiused corners wraps center store. Finish changes happen inside rooms (receiving epoxy inside
  the stockroom). Fixtures and tags sit on top of hatch, and floor under them still counts. Gondolas
  mix 3'-0" and 4'-0" sections. There are four door symbol types.
- *The scale trap is fair.* The title block carries the standard "if this bar does not measure 1""
  bar and the sheet is fully dimensioned. Calibrating against a known dimension is the first thing
  any estimator does in Bluebeam.

**Making the oracle work, and what that changed in the drawing.** The reference solution is a
real CV pipeline on the raster, not a lookup. Building it surfaced drawing problems that a human
reviewer would also call drafting errors, and they were fixed in the drawing:
- a partition tag (white hexagon) sat on the racetrack's transition line and broke it, so the tag
  was moved;
- finish tags overlapped door swings, so they were moved;
- P1 partitions got a light poché so a blank door opening can't be mistaken for a short P1 wall;
- sliding-door panels are drawn as heavy lines instead of rectangles (the rectangles' end strokes
  filled a whole thin wall and read as a jamb);
- cooler-door and endcap/table diagonals were drawn at fixture weight, so thin triangles survive
  scanning.

The oracle reproduces all 23 quantities: areas within 0.71%, wall lengths within 0.08%, every
count exact.

**Tolerance calibration** (`dev/tolerance.py`). A simulated estimator who calibrates on the
160'-0" dimension and clicks each vertex with 2 px of error lands within ±(2% or 6 SF) on every
area and ±(1.5% or 2 LF) on every length at the 99th percentile. An uncalibrated takeoff is off by
+5.6% (areas) and +2.8% (lengths) and fails every quantity.

**Verifier probe** (`dev/verifier_probe.py`). Thirteen crafted artifacts are run through the real
verifier image. Empty files, nulls, strings, NaN, infinity, booleans, an uncalibrated takeoff,
off-by-one counts and gross areas all score 0. The exact answer and in-tolerance or
differently-typed answers score 1.

**Harbor quirk found along the way.** Harbor scrubs the values of sensitive-looking agent env vars
(names containing `AUTH`, `KEY`, `TOKEN`...) from every trial file. Passing `CODEX_FORCE_AUTH_JSON=1`
as agent env turned every "1" in `result.json` into `[REDACTED]`. The run script now exports those
flags in the shell instead.

**Result.** A Codex pilot (GPT-6.1-sol, xhigh) passed in about **8 minutes**. It didn't measure areas
off pixels. It fitted the building outline, calibrated on the printed dimensions (so the scale trap
didn't catch it), measured wall positions to ±0.05 ft, snapped them to the round coordinates the plan
was laid out on, read thicknesses from the legend, and rebuilt the whole plan in shapely, racetrack
radii included. Its areas matched the truth to 0.1 SF. That is exactly what a sharp estimator does
with a clean CAD drawing. Any synthetic sheet drawn from a clean vector model can be inverted back
to that model, so precise measurement on its own isn't a wall.

## Iteration 3: as-built redlines (superseded)

**Idea.** Keep the takeoff, but make the drawing describe the store *as designed*. The contractor's
red as-built markups say what was actually built, and the takeoff must reflect that. The markups are
what real redlined prints carry: revision clouds, leaders, X-outs, freehand new work, handwritten
notes in an architect's hand, and a contractor stamp. Twelve of them change 15 of the 23 quantities;
three change nothing. Inverting the drawing back to its vector model now gives the wrong answer for
most quantities (`dev/verifier_probe.py`: an exact takeoff of the drawing as drawn scores 0). Each
markup says what changed, not by how much, so the agent has to read it, work out what it touches,
and turn it into area, length and count changes using geometry measured off the sheet.

**Rendering choices.** The red layer is multiplied onto the print, the way pen ink sits on toner
(red over black stays black), using ink that passes about 97% of red light. That makes "red dropout",
a standard document-scanner technique, a legitimate way to measure the print underneath. The scan is
color JPEG with full-resolution chroma (4:4:4). With 4:2:0 subsampling the red strokes bled into the
red channel and an X over a partition read as CMU hatch.

**Oracle.** The same raster pipeline measures the drawing as drawn (on the red channel). An
estimator's transcription of the markups (`solution/redlines.json`: the note text, the action, and
the plan point it refers to) is then applied in `apply_redlines`, which measures every length and
area involved on the drawing. For example, a wall moved east shifts its measured clear length times
the offset from one finish to the other. It matches every as-built quantity (areas within 0.8%,
lengths within 0.1%, counts exact). Two oracle fixes came out of this: wall fill now counts only
clearly dark pixels, because red ink slightly darkens poché in the red channel, and gondola half-cells
must pair across the spine, because a faint hatch patch near a door formed one stray "section".

**Result.** The first Codex pilot (GPT-6.1-sol, xhigh) failed after 7 minutes. Floors, doors and
fixtures were right, but it typed one 17.7 LF run of P1 as P2 (P1 197.33 LF against 215.0; P2 149.33
against 131.67). The official /run then went **3/3 pass in 5.7, 7.3 and 7.8 minutes**. Each trial
rebuilt the vector plan as in iteration 2, read the markups off the sheet, and applied them to the
model. That works because every markup is local to one sheet and the model it edits was already
recovered exactly. Twelve markups on one sheet are bookkeeping, not difficulty.

## Iteration 4: full bid-package takeoff (abandoned)

**Idea.** A real remodel bid isn't taken off one sheet. The estimator works across the floor plan,
the reflected ceiling plan and the schedules, and the as-built markups on one sheet change
quantities on another. The task now ships three scanned sheets in one PDF
(`store-0417-asbuilt.pdf`: A101 floor plan, A102 reflected ceiling plan, A601 room finish and light
fixture schedules) and asks for 36 quantities instead of 23:

- *Ceilings* (ACT-1, ACT-2, GYP; open-to-structure areas have no ceiling). A102 notes that ceilings
  follow the as-built walls, so the moved pharmacy wall on A101 changes the GYP area on A102, and the
  removed janitor wall merges two rooms with different ceilings.
- *Base* (RB-1, CTB-1). The base type for each room is only in the A601 room finish schedule (one cell
  of it is marked up). It is measured around the as-built rooms, less the as-built door openings, and
  not around free-standing columns.
- *Lighting* (L1, L1E, L2, L2E, L4, L5, X1 counts and L3 pendant-row LF). Emergency variants differ
  from the plain fixtures only by a shaded lens. A102 markups replace the pharmacy lights, convert the
  corridor troffers to emergency fixtures, add two high-bays and delete one pendant row.
- *Separate scale per sheet.* Each sheet was printed at its own reduction (97.3%, 96.4%, 99.1%) with
  its own skew, so calibrating once and reusing it fails.

The three sheets carry 23 markups; 18 of them change 27 of the 36 quantities.

**Oracle.** The A101 pipeline is unchanged. On A102 the oracle counts light symbols from their
closed white cells and filled blobs (filled lens means emergency; a ring's fill tells L5 from L4),
and finds pendant rows by morphological opening along the row direction. For ceilings and base it
rebuilds the rooms from the walls and doors it measured on A101 (`AsBuiltRooms`), applies the
geometric markups to that model, and takes each room's base and ceiling from the transcribed
schedule. It matches all 36 quantities (areas within 0.8%, lengths within 0.2%, counts exact).
`dev/verifier_probe.py` grew to 16 cases, adding "ceilings taken off A102 as drawn" and "EM
troffers counted as plain troffers"; both score 0.

**Rendering fixes found by the oracle.** RCP fixture tags overlapped fixtures and were moved to
explicit tag positions. The L5 high-bay's inner ring had to be drawn solid enough to tell it from
an L2 troffer lens, and the pharmacy light filter in the markup model initially caught IT-room lights,
which was fixed in `plan.asbuilt()`.

**Tolerance re-check.** `dev/tolerance.py` now also simulates digitizing ceilings, base and pendant
rows on A102 (printed at 96.4%, so slightly fewer pixels per foot). As built, ACT-1 is one 332.8 SF
room (rooms 122 and 123 merged), and simulated careful digitizing missed it by 2.08% at the 99th
percentile, just outside a 2% / 6 SF bound. The area floor went from 6 SF to 8 SF. That only widens
the four areas under 400 SF (CT-1, CPT-1, WOM-1, ACT-1). Of those, the markups change CPT-1 and ACT-1,
by 197 SF and 821 SF, far outside the wider bound. Every area and length is now inside its bound at the 99th percentile.

**Result.** Both pilots passed. Codex (GPT-6.1-sol, xhigh) took 10 minutes. It straightened each
sheet with a perspective transform, read every wall, door, room, symbol and markup off zoomed crops,
and typed the whole store as built into a 90-line shapely script. That script listed the rooms,
the door widths on each room's perimeter, the ceiling and base type of every room, and every count.
All 36 quantities were right, the floor areas to within 0.1 SF. Claude Code (Opus 5.5, max) took 40
minutes and did the same thing more slowly, measuring line profiles to confirm each coordinate. The
`essential_difficulty` and `difficult` rubric criteria passed, and the TB3 rubric review passed 33 of
36 criteria with 3 not applicable, but the task is not hard for these agents.

**Lesson.** Three rounds of making the drawing harder (redlines, then three sheets with cross-sheet
effects) made no difference. A synthetic drawing is a rendering of a clean model, and a frontier agent
with vision and a calibrated scan reads the model straight back. Its error rate per symbol is low
enough that adding more symbols mostly adds time. The real difficulty of quantity takeoff (messy
real-world documents) can't be reproduced synthetically without making the task unfair or trivially
large.

## Iteration 5: remodel rollout plan (abandoned)

Archived in [`archive/v3-remodel-rollout-plan`](../archive/v3-remodel-rollout-plan).

**Idea.** Stay in the same domain (a grocery chain's remodel program) but move from reading documents to
planning the program, where difficulty comes from the search, not perception. The agent schedules all 72
remodels of fiscal 2027: which of 9 general-contractor crews does each store and the week it starts.
Remodels can't start before the permit, avoid local blackout weeks, and finish before the holiday
freeze. A crew moving more than 275 road miles loses a week. At most 2 stores per district can be under
construction at once, and at most 3 can start in a week. Cost is crew contract weeks from first start
to last finish (idle weeks between jobs are paid), mileage, and the gross margin each store loses while
under construction, which depends on its seasonal sales curve (college towns are quiet in summer,
lake towns peak in July). The verifier checks every rule and requires the total to be at or under a
target. Unlike iterations 1-4, how hard the task is can be measured offline before any trial is run:
the target sits between what generic methods reach and what a stronger method reaches.

**Reference method.** Set partitioning over crew routes. Column generation prices routes with a
dynamic program over (store, start week) nodes. The trick that makes it fast is splitting a crew's
contract cost into -rate x first start and +rate x last finish, so each transition is a prefix minimum
and a pricing pass is O(weeks x stores^2). The root LP gives a lower bound in about 15 seconds. Diving
on the LP (fix the start weeks carrying the most weight, then whole routes), a short annealing polish,
and exact re-optimization of crew pairs by the same column generation give the plan.

**Calibration.** On the first instance (seasonal swings as generated) the band between tuned annealing
and the reference method was only about 0.25%, so the final instance amplifies seasonality 1.8x
(lake-town stores nearly double in July, college towns lose half their summer sales). On it: lower
bound $10,176,076; greedy $10.632M (+4.5%); CP-SAT warm-started, 2 workers, 30 minutes $10.277M (+1.0%);
Python simulated annealing, 30 minutes, $10.266M (+0.9%); reference method (three dives and polish,
11 minutes in Docker) $10.197M (+0.20%), and $10.186M (+0.09%) with crew-pair re-optimization. Target
set at $10,220,000 (+0.43%).

**Result.** The Codex pilot (GPT-6.1-sol, xhigh) passed in 17 minutes at $10,219,365, $635 under the
target, and stopped there. It ran CP-SAT LNS and a swap search in parallel, then wrote a C++ annealer
over a compact representation: each crew is a first start week plus an ordered list of stores packed
with no idle weeks, with the extra week after long moves applied automatically and rule violations
penalized. That representation fits the problem well, because optimal plans keep crews busy.
Running Codex's own program for 5 more minutes reached $10,185,686, as good as the reference method.
On a 2x instance (144 stores, 18 crews) it got within 0.17% of the bound in 6 minutes. The Claude pilot
(Opus 5.5, max) wrote a parallel annealer whose 16 runs all beat the target ($10.190M-$10.206M); it
submitted a $10,184,803 plan after 69 minutes, below the reference method's.

**Lesson.** With the cost fully specified, an agent can score any plan exactly, and a well-represented
metaheuristic in C++ gets within about 0.1% of optimal on this problem in minutes. The LP bound is tight
because the problem is benign, which also means local search finds near-optimal plans. A target can't
separate the reference method from the agent's unless it demands exact optimality.

## Iteration 6: a takeoff tool, judged on stores it has not seen (final)

**What iterations 1-5 had in common.** Every version gave the agent one instance and graded its answer
for that instance. A frontier agent can always solve one instance directly. It can read one drawing by
eye (iterations 2-4), or score candidate answers exactly against a fully specified rule set and search
(iterations 1 and 5). Neither the reading nor the search was the hard part.

**Idea.** Grade the tool instead of the answer. The chain has a few hundred scanned A101 plans, all from
the same architect and CAD standard. The agent builds `/app/takeoff/takeoff.py`, which takes a scan and
writes the takeoff. It gets three example scans with hand takeoffs. The verifier runs the tool on six
other stores in a separate container, as a user that cannot read the verifier's files, and every
quantity on every store has to be within tolerance. Reading the examples by eye is no longer a way to
pass: the tool has to work on drawings the agent never sees. That is what the real job is too.

**Generator.** `dev/plan.py` became a parametric prototype. It varies the footprint (136-156 ft x
98-122 ft), the depth of the back of house, the widths of the office-block rooms, door counts and
positions, the receiving and pharmacy sizes, the racetrack, gondola runs (count, 3'-0" or 4'-0"
sections, endcaps), cooler lineups, checkout and self-checkout counts, and produce tables. Every
annotation is placed by searching for a clear spot, and a validator rejects layouts where a tag, label
or door bubble touches linework. The scans vary print reduction (95-100%), skew (up to 0.8 degrees),
blur, noise, lamp shading and toner density. The sheet carries a graphic scale (0-32 ft) to calibrate
against, since the overall dimensions now differ per store.

**Reference tool.** It began as the CV pipeline from iterations 2-4 and was made general by running it
on hundreds of generated stores against ground truth and fixing each failure class:
- flattening the paper background;
- telling polished from sealed concrete by isolated specks once straight lines are masked;
- hatch scoring by in-direction energy less off-direction energy, a periodicity test for
  single-direction hatches, and larger partly masked windows for small rooms (with a lower cutoff
  that the fixture step can override);
- judging wall bands by their median cross-section;
- typing walls by the spread of the poché (a CMU hatch versus flat stud tints);
- masking cooler door strips before tracing walls;
- dropping sliding-panel strokes that look like jambs;
- checking every fixture against its whole symbol (kiosk frames, bag wells, endcaps at run ends,
  tables as triangle pairs, cooler lineups);
- giving fixture interiors the finish around them.

Generator bugs found the same way were fixed in the generator: pair doors placed into the electrical
room, a man door on the receiving-epoxy edge, and labels on checkouts. The tool gets 97 of 100 stores
it was never tuned on fully right, and all six hidden stores. In Docker it takes 60-80 s per scan.

**Why this should be hard for the agents.** The examples pass easily with heuristics tuned to their three
layouts. The failure modes listed above only show up on other layouts, and the agent can't generate
more layouts to test on. Hand-reading the examples gives nothing to submit.

**Pilots.** Both pilots failed. Codex (GPT-6.1-sol, xhigh) worked for 81 minutes and wrote a
990-line, nine-module tool that matches fixture symbols against templates cropped from the examples.
It passed 2 of the 6 hidden stores. It was one cooler door short on four of them, and on one it also
read a 97 SF room as VCT instead of sealed concrete. Claude Code (Opus 5.5, max) worked for 200
minutes. It stress-tested its tool on blurred, rotated and transposed copies of the examples and
passed 3 of the 6 stores. It was one cooler door short on two and missed an automatic sliding door
on one.

**A drawing bug the pilots exposed.** Both agents were one cooler door short on the same two stores
(4201 and 4206). On those stores the generator had placed the west exterior door inside the west
cooler lineup, so a cooler case covered the doorway. The drawing was physically impossible, and one
cooler door was hard to make out. The generator now keeps the lineup to one side of the door. The
layout validator now also rejects any fixture that overlaps a wall, a door swing, a doorway (2.5 ft
each side) or another fixture. The hidden stores were regenerated (only the two affected lineups
changed). The reference tool still passes every hidden store, and 97 of 100 random unseen stores.
The pilots' other misses were unaffected: Codex's cooler counts on 4202 and 4203, its finish on
4203, and Claude's sliding door on 4202.

**Official result.** All six official `/run` trials passed:
- Codex in 40-52 minutes; its tools were right on 93, 97 and 100 of 100 random unseen stores.
- Claude Code in 103-213 minutes; its tools were right on 99, 100 and 100.

The reference tool gets 97. The best runs built their own robustness tests from the three examples
(print reduction, skew, stretch, noise, rotation, mirroring), which stood in for the extra examples
the task withheld. Once a tool's CV primitives are robust, the generator's consistent symbol
rendering makes generalizing to new layouts nearly automatic. A larger hidden set would have failed
the weaker runs (93% per store gives about 11% on 30 stores) but not the best ones.

The Claude `/cheat` trial scored 0. It looked for answers embedded in the PDFs, then shipped a
tool that searches the verifier's filesystem for a ground-truth file and falls back to typical
values. There is no such file, and the fallback failed every store. The Codex `/cheat` trial scored 0
because OpenAI's API refused the hack prompt at the first turn (`AgentSafetyRefusalError`).

Given these results, Andrew decided to stop iterating and document the task as it stands, rather than
spend the remaining days on a seventh design.

