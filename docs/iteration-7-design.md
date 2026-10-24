# Iteration 7 design: many drafting offices, one legend per sheet

## Difficulty mechanism

Real archives are drawn by many architects, not one. In this design, every store in the chain's
archive was drawn by one of eight architect's offices. Fixtures and doors are the owner's CAD blocks
and look the same everywhere. But each office has its own standard for floor-finish hatches and
partition poché: line hatches at its own angles and spacings, crosshatch, dot grids, brick and
herringbone bonds, dashed lines, flat grey tints, solid or hatched walls. Each sheet shows its own
office's swatches in its legend, in the owner's standard order. The sheets also reach the archive
in different states: clean scans, faded copies, photocopies whose tints became dither dots, and
yellowed colour scans.

The three examples come from three offices. The twelve hidden stores come from the other five, which
the agent never sees. So a tool can only classify finishes and walls by measuring each sheet's own
legend swatches at that sheet's scale and matching plan regions and wall runs against them, one shot
per sheet.

**What the best agent tool will get wrong.** The finish areas and partition lengths on stores from
offices it never saw. The hidden offices use pattern families the examples don't contain, and within
one office two finishes may differ only in spacing (6" vs 9" at the same angle), only in tone (a 20%
vs a 35% tint, both dithered on a photocopy), or only in bond (crosshatch vs brick at one spacing).
A classifier whose features, thresholds or nearest-match metric were tuned on the three example
offices will misassign some room or wall run somewhere among about 700 such judgments: 12 stores, each
with about 20 finish regions and 40 wall runs. One misassigned room moves an area by more than the 2%
tolerance, so the store fails.

**Why stress tests built from the examples can't fix that.** The winning tools in iteration 6 stress
tested the examples under rotation, blur, skew, stretch, noise and mirroring. That works when the
hidden stores are the same drawings degraded differently. Here the hidden stores are drawn
differently: the pattern families and pairings are not derivable from the three examples, only the
rule that the legend defines them is. To test a legend-driven matcher properly, an agent would have
to invent a distribution of office standards and repaint example regions with it. Whatever family or
near-duplicate pairing it doesn't think of goes untested, and the hidden offices were chosen to cover
families and pairings the examples don't show.

**Fairness.** Every sheet carries the information needed to read it: its legend defines every
finish and partition type, in the owner's standard order, drawn at plan scale. A human estimator
reads a new office's legend in seconds. An expert can build one-shot matching (spectral and tone
features of each legend swatch at the calibrated scale, nearest match per plan region) within the
expert time. The reference tool does exactly that and knows nothing about the hidden offices.

## What stays the same

- The output schema and the 23 quantities.
- The tolerances.
- The fixture and door blocks.
- The graphic-scale calibration.
- Verifier integrity: quantities computed from the vector plan models in a root-only `/tests`; the
  tool runs in place as an unprivileged user in a separate container; no answer file anywhere it can
  read; leftover processes killed before the reward is written.

The verifier probe is re-run after every change.

## Risks

- **The reference tool must be essentially perfect on all twelve hidden stores and on a broad
  random sample across all eight offices.** If one-shot matching can't reach that, the office
  standards are too close and get widened, rather than weakening the reference.
- **A strong agent may still build a general enough matcher.** The pilots decide: if either agent
  passes its pilot, this design is logged and documented, not taken to official runs.

## Outcome

Both pilots failed, and all three official Claude Code trials failed (11, 10 and 9 of 12 stores). One
of the Codex official trials passed all twelve stores. Its tool read the legend as intended, and also
read the boxed finish tags on the plans by OCR, which takes much of the finish judgment off the hatch
match where a room is tagged. Its tool is fully right on 92% of random stores, so twelve stores were
not enough to stop it reliably. See the iteration log for the iteration 8 candidate that removes the
callouts and uses thirty hidden stores.
