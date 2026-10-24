# Iteration 8 candidate (built and checked, not trialled)

A change to the iteration 7 task that would likely have stopped the Codex trial that passed. It was
never run against the agents, so nothing here claims it meets the difficulty bar.

## What it changes

- **No callouts on the plans.** `dev/render.py` stops drawing the boxed finish tags and the
  partition hexagons. The hatch and poché, read against each sheet's legend, become the only record
  of which finish and wall type is where. The plan model still reserves the callouts' places, so the
  layouts and the ground truth are unchanged. The passing Codex tool read those finish tags by OCR.
- **Thirty hidden stores instead of twelve.** `dev/generate.py` makes six hidden stores per hidden
  office, every office in every scan state. A tool that is right on 92% of stores (the passing Codex
  tool's rate on 100 unseen stores) would pass thirty about 8% of the time, against about 37% for
  twelve.
- **Verifier timeout** raised from 4,500 s to 9,600 s in `task.toml`, for thirty stores at up to five
  minutes each.

The instruction, the reference tool, the verifier and the tolerances are unchanged.

## Checks it passed

| Check | Result | Record |
|---|---|---|
| Static checks | 28 / 28 | [`results/static_checks.txt`](results/static_checks.txt) |
| Docker oracle | reward 1 (31 / 31 tests) | [`results/oracle_nop.txt`](results/oracle_nop.txt) |
| Docker nop | reward 0 (0 / 31) | [`results/oracle_nop.txt`](results/oracle_nop.txt) |
| Verifier probe | 9 / 9 as expected | [`results/verifier_probe.txt`](results/verifier_probe.txt) |
| Reference tool on its scans | 3 / 3 examples and 30 / 30 hidden stores | [`results/reference_tool_task_scans.txt`](results/reference_tool_task_scans.txt) |

## How to build it

The scans are deterministic, so only the code change is kept here (the thirty hidden PDFs are about
120 MB). From the repository root:

```bash
git apply archive/iteration-8-candidate/iteration-8.patch
```

```bash
PYTHONPATH=dev .venv/bin/python dev/generate.py
```

```bash
.venv/bin/python dev/check_task_data.py
```
