# Iteration 5: remodel rollout plan (archived)

Schedule 72 store remodels across 9 crews at or under a cost target: column generation over crew
routes for the reference plan and a proven lower bound, with CP-SAT, simulated annealing and LNS
baselines for calibration. `task/` is the task as it was trialled; `dev/` holds the generator, the
rules module, the baselines (`cpsat_baseline.py`, `anneal.py`, `lns.py`, `greedy.py`), the column
generation (`colgen.py`) and calibration (`calibrate.py`). `results/trials.json` has the pilot
outcomes: both Codex and Claude Code passed. See `docs/iteration-log.md`.
