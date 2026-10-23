# Iterations 2-4: store remodel quantity takeoff (archived)

One store's scanned drawing set, taken off by the agent: a single A101 sheet (v2), the same sheet with
the contractor's red as-built markups (v3), and three sheets (A101 floor plan, A102 reflected ceiling
plan, A601 schedules) with markups that affect other sheets (v4). `task/` is the final v4 task as it
was trialled; `dev/` holds its generator, renderer, tolerance study and verifier probe (paths inside
refer to the repository layout of the time, `tasks/store-remodel-takeoff`). `results/` has the
static checks, rubric review, tolerance study, verifier probe and every trial outcome
(`trials.json`). Codex passed v2, v3 and v4; Claude Code passed v4. See `docs/iteration-log.md`.
