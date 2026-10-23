#!/usr/bin/env python3
"""Summarize this task's trial jobs into results/trials.json and keep each trial's tool.

For every trial: agent and model, reward, agent minutes, any harbor exception,
which hidden stores passed, and the verifier's assertion messages. The agent's
submitted /app/takeoff directory (code only) is copied to results/trial-artifacts/
so the failure analysis can point at the code.

Usage: python dev/summarize_trials.py JOB [JOB ...]
"""

import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def minutes(a, b):
    t = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return round((t(b) - t(a)).total_seconds() / 60, 1)


def main():
    out_path = ROOT / "results" / "trials.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    rows = [r for r in rows if r["job"] not in sys.argv[1:]]
    for job in sys.argv[1:]:
        cfg = json.loads((ROOT / "jobs" / job / "config.json").read_text())
        agent = cfg["agents"][0]
        for trial in sorted((ROOT / "jobs" / job).glob("*/")):
            if not (trial / "result.json").exists():
                continue
            r = json.loads((trial / "result.json").read_text())
            ae = r.get("agent_execution") or {}
            log = (trial / "verifier" / "test-stdout.txt").read_text() if (trial / "verifier" / "test-stdout.txt").exists() else ""
            reward_file = trial / "verifier" / "reward.txt"
            row = dict(
                job=job, trial=trial.name, agent=agent["name"], model=agent["model_name"],
                kwargs=agent.get("kwargs"), cheat=bool(cfg.get("extra_instructions")),
                reward=float(reward_file.read_text()) if reward_file.exists() else None,
                agent_minutes=minutes(ae["started_at"], ae["finished_at"]) if ae.get("finished_at") else None,
                exception=(r.get("exception_info") or {}).get("exception_type"),
                exception_message=str((r.get("exception_info") or {}).get("exception_message") or "")[:300] or None,
                stores={m.group(2): m.group(1) for m in
                        re.finditer(r"^(PASSED|FAILED) test_takeoff.py::test_store\[(\d+)\]", log, re.M)},
                failures=sorted(set(re.findall(r"^E\s+(?:AssertionError|Failed): (.*)$", log, re.M))),
            )
            rows.append(row)
            src = trial / "artifacts" / "app" / "takeoff"
            if src.is_dir():
                dst = ROOT / "results" / "trial-artifacts" / job / trial.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.png", "*.jpg", "*.pdf",
                                                                         "*.npy", "*.npz"))
    out_path.write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        passed = sum(v == "PASSED" for v in r["stores"].values())
        print(f"{r['job']:14} {r['trial']:34} reward={r['reward']} {r['agent_minutes']} min "
              f"stores {passed}/{len(r['stores'])} {r['exception'] or ''}")


if __name__ == "__main__":
    main()
