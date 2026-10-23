#!/usr/bin/env python3
"""Stage the TB3 rubric review for the local task.

Same review task that upstream's scripts/review/stage_task.py builds in CI
(same instruction template, rubric, word-count helper and verifier), except the
task under review is copied from this checkout instead of fetched from a PR
commit.

Usage: python dev/review/stage_local.py <tb3-upstream checkout> <output dir>
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks" / "store-remodel-takeoff"


def stage(upstream: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    environment = destination / "environment"
    tests = destination / "tests"
    environment.mkdir(parents=True)
    tests.mkdir()
    shutil.copytree(TASK, environment / "task-under-review" / TASK.name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    shutil.copy(upstream / "scripts/review/instruction_word_counts.py", environment)
    (environment / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "ca-certificates curl git nodejs npm procps python3 "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN npm install -g @anthropic-ai/claude-code\n"
        "COPY task-under-review /app/task-under-review\n"
        "COPY instruction_word_counts.py /app/instruction_word_counts.py\n"
        "WORKDIR /app\n"
    )
    rubric = (upstream / "docs/prompts/task-implementation.toml").read_text()
    instruction = (
        (upstream / "scripts/rubric-regression/templates/instruction.md")
        .read_text()
        .replace("the quality rubric is at `/app/rubric.toml`.", "apply the rubric included below.")
    )
    (destination / "instruction.md").write_text(instruction + "\n\nRubric:\n\n```toml\n" + rubric + "```\n")
    (destination / "task.toml").write_text(
        'schema_version = "1.0"\n'
        'artifacts = ["/app/verdicts.json"]\n\n'
        "[agent]\n"
        "timeout_sec = 1800.0\n"
        'network_mode = "allowlist"\n'
        'allowed_hosts = ["api.anthropic.com"]\n\n'
        "[environment]\n"
        'network_mode = "no-network"\n'
    )
    (tests / "test.sh").write_text(
        "#!/bin/bash\n"
        "if [ -f /app/verdicts.json ] && [ -s /app/verdicts.json ]; then\n"
        "  echo 1 > /logs/verifier/reward.txt\n"
        "else\n"
        "  echo 0 > /logs/verifier/reward.txt\n"
        "fi\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.upstream.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
