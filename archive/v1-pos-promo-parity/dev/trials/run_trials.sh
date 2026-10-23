#!/bin/bash
# Local equivalent of the TB3 CI /run and /cheat commands for pos-promo-parity.
#
# /run   : 3 attempts per agent (harbor-run-defaults.yml `trials: 3`)
# /cheat : 1 attempt per agent with docs/prompts/hack-trial-prompt.md passed as
#          `extra_instructions`, exactly as scripts/ci/hosted_trial_workflow.py does.
#
# Agents follow the assignment: claude-code + anthropic/claude-opus-5-5 at
# reasoning_effort=max, and codex + openai/gpt-6.1-sol at reasoning_effort=xhigh.
# CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 mirrors the env the CI config sets for
# claude-code. Subscription auth: CLAUDE_FORCE_OAUTH=1 with CLAUDE_CODE_OAUTH_TOKEN
# from .env.trials (gitignored), and CODEX_FORCE_AUTH_JSON=1 with ~/.codex/auth.json.
#
# Usage: dev/trials/run_trials.sh run|cheat claude|codex [job-name]
set -euo pipefail
cd "$(dirname "$0")/../.."
kind=${1:?run or cheat}
agent=${2:?claude or codex}
name=${3:-}
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
env_file=()
if [ "$agent" = claude ]; then
  env_file=(--env-file .env.trials)
fi
harbor run -c "dev/trials/${kind}-${agent}.json" ${name:+--job-name "$name"} ${env_file[@]+"${env_file[@]}"} --yes
