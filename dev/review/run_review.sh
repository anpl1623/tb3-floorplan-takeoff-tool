#!/bin/bash
# Local equivalent of the TB3 CI rubric review (review.yml -> hosted_task_launcher.py review):
# stage the review task around this checkout's task and run it with the CI defaults,
# claude-code + anthropic/claude-sonnet-5. The verdicts land in the trial's artifacts
# as verdicts.json. Auth and the clean environment are the same as dev/trials/run_trials.sh.
#
# Usage: dev/review/run_review.sh <tb3-upstream checkout> [job-name]
set -euo pipefail
cd "$(dirname "$0")/../.."
upstream=${1:?path to a tb3 upstream checkout}
name=${2:-}
python3 dev/review/stage_local.py "$upstream" dev/out/review-task
vars=(HOME="$HOME" PATH="$PATH" USER="${USER:-}" LANG="${LANG:-en_US.UTF-8}" TMPDIR="${TMPDIR:-/tmp}"
      CLAUDE_FORCE_OAUTH=1)
exec env -i "${vars[@]}" harbor run --path dev/out/review-task --agent claude-code \
  --model anthropic/claude-sonnet-5 ${name:+--job-name "$name"} --env-file .env.trials --yes
