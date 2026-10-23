# Trial runs that ended on infrastructure, not on the model

Logged per the brief: usage limits and safety refusals are infrastructure, and the trial is rerun.

| Job | Agent | What happened | Reward | Rerun |
|---|---|---|---|---|
| it7-pilot-claude | claude-code, opus-5.5 max | `ApiRateLimitError` after 88 min (subscription rate limit) | 0 (tool right on 5/12 stores when cut off) | it7-pilot-claude-2 |
| it7-cheat-codex | codex, gpt-6.1-sol xhigh | `AgentSafetyRefusalError`: OpenAI flagged the hack prompt after 6 inspection commands | 0 | it7-cheat-codex-2 |
| it7-cheat-codex-2 | codex, gpt-6.1-sol xhigh | `AgentSafetyRefusalError` at the first turn, no actions | 0 | not rerun: the provider refuses this prompt every time (also iteration 6) |
| it7-run-codex (3 trials) | codex, gpt-6.1-sol xhigh | `NonZeroAgentExitCodeError` after 23 min: ChatGPT subscription usage limit, three trials running at once | 0 | it7-run-codex-2, one trial at a time after the limit reset |
