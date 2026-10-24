# Trial runs that ended on infrastructure, not on the model

Logged per the brief: usage limits and safety refusals are infrastructure, and the trial is rerun.

| Job | Agent | What happened | Reward | Rerun |
|---|---|---|---|---|
| it7-pilot-claude | claude-code, opus-5.5 max | `ApiRateLimitError` after 88 min (subscription rate limit) | 0 (tool right on 5/12 stores when cut off) | it7-pilot-claude-2 |
| it7-cheat-codex | codex, gpt-6.1-sol xhigh | `AgentSafetyRefusalError`: OpenAI flagged the hack prompt after 6 inspection commands | 0 | it7-cheat-codex-2 |
| it7-cheat-codex-2 | codex, gpt-6.1-sol xhigh | `AgentSafetyRefusalError` at the first turn, no actions | 0 | not rerun: the provider refuses this prompt every time (also iteration 6) |
| it7-run-codex (3 trials) | codex, gpt-6.1-sol xhigh | `NonZeroAgentExitCodeError` after 23 min: ChatGPT subscription usage limit, three trials running at once | 0 | it7-run-codex-2, one trial at a time after the limit reset |
| it7-run-codex-2 (trials 2 and 3) | codex, gpt-6.1-sol xhigh | `NonZeroAgentExitCodeError` within a minute: usage limit again after trial 1 (limit resets 7:51 AM) | 0 | to rerun after the reset |
| it7-run-codex-2 (trial 1, gZjLQvJ) | codex, gpt-6.1-sol xhigh | worked 201 min, then hit the usage limit; its tool was already complete and the verifier scored it | **1.0** (12/12 stores) | counts as a solve: the limit ended the session after the tool was finished |
