@AGENTS.md

## Independent validation (Claude Code)

- After a feature or experiment and its tests, validate the worker's results before drawing conclusions: spawn a fresh-context Opus sub-agent (not a fork) with self-contained instructions and the raw evidence paths, ask it to recheck numbers, method and claims adversarially, and present its findings alongside the conclusion.
- Same for plans and designs before they are presented for approval.
