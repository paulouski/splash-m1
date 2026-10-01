# Repository guidance

This is an unofficial Splash fork based on Inco's source and the community Apple7 kernel port. The macOS 15 build uses Metal 3.2 and targets Apple7/8 GPUs. The reference machine is an M1 Max with a 24-core GPU and 32 GB memory on macOS 15.7. M2 support is intended but unverified; Apple9 and newer GPUs are not a supported guarantee.

## Start here

- README.md describes supported hardware, models, and current download availability.
- DEVELOPMENT.md covers source builds and package review.
- RUN.md has portable source and local-checkpoint examples.
- Checkout-local files under _local/, when present, are optional research notes and artifacts. Contributor setup and reviews must not depend on them.

## Scope and collaboration

- Make the smallest change that fixes the requested issue. Do not refactor adjacent code or add speculative behavior.
- Do not edit generated files or downloaded model artifacts unless the task explicitly requires it.
- Root owns requirements, decisions, integration review, and final verification. Use Codex-native GPT-6 Luna explorers or workers for non-trivial tasks when delegation materially reduces context or enables independent work. Native workers use fork_turns="none" and never run Git state-changing commands such as stash, checkout, reset, restore, clean, commit, or push. Root independently checks decision-critical source and diffs, then owns final verification. Do not use Claude Code agents.
- Keep user-specific paths, machine configuration, prompts, and private research data out of tracked files.

## Build and production safety

- The macOS 15 variant is the default build. Use make MACOS15=1 BUILD=build for an explicit source build; BUILD defaults to build.
- Put disposable experiments under `BUILD=build/experiments/<tag>`; preserve local snapshots under `_local/builds/`. Never overwrite a binary or metallib used by the production service.
- Production stays OFF unless the user explicitly asks to start it. Never start, stop, or restart it as an assumed part of another task.
- Only one large model process may use the GPU at a time. Before an authorized model run, confirm the production service is stopped and watch system memory and swap. Stop the run if swap grows by more than 2 GB or free plus inactive memory falls below 1.5 GB. A microbenchmark under 1 GB may run alongside production only when free plus inactive memory is at least 3 GB.
- Do not access or modify local model stores, application settings, or LaunchAgents unless the task explicitly calls for it.
- Never use git stash, checkout, reset, restore, or clean. Commit and push only when explicitly requested. Do not create worktrees.

## Measurement lessons

- Diagnose before changing kernels: profile per operation, use one-kernel counters or ablations, inspect shader costs, then implement only with a numeric forecast. Source-line cost is not the same as removable time.
- Compare identical shapes, dtypes, and timing methods. Mark extrapolations as estimates, not measurements, and recheck arithmetic against raw output.
- "Not built" does not mean "refuted." Confirm the intended workload reached the changed path before judging an experiment.
- Validate output quality with the full-vocabulary metric and report every gate that passes or fails. Do not substitute top-k comparisons for KL.
- Keep GPU captures small; a single dispatch or short kernel loop is usually enough for shader inspection. Fused totals and per-dispatch sums from separate runs are not exact addends.
- Independent review of raw data has found arithmetic mistakes and coverage gaps. Give reviewers the source evidence and raw measurements when available.

## Existing M1 observations

These are setup-specific historical measurements, not portable performance claims. Recheck them after changing the build, workload, or GPU.

- Eight-row Q4 decode was ALU-limited on the tested M1 Max. Making weights cache-resident did not improve it; profile ALU work and occupancy before assuming a memory bottleneck.
- Prefill kernels were at parity with MLX fp16 on identical shapes. Use fp16 for comparisons with this model.
- The current autoregressive mode still computes eight target rows; do not treat it as a one-token target pass.
- The full-vocabulary correctness gate has a top-1 bar of 99.7% and a mean KL bar of 1e-3. The latest noted result was 99.26% top-1 and 1.9e-4 mean KL, so it failed the top-1 bar; re-run before making current correctness claims.
- Prefix-cache snapshots retain the exact generated tokens and state needed for later turns. Preserve token identity when changing cache behavior.

Local experiment logs and artifacts may be kept under _local/ if available. They are optional evidence, not a prerequisite for building, contributing, or reviewing changes.
