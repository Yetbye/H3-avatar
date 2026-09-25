# Project Agent Instructions

## Goal

Build and evaluate a MiniMax H3 based system for continuous, controllable audio-video generation with realtime interaction and Agent integration.

## Workspace boundary

- The only writable remote project root is `/mnt/data/yetbye/h3-interactive`.
- Treat every sibling under `/mnt/data/yetbye` as read-only, including `minWM`.
- Stop before a command whose resolved output, cache, environment, log, checkpoint, or temporary path escapes the project root.
- Never use broad cleanup commands, unresolved destructive globs, or recursive deletion outside a verified run directory.
- Never store credentials in files, commands, logs, manifests, or Git history.

## Required context

- Read `docs/DEVELOPMENT_PLAN.md` before changing project scope or stage.
- Read `docs/ARCHITECTURE.md` before introducing modules or service interfaces.
- L2-L4 experiments require a completed `experiments/<campaign>/<run_id>/experiment.md` before execution.
- External repositories in `third_party/` are pinned references; implement local changes in `src/` or as explicit patch files.

## Working rules

- State assumptions and measurable success criteria before non-trivial work.
- Distinguish external claims, local smoke tests, offline evidence, realtime evidence, and long-horizon evidence.
- Do not silently change data splits, audio/video timing, token layout, attention masks, preprocessing, metrics, checkpoints, or serving contracts.
- Maintain one project-owned Conda environment at `.conda/envs/h3-interactive`. Treat dependency changes as stage transitions: export lock files before changes and rerun required smoke tests after changes.
- Run the smallest relevant smoke test before expensive inference or training.
- Never mark a Gate passed unless commands, logs, metrics, and artifacts support it.
- Do not commit, push, publish, download restricted assets, launch expensive training, or expose a public service without user authorization.
