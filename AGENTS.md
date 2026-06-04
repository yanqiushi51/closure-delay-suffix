# Repository Guidelines

## Project Structure & Module Organization

This repository is a local validation harness for the online `exit_hazard` reasoning-process proxy. Core Python modules live in `closure_delay/`; keep reusable logic there rather than in scripts. Command-line entry points live in `scripts/`, including `probe_decode_gate.py`, `extract_exit_hazard_features.py`, `score_exit_hazard_logits.py`, `train_exit_hazard_proxy.py`, and `evaluate_exit_hazard_proxy.py`. Generated metrics, CSVs, JSON summaries, hidden-state arrays, and plots belong under `outputs/`; treat these as experiment artifacts, not source modules.

## Build, Test, And Development Commands

There is no package build step or dependency lockfile in this repo. Run scripts with the project root as the working directory so local imports resolve correctly.

```bash
/home/ssn/.conda/envs/new_env/bin/python scripts/evaluate_exit_hazard_proxy.py \
  outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged \
  --hazard-points-csv outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged/exit_hazard_proxy/exit_hazard_points.csv \
  --eval-subdir exit_hazard_eval
```

Prefer small smoke runs before launching GPU jobs. Use synthetic or capped data for CPU-only path checks, and use `--max-samples` where available.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, `Path` for filesystem paths, dataclasses for structured experiment records, and type hints for public helpers. Follow existing naming: modules and functions use `snake_case`, dataclasses use `PascalCase`, and CLI flags use kebab-case such as `--hazard-points-csv`. Keep script files thin: parse arguments, call `closure_delay` library code, and print output locations.

## Testing Guidelines

No formal `tests/` directory or pytest configuration is currently present. Validate changes with focused smoke commands that exercise the affected path, then inspect the output CSV/JSON files in the chosen output directory. At minimum, run `py_compile` on changed scripts. If adding tests, place them under `tests/`, name files `test_*.py`, and prefer deterministic unit tests for metric helpers in `closure_delay/exit_hazard.py`, `stats.py`, and `runtime.py`.

## Commit & Pull Request Guidelines

Recent commits use short Chinese summaries such as `更新代码和数据` and `首次提交`. Keep commits concise and focused, and mention whether code, data, or generated outputs changed. Pull requests should describe the experiment or bug being addressed, list the exact validation command run, note GPU/model assumptions, and include key output paths. Add screenshots only when plot appearance changes.

## Security & Configuration Tips

Do not hard-code new private model paths, credentials, or machine-specific secrets beyond documented local defaults. Keep large generated artifacts in `outputs/` and avoid mixing caches such as `__pycache__/` into source changes.
