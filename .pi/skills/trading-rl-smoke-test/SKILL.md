---
name: trading-rl-smoke-test
description: Verify changes to the trading_rl package or the CPU-only ladder_ablation notebook (Ollama backend) offline, with synthetic Alpaca bars and a fake Ollama instead of real APIs or a GPU. Use after editing backtest/reward/prompt/evaluate code or those notebooks, or before telling the user something works.
---

# trading_rl offline smoke tests

This repo has no test suite, and the real flows need Alpaca keys and sometimes a CUDA GPU. These scripts stub both out. **Never call the real Alpaca API from a test.**

All paths below are relative to this skill directory. Run the commands from the repo root. Python is Homebrew 3.14; macOS has no `timeout` command, so use the bash tool's timeout.

## 1. Package smoke test (run after any `trading_rl/` change)

```bash
python3 scripts/smoke.py
```

It checks every reward status and its reward value, `<think>` stripping, the prompt round-trip, the TRL reward function, the timeout (on the main thread and in a thread), and `evaluate_symbol` + `save_strategy` including `plot_1.png`.

For the ladder it checks:
- the few-shot examples run;
- repair, refine and the analyst → coder graph, using a scripted fake LLM;
- **no test leakage**: a spy on `Backtester.run` asserts that feedback uses only the pre-window and that the test window is backtested once;
- anonymization, `LadderConfig()` == the original prompt, RAG, resumable `run_ablation`, `report` / `paired_delta` / `select_best`, and `dump_prompts`.

It exits 1 if any check fails.

If you change the reward ladder or add a status, update `SAMPLE_OUTPUTS` in `scripts/stubs.py` and the checks in `scripts/smoke.py`.

## 2. Run a CPU notebook end to end

- It runs from a temp copy of the repo, so no artifacts land in the real tree.
- Magics are dropped, so the notebook imports the **local** `trading_rl`, not the GitHub version.
- `--replace OLD=>NEW` asserts that `OLD` exists. Use it to shrink sample counts.

Ladder notebook, tiny config (about 1 min):

```bash
python3 scripts/run_notebook.py unsloth/ladder_ablation.ipynb --ollama \
  --replace "MODEL          = 'qwen3.8:27b-mlx'=>MODEL          = 'llama3.2:3b'" \
  --replace "SYMBOLS        = TESTER_SYMBOLS[:5]=>SYMBOLS        = TESTER_SYMBOLS[:3]" \
  --replace "N_SAMPLES_TEST = 2=>N_SAMPLES_TEST = 1" \
  --replace "TIMEOUT_SECONDS = 10=>TIMEOUT_SECONDS = 1"
```

## What can't be tested here

`trading_rl/model.py` and `unsloth/main.py` need unsloth/CUDA. Check them statically with the `notebook-editing` skill's `nb_lint.py` and `uvx pyflakes`, and tell the user they were not executed.

## Stub details

- `stubs.install_fake_bars()` patches `Backtester.get_bars` to return a deterministic random walk per symbol and fills the shared class-level cache. You can also put a custom frame in `Backtester._cache[(symbol, str(timeframe), start, end)]`; bars need `open/high/low/close/volume` columns and a tz-aware index.
- Synthetic prices are not realistic, so check statuses and plumbing, not return values.
