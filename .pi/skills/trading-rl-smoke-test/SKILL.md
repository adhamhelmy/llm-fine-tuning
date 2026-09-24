---
name: trading-rl-smoke-test
description: Verify changes to the trading_rl package, strategies/tester.py, or the CPU-only notebooks (strategy_tester_ollama, strategy_benchmark) offline, with synthetic Alpaca bars and a fake Ollama instead of real APIs or a GPU. Use after editing backtest/reward/prompt/evaluate code or those notebooks, or before telling the user something works.
---

# trading_rl offline smoke tests

This repo has no test suite, and the real flows need Alpaca keys and sometimes a CUDA GPU. These scripts stub both out. **Never call the real Alpaca API from a test.**

All paths below are relative to this skill directory. Run the commands from the repo root. Python is Homebrew 3.14; macOS has no `timeout` command, so use the bash tool's timeout.

## 1. Package smoke test (run after any `trading_rl/` change)

```bash
python3 scripts/smoke.py
```

It checks every reward status and its reward value, `<think>` stripping, the prompt round-trip, the TRL reward function, the timeout (on the main thread and in a thread), and `evaluate_symbol` + `save_strategy` including `plot_1.png`. It exits 1 if any check fails.

If you change the reward ladder or add a status, update `SAMPLE_OUTPUTS` in `scripts/stubs.py` and the checks in `scripts/smoke.py`.

## 2. Run a CPU notebook end to end

```bash
python3 scripts/run_notebook.py unsloth/strategy_tester_ollama.ipynb --ollama \
    --replace "N_SAMPLES      = 20=>N_SAMPLES      = 6"
```

- It runs from a temp copy of the repo, so no artifacts land in the real tree.
- Magics are dropped, so the notebook imports the **local** `trading_rl`, not the GitHub version.
- `--replace OLD=>NEW` asserts that `OLD` exists. Use it to shrink sample counts.
- `strategy_benchmark.ipynb` needs a `model_comparison_results*.csv`. Create one with `--setup`, e.g. write a small DataFrame with columns `model, symbol, status, strategy_code, avg_annual_return_pct, sharpe_ratio` built from `stubs.SAMPLE_OUTPUTS['profitable']` (strip the markdown fence first).

## 3. strategies/tester.py

```bash
cd strategies && python3 -c "
import sys; sys.path[:0]=['..','../.pi/skills/trading-rl-smoke-test/scripts']
import stubs; stubs.install_fake_bars()
import tester
print(tester.run_backtest(tester.load_strategy('dow30_momentum'), 'AAPL,MSFT,KO', '2020-01-01', '2021-12-31', plot=False))"
```

## What can't be tested here

`trading_rl/model.py`, `unsloth/main.py`, `unsloth/kaggle_train.py`, and the notebooks `strategy_generator`, `strategy_tester`, `model_comparison` and `model_comparison_mlx` need unsloth/CUDA or mlx. Check them statically with the `notebook-editing` skill's `nb_lint.py` and `uvx pyflakes`, and tell the user they were not executed.

## Stub details

- `stubs.install_fake_bars()` patches `Backtester.get_bars` to return a deterministic random walk per symbol and fills the shared class-level cache. You can also put a custom frame in `Backtester._cache[(symbol, str(timeframe), start, end)]`; bars need `open/high/low/close/volume` columns and a tz-aware index.
- Synthetic prices are not realistic, so check statuses and plumbing, not return values.
