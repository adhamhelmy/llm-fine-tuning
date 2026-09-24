# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo fine-tunes LLMs with reinforcement learning (GRPO) so they write profitable Backtrader trading strategies. The model generates a Python `Strategy` class, which is backtested on historical Alpaca market data; the backtest result becomes the reward signal. Trained LoRA adapters are then compared against their base models on held-out periods.

## Repository Structure

```
trading_rl/                    # Shared package — ALL backtest / prompt / reward logic lives here
  config.py                    #   symbol universes, training date range, Alpaca URL
  prompt.py                    #   make_prompt(symbol, start, end, compact=False)
  codegen.py                   #   extract / validate / exec generated Strategy code
  backtest.py                  #   Backtester (Alpaca bars + cerebro), BacktestResult, timeouts
  rewards.py                   #   reward ladder, score_strategy, make_strategy_reward, save_strategy
  evaluate.py                  #   evaluate_sample / evaluate_symbol / summarize (model-agnostic)
  dataset.py                   #   build_dataset() for GRPO
  model.py                     #   Unsloth wrapper (GPU only; not imported by trading_rl/__init__)
pyproject.toml                 # makes trading_rl pip-installable; [train] extra = GPU deps
unsloth/
  main.py                      # local GRPO training / quick eval entry point (CUDA)
  kaggle_train.py              # Kaggle script kernel (config in kernel-metadata.json)
  strategy_generator.ipynb     # Colab GRPO training notebook (wandb, checkpoint resume)
  strategy_tester.ipynb        # Colab: sample N strategies from one Unsloth model
  strategy_tester_ollama.ipynb # same, against a local Ollama model
  model_comparison.ipynb       # Colab: base vs LoRA models across many symbols
  model_comparison_mlx.ipynb   # same, on Apple Silicon via mlx-lm
  convert_lora_adapters.ipynb  # PEFT adapters → MLX format (run before the MLX comparison)
  strategy_benchmark.ipynb     # re-backtest best strategies from model_comparison_results*.csv
strategies/
  tester.py                    # CLI backtester for hand-written strategies in this folder
  dow30_momentum.py, day_trading_orb.py
models/<model>/Summary.csv     # saved evaluation results per model
docs/superpowers/              # design spec + plan for the day_trading_orb strategy
remote-server.md               # university GPU server setup notes
```

## Running

**Never copy backtest/reward/prompt code into a notebook or script — import it from `trading_rl`.** Before the package existed this code was duplicated in 9 places and drifted.

- **Local, no GPU** (backtesting, Ollama/MLX evaluation): `pip install -e .`
- **Local training** (CUDA): `pip install -e ".[train]"`, then `python unsloth/main.py`
- **Colab / Kaggle**: notebooks run `%pip install "trading-rl @ git+https://github.com/adhamhelmy/llm-fine-tuning.git"`, so **changes to `trading_rl/` must be pushed before they're visible there**. When run from a local clone, notebooks import `../trading_rl` directly instead.
- **Hand-written strategies**: `cd strategies && python tester.py dow30_momentum --symbols dow30 --interval day`

There is no test suite. Offline tooling lives in `.pi/skills/`. The scripts are plain Python, so any agent can run them:
- `trading-rl-smoke-test/`: `scripts/smoke.py` exercises every reward path using synthetic Alpaca bars. `scripts/run_notebook.py` runs CPU notebooks end to end with Alpaca and Ollama stubbed.
- `notebook-editing/`: `nbedit.py` makes asserting, minimal-diff notebook edits; `nb_dump.py` prints readable cells; `nb_lint.py` checks JSON, outputs, secrets and pyflakes. Read its SKILL.md before touching any `.ipynb`.

Anything in `trading_rl.model` needs CUDA and cannot run on a Mac.

Import `trading_rl.model` (i.e. unsloth) **before** trl/transformers so Unsloth can patch them.

## Credentials

Alpaca (historical bars only; paper endpoint `https://paper-api.alpaca.markets`), Hugging Face and W&B keys are **not** in the repo. Notebooks have empty placeholders (`HF_TOKEN = ''`) that the user fills in at runtime. `Backtester()` falls back to the `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` env vars; `strategies/tester.py` loads them from `.env` (gitignored). Never commit keys.

## Architecture

### Generation → reward pipeline

1. `make_prompt()` asks for a single `class Strategy(bt.Strategy)` with `__init__` and `next`. The prompt is strict: no imports (only `bt` is in scope), no external APIs, orders only via `self.buy/sell/close/order_target_percent`. `compact=True` gives the shorter prompt (no class skeleton) used by the comparison notebooks.
2. `codegen.extract_function()` strips `<think>` blocks and pulls the class from a fenced block or raw text.
3. `rewards.score_strategy()` validates the code, then backtests it with a 10s timeout, and returns a `Score(status, reward, result, error)`:

   | status | reward |
   |---|---|
   | `missing_methods` (no `__init__`/`next`) | -10 |
   | `invalid_code` (syntax error / non-stdlib import) | -3 |
   | `exception` / `timeout` | -2 |
   | `no_trades` | -1 |
   | `loss` | 0 |
   | `profitable` | `max(avg_annual_return_pct, 1)` |

   Both GRPO training (`make_strategy_reward`) and every evaluation path use this same function, so statuses and rewards stay consistent.
4. `make_strategy_reward(backtester, save_dir=...)` wraps it in TRL's reward signature. Each completion is scored on the symbol/date range parsed from its prompt, and profitable strategies can be saved (`strategy.py`, `stats.json`, `plot_*.png`).

### Backtester

- The bar cache (`Backtester._cache`) is class-level, so every instance shares it. Call `load_bars(symbols, start, end)` once up front; training then never hits Alpaca.
- Intraday timeframes are downloaded in 3-month chunks.
- `run()` returns `BacktestResult(return_pct, sharpe_ratio, avg_annual_return_pct, max_drawdown_pct)`. `sharpe_ratio is None` means no trades.
- Sharpe uses backtrader defaults, which is what the reward was trained with. `strategies/tester.py` passes `sharpe_kwargs` for an annualised daily Sharpe with a 5% risk-free rate.
- Plots are saved with `backtrader.plot.Plot` + `fig.savefig`. Don't use `cerebro.plot()`: it calls `plt.show()`, which blocks scripts and closes figures before they can be saved.
- `call_with_timeout` uses SIGALRM on the main thread and falls back to a daemon thread elsewhere.
- `codegen.check_python_modules` is a dependency-free copy of unsloth_zoo's safety check (only stdlib imports allowed). The generated code is still `exec`'d, so this is not a real sandbox.

### Training (GRPO)

- `Unsloth` loads the model with `FastVisionModel` (Unsloth's unified loader, also used for text-only models). It either creates a fresh LoRA adapter (`r=32`, `alpha=64`, q/k/v/o + gate/up/down proj) or loads an existing one via `lora_adapter_path`.
- `train()` defaults: lr 5e-6, `num_generations=2`, batch 1 × grad-accum 8, `adamw_8bit`. Override any `GRPOConfig` field through `**grpo_overrides`. It can push to the HF Hub (`hub_strategy="checkpoint"` makes interrupted runs resumable), and saves the adapter to `grpo_saved_lora/`.
- `generate()` decodes only the new tokens. Decoding the prompt too would let the extractor match the class skeleton inside the prompt.
- Dataset: one prompt per symbol in `TRAINING_SYMBOLS` (38 US equities).
- Date ranges: `config.TRAIN_START/END` is 2016–2024, but `strategy_generator.ipynb` overrides it to 2012–2021 so that 2022+ stays held out for evaluation.
- Models used so far: Ministral 3 (3B/8B/14B), Qwen2.5-Coder (7B/14B/32B), Llama 3.1 8B, DeepSeek-R1-Distill-Qwen-14B.

### Evaluation

- `evaluate_sample(generate, backtester, symbol, start, end, ...)` accepts any `generate(prompt) -> str` callable (Unsloth, MLX, Ollama) and returns a flat record (status, reward, metrics, code, error).
- `summarize(df)` prints the status breakdown and metrics.
- Result CSVs use the columns `status` and `strategy_code`. Older `models/*/Summary.csv` files may still have `ok` and `code` from before the refactor.
- MLX flow: `convert_lora_adapters.ipynb` → `mlx_adapters/`, then `model_comparison_mlx.ipynb`.
- `strategy_benchmark.ipynb` concatenates `model_comparison_results*.csv` and re-runs the best strategy for each model/symbol.

## Generated artifacts (gitignored)

`outputs/`, `grpo_saved_lora/`, `successful_strategies/`, `mlx_adapters/`, `mlx_models/`, `backtest_plot_*.png`, `model_comparison_results*.csv`, `model_comparison_strategies/`, `strategy_benchmark_results.csv`, `.env`.
