# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo fine-tunes LLMs via reinforcement learning (GRPO) to generate profitable Backtrader trading strategies. The model generates Python strategy classes, which are then backtested against real market data from Alpaca — the backtest return becomes the reward signal.

## Running

All primary work happens in Jupyter notebooks designed for **Google Colab** or **Kaggle** (GPU required: T4 minimum, A100/H100 preferred). There is no local test suite.

To run the class-based implementation locally (requires CUDA GPU and all dependencies):
```bash
cd unsloth
python main.py
```

To install dependencies in Colab/Kaggle, use the `%%capture` install cell at the top of each notebook. Key packages: `unsloth`, `trl==0.22.2`, `backtrader`, `alpaca_trade_api`, and a pinned transformers branch for Ministral support.

## Repository Structure

```
unsloth/
  main.py                          # Class-based Python implementation (canonical)
  strategy_generator.ipynb         # Latest notebook using the class-based approach
  Ministral_3_(3B)_Trading_Alpha_Miner.ipynb  # Main Kaggle notebook (flat/inline version)
  Llama3.1_(8B)-GRPO_trading.ipynb            # Earlier Llama 3.1 experiment
  Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb  # Sudoku RL reference
alpaca/
  strategy_backtesting.ipynb       # Standalone Alpaca + Backtrader backtesting demo
```

## Architecture

### Core Classes (`unsloth/main.py`)

**`Unsloth`** — wraps `FastVisionModel` (Unsloth's efficient loader) with LoRA config and exposes:
- `generate(inputs)` — runs inference with streaming
- `train(steps, reward_functions, dataset)` — runs GRPO via `GRPOTrainer`

LoRA settings: `r=32`, `lora_alpha=64`, targets `q/k/v/o_proj` + `gate/up/down_proj`, `use_gradient_checkpointing="unsloth"`.

**`Backtrader`** — wraps the Alpaca REST API and Backtrader cerebro engine:
- `run_backtest(strategy, symbols, start, end, ...)` — returns `(return_pct, sharpe_ratio)`
- `execute_strategy(...)` — same, but with a **10-second timeout** via `@execute_with_time_limit`

**`Data`** — builds the GRPO training dataset: a fixed prompt repeated `n_samples` times (default 10), formatted as `[{"role": "user", "content": prompt}]` with `"answer": 0`.

**`RewardFunctions`** — static methods used as GRPO reward functions:
- `extract_function(text)` — parses model output for a `class Strategy(bt.Strategy):` block
- `function_works(function)` — validates Python safety via `check_python_modules`
- `has_required_functions(text)` — checks for `__init__` and `next` methods
- `strategy_succeeds(completions, **kwargs)` — the main reward function; scores per completion:
  - `-10`: missing `__init__`/`next`
  - `-3`: invalid/unsafe code
  - `-2`: runtime exception or timeout
  - `-1`: no trades executed
  - `max(return_pct, 0)`: actual backtest return (clipped at 0 for losses)

### Training Loop (GRPO)

1. Prompt asks the model to generate a `class Strategy(bt.Strategy)` with `__init__` and `next`.
2. Model produces 2 completions per step (`num_generations=2`).
3. `strategy_succeeds` extracts, validates, and backtests each completion against a randomly chosen symbol/date range (2015–2024, 1-year windows, from a pool of 15 US equities).
4. Returns are used as rewards; GRPO updates model weights.
5. LoRA adapter is saved to `grpo_saved_lora/` or `outputs/` at the end.

### Alpaca API

Alpaca's **paper trading** endpoint (`https://paper-api.alpaca.markets`) is used for historical bar data only. API keys must be set as Colab secrets or hardcoded (avoid committing real keys). The `alpaca/strategy_backtesting.ipynb` notebook has a standalone demo with `run_backtest` and example strategies.

## Key Implementation Notes

- `FastVisionModel` is used even for text-only Ministral models (Unsloth's unified loader).
- `fast_inference=False` during training; set to `True` only for inference.
- `max_seq_length=4096`-`6144`; `max_prompt_length` is computed from tokenizing the actual prompt.
- The dataset prompt is strict: no imports, class must be named `Strategy`, no external APIs, orders only via `self.buy/sell/close/order_target_percent`.
- `check_python_modules` (from Unsloth) sandboxes code before `exec` to prevent unsafe execution.
- Models supported: Ministral 3B/8B/14B (Instruct, Reasoning, Base variants) and Llama 3.1 8B.
