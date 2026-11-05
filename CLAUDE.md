# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo fine-tunes LLMs with reinforcement learning (GRPO) so they write profitable Backtrader trading strategies. The model generates a Python `Strategy` class, which is backtested on historical Alpaca market data; the backtest result becomes the reward signal. Trained LoRA adapters are then compared against their base models on held-out periods.

## Repository Structure

```
trading_rl/                    # Shared package — ALL backtest / prompt / reward logic lives here
  config.py                    #   symbol universes + SYMBOL_ISSUES, train/dev/test splits, COST_LEVELS
  prompt.py                    #   make_prompt(symbol, start, end, compact, sections, plan, anonymize)
  context.py                   #   API reference, pre-window market stats, few-shot examples
  knowledge.py                 #   financial RAG: strategy corpus + BM25 KnowledgeBase
  codegen.py                   #   extract / validate / exec generated Strategy code
  backtest.py                  #   Backtester (alpaca-py bars, memory+disk cache, cerebro), BacktestResult
  rewards.py                   #   reward ladder, RewardConfig, score_strategy/score_many, make_strategy_reward
  scoring.py                   #   ScoringPool: backtests in killable worker subprocesses (hard timeout, parallel)
  evaluate.py                  #   evaluate_sample / evaluate_symbol / summarize (model-agnostic)
  baselines.py                 #   buy-and-hold / SMA 50-200 through the same backtester
  ladder.py                    #   LLM-engineering ladder: LadderConfig, stages, pipeline, stats
  robustness.py                #   post-hoc: cost levels, vs buy-and-hold, Wilcoxon/McNemar, by-year, duplicates
  dataset.py                   #   build_dataset() for GRPO (prompt + symbol/start/end columns)
  model.py                     #   Unsloth wrapper (GPU only; not imported by trading_rl/__init__)
pyproject.toml                 # pip-installable; [train] = GPU deps, [dev] = pytest + ruff
scripts/
  main.py                      # try / test / compare / train models (settings at the top of the file)
tests/test_smoke.py            # runs the smoke test under pytest (CI: .github/workflows/ci.yml)
docs/paper/                    # revised methodology (ladder)
```

## Running

**Never copy backtest/reward/prompt code into a notebook or script — import it from `trading_rl`.** Before the package existed this code was duplicated in 9 places and drifted.

- **Local, no GPU** (backtesting, Ollama/MLX evaluation): `pip install -e .`
- **Local training** (CUDA): `pip install -e ".[train]"`, then `python scripts/main.py train`
- **Colab / Kaggle**: notebooks (none are tracked right now) install with `%pip install "trading-rl @ git+https://github.com/adhamhelmy/llm-fine-tuning.git"`, so **changes to `trading_rl/` must be pushed before they're visible there**.

Tests: `pip install -e ".[dev]" && ruff check . && pytest` (CI runs the same on Python 3.10 and 3.12). Offline tooling lives in `.pi/skills/`. The scripts are plain Python, so any agent can run them:
- `trading-rl-smoke-test/`: `scripts/smoke.py` exercises every reward path and the ladder (including the leakage assertions) using synthetic Alpaca bars. `scripts/run_notebook.py` runs CPU notebooks end to end with Alpaca and Ollama stubbed.
- `notebook-editing/`: `nbedit.py` makes asserting, minimal-diff notebook edits; `nb_dump.py` prints readable cells; `nb_lint.py` checks JSON, outputs, secrets and pyflakes. Read its SKILL.md before touching any `.ipynb`.

Anything in `trading_rl.model` needs CUDA and cannot run on a Mac.

Import `trading_rl.model` (i.e. unsloth) **before** trl/transformers so Unsloth can patch them.

## Credentials

Alpaca (historical bars only; paper endpoint `https://paper-api.alpaca.markets`), Hugging Face and W&B keys are **not** in the repo. Notebooks have empty placeholders (`HF_TOKEN = ''`) that the user fills in at runtime. `Backtester()` falls back to the `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` env vars (a local `.env` is gitignored). Never commit keys.

## Architecture

### Generation → reward pipeline

1. `make_prompt()` asks for a single `class Strategy(bt.Strategy)` with `__init__` and `next`. The prompt is strict: no imports (only `bt` is in scope), no external APIs, orders only via `self.buy/sell/close/order_target_percent`. `compact=True` gives the shorter prompt (no class skeleton), selectable in the ladder via `LadderConfig(compact=True)`.
2. `codegen.extract_function()` strips `<think>` blocks and pulls the class from a fenced block or raw text.
3. `rewards.score_strategy()` validates the code, then backtests it with a 10s timeout, and returns a `Score(status, reward, result, error)`:

   | status | reward |
   |---|---|
   | `missing_methods` (no `__init__`/`next`) | -10 |
   | `invalid_code` (syntax error / non-stdlib import) | -3 |
   | `exception` / `timeout` | -2 |
   | `no_trades` | -1 |
   | `loss` | 0 |
   | `profitable` | `max(metric, floor)`, by default `max(avg_annual_return_pct, 1)` |

   Both GRPO training (`make_strategy_reward`) and every evaluation path use this same function, so statuses and rewards stay consistent.
   - **`RewardConfig`** decides what a traded strategy earns and what the backtest charges: `metric` (`avg_annual`, `cagr`, `return`, `sharpe`, `calmar`, `excess_cagr`, `excess_return`, or any `fn(result)` / `fn(result, benchmark)`), `floor`, `loss` (None = use the metric), `clip`, and `costs` (a `COST_LEVELS` name or `(commission_bps, slippage_bps)`).
   - The default is `RewardConfig()`: `avg_annual` with `'low'` costs (1 + 2 bps per side). `RewardConfig.original()` is the frictionless reward of the first runs.
   - `cfg.name` is stored in the `reward` column of result CSVs. `run_ablation` resumes per reward, and rows without that column are read as `original()`.
   - Pass the **same** config to training, `run_ablation`, `evaluate_*` and `baseline_table`.
   - "Traded" means at least one filled order (`BacktestResult.n_orders > 0`).
4. `make_strategy_reward(backtester, reward=..., workers=..., save_dir=...)` wraps it in TRL's reward signature.
   - A TRL reward call holds a whole generation batch (per-device batch × grad-accum completions from several prompts), so each completion is scored on **its own row's** `symbol`/`start`/`end` dataset columns (falling back to parsing its own prompt). Never use `prompts[0]` for the whole batch.
   - Backtests run in parallel in a `ScoringPool`, and identical (code, window) pairs are memoised.
   - Profitable strategies can be saved (`strategy.py`, `stats.json`, `plot_*.png`).

### LLM-engineering ladder (the paper's methodology; see docs/paper/methodology.md)

- `LadderConfig` = one generation setting. Its fields are grouped by stage: prompt (`compact`, `examples`, `plan`, `anonymize`), context (`api_reference`, `market_summary`, `rag_k`), harness (`repair_rounds`), loop (`refine_rounds`) and graph (`analyst`). `cfg.name` lists the non-default fields and is the key stored in result CSVs.
- `STAGES` / `stage_configs(stage, best)`: greedy stage-wise selection. Stages are selected on **DEV**, and **TEST** is used only for the final rungs.
- **Leakage rules, which must hold for any change:**
  - repair/refine feedback and market stats use only `feedback_window(start)`, i.e. the 2 years before the evaluated window;
  - the evaluated window is backtested once per sample;
  - the RAG corpus contains no dated facts.
  The smoke test asserts all of these.
- `generate_strategy` runs one sample through the whole pipeline. `run_ablation` is resumable and appends to a CSV. `report` gives rates over *all* samples with Wilson CIs. `paired_delta` is a symbol-level bootstrap. `dump_prompts` writes the exact prompts for the paper appendix.
- **GRPO with the selected prompt:** `build_dataset(..., prompt_fn=ladder.prompt_fn(best, backtester))`. The prompt header must stay first (the reward falls back to parsing symbol/dates from it, and the model needs them), so `anonymize` is evaluation-only.
- `LadderConfig()` must render exactly `make_prompt(...)`, the original training prompt. Don't change `prompt._RULES`/`_TEMPLATE` without a new config flag; that would silently change the baseline.

### Backtester

- Bars come from **alpaca-py** (`StockHistoricalDataClient`, imported lazily). Timeframes are names (`'day'`, `'hour'`, …), and legacy alpaca `TimeFrame` objects are also accepted.
- **Bar cache:**
  - The in-memory cache (`Backtester._cache`) is class-level, so every instance shares it.
  - Bars are also pickled to `~/.cache/trading_rl/bars` (`TRADING_RL_CACHE` / `cache_dir=None` to disable). Reruns are offline and reproducible, even after a later split re-adjusts Alpaca's history.
  - Call `load_bars(symbols, start, end)` once up front. It warns about empty or forward-filled data (`bar_issues`, `config.SYMBOL_ISSUES`).
  - Tests patch `Backtester._fetch_bars`, never `get_bars`.
- Intraday timeframes are downloaded in 3-month chunks.
- `run(..., commission, slippage)` returns `BacktestResult(return_pct, sharpe_ratio, avg_annual_return_pct, max_drawdown_pct, cagr_pct, n_orders)`.
  - New fields go at the end, with defaults.
  - Observers other than Broker are only added when plotting (a speed-up; `AnnualReturn` needs Broker).
- Sharpe uses backtrader defaults, which is what the reward was trained with. Pass `sharpe_kwargs` for an annualised daily Sharpe (e.g. with a 5% risk-free rate).
- Plots are saved with `backtrader.plot.Plot` + `fig.savefig`. Don't use `cerebro.plot()`: it calls `plt.show()`, which blocks scripts and closes figures before they can be saved.
- Timeouts:
  - `call_with_timeout` uses SIGALRM on the main thread and falls back to a daemon thread elsewhere. This is in-process and soft: it can't stop C-level loops.
  - `scoring.ScoringPool` runs backtests in `python -m trading_rl.scoring` workers and kills/replaces any that overrun `timeout + grace`. Worker crashes become `exception` scores.
  - Pass `pool=` to `score_strategy` / `evaluate_*` / `generate_strategy` / `run_ablation`. `make_strategy_reward` creates one itself (`workers=0` = in-process).
- `codegen.extract_function` takes the first fenced block containing the class (else the raw text) and cuts at the first non-comment line not indented under the class header. `has_required_functions` uses `ast`, falling back to a regex on syntax errors so they stay `invalid_code`.
- `codegen.check_python_modules` is a dependency-free copy of unsloth_zoo's safety check (only stdlib imports allowed). The generated code is still `exec`'d (in a subprocess when pooled), so this is not a real sandbox.

### Training (GRPO)

- `Unsloth` loads the model with `FastVisionModel` (Unsloth's unified loader, also used for text-only models). It either creates a fresh LoRA adapter (`r=32`, `alpha=64`, q/k/v/o + gate/up/down proj) or loads an existing one via `lora_adapter_path`.
- `train()` defaults: lr 5e-6, `num_generations=2`, batch 1 × grad-accum 8, `adamw_8bit`. Override any `GRPOConfig` field through `**grpo_overrides`. It can push to the HF Hub (`hub_strategy="checkpoint"` makes interrupted runs resumable), and saves the adapter to `grpo_saved_lora/`.
- `generate()` decodes only the new tokens. Decoding the prompt too would let the extractor match the class skeleton inside the prompt.
- Dataset: one prompt per symbol in `TRAINING_SYMBOLS` (38 US equities), with `symbol`/`start`/`end` columns. With the defaults, TRL's `generation_batch_size` = 1 × 8 = 8 completions = 4 prompts × G=2 per reward call.
- **Known data problems (not yet fixed in training):** COIN has no bars before 2021-04, NIO is a muni fund's ticker before 2018-09, and DOW is flat 2017-09 → 2019-04 (`config.SYMBOL_ISSUES`, `usable_symbols`).
- Date ranges: `config` defines train 2016–2019, dev 2020–2021 and test 2022–2024 (no overlap).
- Models used so far: Ministral 3 (3B/8B/14B), Qwen2.5-Coder (7B/14B/32B), Llama 3.1 8B, DeepSeek-R1-Distill-Qwen-14B.

### Evaluation

- **Robustness (paper results):** `robustness.full_report(results_csv, backtester, out_dir=...)` re-backtests every traded strategy plus buy-and-hold at `COST_LEVELS` (commission + slippage bps per side) via `Backtester.run_detailed` (adds trade count, turnover, volatility, calendar-year returns), then writes the cost report vs B&H, per-asset Wilcoxon + BH-adjusted p, by-year/regime table and near-duplicate clusters. `mcnemar` / `paired_wilcoxon` compare two configs. Reporting only: never select configs on it. `COST_LEVELS` lives in `config` and the default reward uses `'low'`, so the matching cost level reproduces the evaluation numbers.

- `evaluate_sample(generate, backtester, symbol, start, end, ...)` accepts any `generate(prompt) -> str` callable (Unsloth, MLX, Ollama) and returns a flat record (status, reward, metrics, code, error).
- `summarize(df)` prints the status breakdown and metrics.
- Result CSVs use the columns `status`, `reward` (RewardConfig name) and `strategy_code`. `ladder.append_csv` widens the file when new columns appear, and `read_results` labels old rows.
- Base vs LoRA: list both in `MODELS` in `scripts/main.py` and run `compare`. Results go to `ladder_results/<model>/eval_<split>.csv`.

## Generated artifacts (gitignored)

`ladder_results/`, `outputs/`, `grpo_saved_lora/`, `successful_strategies/`, `mlx_adapters/`, `mlx_models/`, `backtest_plot_*.png`, `.env`.
