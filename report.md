# Repo review: issues found and what was done

Status as of commit `c23f1f9` and its follow-up. "Fixed" means fixed in code and covered by the
smoke test (`pytest`). "Open" means it needs your decision or a GPU run.

## 1. The GRPO reward scored most completions on the wrong stock (fixed, retrain needed)

**What happened.** TRL ≥ 0.18 generates a whole *generation batch* per optimizer step, and calls the
reward function once for all of it:
`generation_batch_size = per_device_train_batch_size × num_processes × steps_per_generation`, where
`steps_per_generation` defaults to `gradient_accumulation_steps`. With our settings (1 × 1 × 8),
that is 8 completions from 4 different prompts (G = 2) per call.

The old `make_strategy_reward` read the symbol and dates from `prompts[0]` and backtested all 8
completions on that symbol.

**Trace.** This is real TRL 0.22.2 `GRPOTrainer` on CPU with a tiny model and our batch settings.
Reproduce with `python scripts/trace_grpo_batches.py`:

```
generation_batch_size=8 steps_per_generation=8 remove_unused_columns=False
=== reward call 1: 8 completions, 4 distinct prompts
 i  prompt symbol kwargs symbol OLD reward scored on wrong?
 0  TSLA          TSLA          TSLA
 1  TSLA          TSLA          TSLA
 2  AAPL          AAPL          TSLA                 WRONG
 3  AAPL          AAPL          TSLA                 WRONG
 4  WMT           WMT           TSLA                 WRONG
 5  WMT           WMT           TSLA                 WRONG
 6  KO            KO            TSLA                 WRONG
 7  KO            KO            TSLA                 WRONG
-> old reward: 6/8 completions scored on the wrong symbol
=== reward call 2: 8 completions, 4 distinct prompts
 ... JPM, JPM, NVDA, NVDA, MSFT, MSFT, GS, GS -> 6/8 scored on JPM
```

**Impact.**
- In every previous GRPO run, 75% of rewards came from a backtest on a stock the model was not asked
  about.
- The format part of the reward still worked, since missing methods, invalid code and runtime
  errors don't depend on the stock. The *profitability* signal, which is the point of the paper,
  was mostly noise.
- Previous base-vs-LoRA comparisons measure a model trained on that noise.

**Fix.**
- `build_dataset` adds `symbol` / `start` / `end` columns. GRPO keeps them
  (`remove_unused_columns=False` by default) and passes them to the reward as kwargs, shown in the
  `kwargs symbol` column above.
- Each completion is scored on its own row. If the columns are missing, the reward parses the
  completion's *own* prompt instead.

**Caveat.** This was traced with plain TRL. Unsloth's patched trainer is generated from TRL's source
and, as far as I can tell, batches the same way. The fix is correct either way because it reads
per-row data.

**To do:** retrain every adapter you want to report.

## 2. GRPO had very little variety to learn from (open: suggested fixes)

- **G = 2 throws away the reward size.** TRL normalises each group's rewards: `(r − mean) / std`,
  with `scale_rewards="group"` by default (TRL 0.22). With two samples this is always ±0.707 or 0, whatever the
  gap. So +60% vs +1% annual return gives the same update as 0 vs −1. Groups where both samples
  get the same reward give no gradient at all, and those are common early on (both −2 or both −10).
  Only the *ranking* within a pair reaches the model. Checked with TRL's formula: rewards
  `(60, 1)` and `(0, −1)` both become advantages `(+0.707, −0.707)`.
- **Only one prompt per symbol.** 35 prompts, all on the same window (2016-01-01 → 2019-12-31).
  100 steps × 4 prompts per step = 400 draws, so every prompt is seen about 11 times. It is easy to
  memorise "what works for AAPL in 2016–2019".
- **The training period was a bull market.** Long-biased strategies earn the "profitable" reward
  almost automatically.

**Suggested fixes, roughly in order of value:**
1. `num_generations=6–8`, passed via `model.train(..., num_generations=8)`. Scoring is now parallel
   and memoised, so the reward side is cheap. Keep `generation_batch_size` divisible by G, e.g.
   batch 1 × grad-accum 8 with G = 8.
2. Consider `scale_rewards="batch"` (or `"none"`) so the reward size matters again.
3. Random sub-windows: several rows per symbol, each a random 1–2 year window inside
   2016–2019. `build_rows` only needs a list of (symbol, start, end) instead of one fixed window.
4. A metric relative to buy-and-hold (`RewardConfig('excess_cagr', floor=None, loss=None,
   clip=(-0.99, None))`), so "just be long in a bull market" stops paying.
5. Speed: vLLM generation (`fast_inference=True` in Unsloth) is the biggest remaining training
   speed-up. Generation dominates step time now that backtests run in parallel.

## 3. Training prompts were truncated and completions capped at 511 tokens (fixed)

- **Before:** `train_model` used `max_seq_length=1024` and `max_prompt_length=512`.
  - Completions were capped at about 511 tokens. A full strategy class is often 300–600 tokens, and
    reasoning models (DeepSeek-R1-Distill) spend that on `<think>` alone, which ends up as
    `missing_methods` (−10).
  - TRL cuts prompts that are too long from the **left**, which drops the header with the symbol and
    dates.
- **Measured prompt sizes** (Qwen tokenizer): baseline about 320 tokens, + API reference about
  1,420, full context stack about 1,990.
- **Now:**
  - `TRAIN_MAX_PROMPT = 3072` and `TRAIN_MAX_SEQ = 5120` (completions get about 2k tokens). The same
    values are the `Unsloth(...)` defaults.
  - `Unsloth.train()` calls `check_prompt_lengths(dataset)` and refuses to start if any prompt would
    be cut. I tested this with a real Qwen tokenizer: the old 512 budget is refused for every
    context config.
- **Mismatch:** evaluation generates up to `MAX_NEW_TOKENS = 1536`, while training now allows about
  2k. Align them if you want train/eval parity.

## 4. Symbols with bad data (fixed: dropped)

Alpaca fills inactive tickers with flat, zero-volume bars instead of returning nothing:

| Symbol | Problem | Where it hurt |
|---|---|---|
| COIN | no bars before the 2021-04-14 IPO | every training prompt scored −2, whatever the model wrote |
| NIO | a Nuveen muni fund's ticker until the 2018-09-12 IPO, flat at $14.67 | training |
| DOW | flat at $44.18 from the 2017 DowDuPont merger to the 2019-04-02 spin-off | training **and** the dev split's feedback window (2018–2019), which feeds repair/refine and the market summary |

- **What changed:**
  - `TRAINING_SYMBOLS`: 38 → 35 symbols (the first 38 minus these 3).
  - New `EVAL_SYMBOLS` (`SYMBOLS='eval'` in `scripts/main.py`): DOW30 minus DOW = 29 symbols, the
    same universe for dev and test.
  - `config.SYMBOL_ISSUES` documents the three.
  - `bar_issues()` and `load_bars()` flag empty, late-starting or forward-filled bars.
- **Scan:** every other symbol is clean in every window (train, both feedback windows, dev, test).
- **Existing results** in `ladder_results/` used the 30-symbol `dow30`, including DOW. Rerun them,
  or drop the DOW rows before comparing.

## 5. Other things you should know

- **`market_summary` is empty in training prompts.** It needs the 2 years before the window, but
  Alpaca has no bars before 2016, so the training-window stats fail and the section is left out.
  Dev/test prompts do get it. If the ladder selects `market_summary`, GRPO trains without it.
  Options: start the train split in 2018 (2016–2017 as look-back), or accept and document it.
- **The default Sharpe is computed from yearly returns.** backtrader's `SharpeRatio` defaults to
  `timeframe=Years`, `riskfreerate=0.01`, no annualisation. On 2–3 year windows that is 2–3 data
  points, so `RewardConfig('sharpe')` and `median_sharpe_traded` in reports are close to
  meaningless. If you want Sharpe, use
  `Backtester(sharpe_kwargs=dict(timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.0))`
  everywhere (training and evaluation).
- **The reward default changed: costs are on.** `RewardConfig()` charges 1 bp commission + 2 bps
  slippage per side, and "profitable" means after costs. Old CSV rows are labelled
  `avg_annual,costs=none`; `run_ablation` reruns them instead of mixing them in.
  `RewardConfig.original()` reproduces the old reward exactly.
- **`avg_annual` vs `cagr`.** The original metric is the mean of calendar-year returns: no
  compounding, and partial years count fully. On full-year windows the two are close (smoke example:
  71.9 vs 70.5). CAGR is the standard number to report.
- **Not a security sandbox.** Generated code may import any stdlib module (`os`, `subprocess`,
  `socket`). Worker processes contain crashes and hangs but not malicious code. Tightening
  `validate_code` to an allow-list (e.g. `math`, `statistics`) would change the `invalid_code` rate,
  so decide before the final runs.
- **The disk cache freezes Alpaca's price adjustments.** Bars in `~/.cache/trading_rl/bars` keep the
  split/dividend adjustment from when they were downloaded. That is good for reproducibility, but
  delete the cache if you want refreshed history.
- **Hindsight in the universe.** Training includes names known *today* to have done well (NVDA,
  TSLA, NFLX). Report buy-and-hold on the same names alongside every result; the report already
  computes `beat_bh`.
- **Not run here:** `trading_rl/model.py` (Unsloth/CUDA) and `scripts/main.py train`. The new
  prompt-length check was tested with Unsloth stubbed out and a real tokenizer. Everything else is
  covered by `pytest` (synthetic data), and the data code was checked against real Alpaca bars
  (alpaca-py bars are identical to the old API's).

## Checklist before the final runs

- [ ] Pick the reward (`REWARD` in `scripts/main.py`) and use it for training and evaluation.
- [ ] Decide on G / `scale_rewards` / random sub-windows (§2).
- [ ] Decide on `market_summary` for training (§5).
- [ ] Use `sharpe_kwargs` if Sharpe is reported or used as a reward (§5).
- [ ] Rerun the dev ladder on `EVAL_SYMBOLS` with the new reward. Old results are not comparable.
- [ ] Retrain the GRPO adapters (§1) and redo base-vs-LoRA.
- [ ] Push to GitHub before Colab/Kaggle runs (they install from GitHub).
