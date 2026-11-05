# Repo review

**Status:** ✅ fixed · ⚠️ needs a decision · 🔁 rerun needed

| # | Issue | Status |
|---|---|---|
| 1 | GRPO reward scored 75% of completions on the wrong stock | ✅ fixed · 🔁 retrain adapters |
| 2 | GRPO had too little variety; with G = 2 the reward size is ignored | ⚠️ |
| 3 | Training prompts truncated, completions capped at 511 tokens | ✅ |
| 4 | COIN / NIO / DOW have missing or fake (forward-filled) bars | ✅ dropped · 🔁 rerun ladder |
| 5 | Market summary is empty in training prompts | ⚠️ |
| 6 | Default Sharpe is based on 2–3 yearly data points | ⚠️ |
| 7 | Reward now includes trading costs, so old results aren't comparable | 🔁 |

---

## 1. The reward scored the wrong stock ✅ 🔁

TRL calls the reward once per *generation batch*: per-device batch × grad-accum = 1 × 8 = 8
completions, i.e. **4 prompts** at G = 2. The old reward took the symbol from `prompts[0]` and
backtested all 8 completions on it.

Trace from real TRL 0.22.2 `GRPOTrainer` (`python scripts/trace_grpo_batches.py`):

```
=== reward call 1: 8 completions, 4 distinct prompts
 i  prompt symbol   OLD reward scored on
 0  TSLA            TSLA
 1  TSLA            TSLA
 2  AAPL            TSLA   WRONG
 3  AAPL            TSLA   WRONG
 4  WMT             TSLA   WRONG
 5  WMT             TSLA   WRONG
 6  KO              TSLA   WRONG
 7  KO              TSLA   WRONG
-> 6/8 completions scored on the wrong symbol
```

- **Impact:** the format part of the reward still worked, but the profitability signal was
  mostly noise. Every adapter trained so far is affected.
- **Fix:** dataset rows carry `symbol` / `start` / `end` columns, and each completion is scored on
  its own row.
- **Action:** retrain, then redo base vs LoRA.

## 2. GRPO had too little variety ⚠️

- **G = 2 ignores reward size.** TRL normalises rewards within each group
  (`scale_rewards="group"`), so a pair always becomes ±0.707. Rewards (60, 1) and (0, −1) produce
  the same update. Groups with equal rewards (e.g. both −2) produce no update at all.
- **One prompt per symbol.** 35 prompts on one window (2016–2019). 100 steps × 4 prompts means
  each prompt is seen about 11 times.
- **The training period was a bull market,** so long-biased strategies are rewarded almost
  automatically.

**Suggested fixes, in order:**
1. `num_generations=8`
2. `scale_rewards="batch"`, so reward size counts again
3. Several random 1–2 year sub-windows per symbol
4. `RewardConfig('excess_cagr', floor=None, loss=None, clip=(-0.99, None))`, which rewards beating
   buy-and-hold
5. vLLM generation (`fast_inference=True`), the largest remaining speed-up

## 3. Prompt truncation ✅

| | Before | Now |
|---|---|---|
| max prompt | 512 | 3072 |
| max sequence | 1024 | 5120 |
| completion budget | ~511 | ~2047 |

Measured prompts (Qwen tokenizer): baseline ~320 tokens, + API reference ~1,420, full context
~1,990. Before, prompts with context were cut from the left, losing the symbol and dates, and
reasoning models ran out of tokens inside `<think>`. `train()` now refuses to start if any prompt
would be cut.

Evaluation still uses `MAX_NEW_TOKENS = 1536`; align it with training if you want them to match.

## 4. Bad-data symbols ✅ 🔁

Alpaca fills inactive tickers with flat, zero-volume bars:

| Symbol | Problem |
|---|---|
| COIN | no bars before the 2021-04 IPO, so every training prompt scored −2 |
| NIO | a muni fund's ticker until the 2018-09 IPO, flat at $14.67 |
| DOW | flat at $44.18 from 2017-09 to 2019-04, which includes the dev feedback window |

- `TRAINING_SYMBOLS` now has 35 symbols; `EVAL_SYMBOLS` has 29 (the Dow 30 minus DOW).
- Every other symbol is clean in every window.
- `load_bars()` warns about empty, late-starting or flat data.
- **Action:** existing `ladder_results/` include DOW. Rerun them.

## 5. Market summary is empty in training ⚠️

It needs the 2 years before the window, and Alpaca has no bars before 2016. Dev and test prompts
do get it, so if the ladder selects `market_summary`, GRPO trains without it.
**Options:** start the train split in 2018, or document the gap.

## 6. Default Sharpe is unreliable ⚠️

backtrader computes Sharpe from **yearly** returns by default, which on these windows is 2–3 data
points. Before using the `sharpe` reward or reporting Sharpe:

```python
Backtester(sharpe_kwargs=dict(timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.0))
```

## 7. Reward changes 🔁

- `RewardConfig()` charges 1 bp commission + 2 bps slippage per side. "Profitable" means after
  costs.
- `RewardConfig.original()` reproduces the old frictionless reward.
- CSVs record the reward in the `reward` column. `run_ablation` reruns old rows rather than mixing
  them with new ones.
- Prefer `cagr` over `avg_annual` for reporting; the old metric doesn't compound.

---

## Also worth knowing

- **Not a sandbox.** Generated code can import stdlib modules (`os`, `subprocess`). The worker
  processes contain crashes and hangs, not malicious code.
- **Bar cache.** `~/.cache/trading_rl/bars` keeps the price adjustments from download time. Delete
  it to refresh.
- **Hindsight bias.** The training universe includes stocks known *today* to have done well (NVDA,
  TSLA). Always report buy-and-hold alongside.
- **Not executed here:** `trading_rl/model.py` and `scripts/main.py train` (need CUDA). Everything
  else is covered by `pytest` and was checked against real Alpaca data.

## Checklist before the final runs

- [ ] Choose `REWARD` in `scripts/main.py` and use it for training and evaluation
- [ ] Decide G / `scale_rewards` / sub-windows (§2)
- [ ] Decide on market summary in training (§5)
- [ ] Set `sharpe_kwargs` if Sharpe is used (§6)
- [ ] Rerun the dev ladder on `EVAL_SYMBOLS`
- [ ] Retrain adapters, then redo base vs LoRA
- [ ] Push to GitHub before Colab/Kaggle runs
