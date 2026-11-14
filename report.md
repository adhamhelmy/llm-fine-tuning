# What to know going forward

## Old results you can't reuse

- **GRPO adapters trained before `c23f1f9`.** About 75% of their rewards came from backtests on
  the wrong stock. Retrain them, then redo base vs LoRA.
- **`ladder_results/` from before `b6df070`.** They include DOW (bad data), use the old universe,
  and were scored without trading costs. Rerun them on `EVAL_SYMBOLS`.

## Decisions before the final runs

1. **Reward: decided (2026-11-14): Jensen's alpha**, `RewardConfig.alpha()` in `scripts/main.py`
   and the Colab notebook; use the same one for training and evaluation.
   - Alpha is the intercept of an OLS of daily strategy returns on daily buy-and-hold returns of
     the same symbol (risk-free 0), annualised in %/yr (`BacktestResult.alpha_pct`, `beta`, `alpha_t`).
   - Every trading strategy earns its alpha. Negative alphas are squashed into (-0.99, 0)
     (`alpha_soft`), so trading still beats not trading (-1), and the reward is capped at 50 %/yr.
   - Why: under `avg_annual`, staying long in the 2016–19 bull market wins by default. On the
     existing TEST rows, qwen3.8's AXP strategy made 21.3 %/yr (reward 21.3) with beta 0.80 and
     alpha −1.0: it was just riding the stock.
   - Costs stay at `'low'` (1 bp commission + 2 bps slippage per side). Report CAGR rather than
     `avg_annual`, which doesn't compound.
2. **GRPO group size: decided.** `num_generations=8`, `scale_rewards="batch"` (TRL 0.22 supports
   it). With `num_generations=2`, TRL's group scaling turned every reward pair into ±0.707.
3. **Training variety: decided.** `train_windows()` gives 2016–19 plus 2016–17 and 2018–19, i.e.
   35 × 3 = 105 prompts. The halves are cut from the cached full window (`Backtester._from_superset`),
   so they need no extra downloads.
4. **Market summary in training.** It needs 2 years of bars before the window, and Alpaca has none
   before 2016, so training prompts never include it while dev/test prompts do. Either start the
   train split in 2018 or document the gap.
5. **Sharpe.** backtrader computes Sharpe from **yearly** returns by default, which on these windows
   is 2–3 data points. If you report Sharpe or use it as a reward, set this everywhere:
   ```python
   Backtester(sharpe_kwargs=dict(timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.0))
   ```
6. **Token budgets.** Training allows about 2k completion tokens, but evaluation uses
   `MAX_NEW_TOKENS = 1536`. Align them if you want train/eval parity.

## Gotchas to remember

- **TRL batches several prompts into one reward call** (per-device batch × grad-accum completions).
  Any new reward function must score each completion on its own row (the `symbol`/`start`/`end`
  kwargs), never on `prompts[0]`. Verify with `python scripts/trace_grpo_batches.py`.
- **Prompt length grows with context.** Measured: baseline ~320 tokens, + API reference ~1.4k, full
  context ~2k. `train()` refuses prompts over `TRAIN_MAX_PROMPT` (3072). If you add context, check
  that it still fits.
- **Alpaca fills dead tickers with flat bars** instead of returning nothing (examples: COIN, NIO,
  DOW, listed in `config.SYMBOL_ISSUES`). Before adding a symbol or moving a window, run
  `load_bars()` and read its warnings. Remember the feedback window reaches 2 years back.
- **Reward settings are stored per row.** Result CSVs have a `reward` column. Changing the reward
  makes `run_ablation` rerun everything rather than mixing old and new rows.
- **Prices are frozen in the disk cache.** `~/.cache/trading_rl/bars` keeps the split/dividend
  adjustment from download time. Delete it to refresh.
- **Generated code isn't sandboxed.** Stdlib imports (`os`, `subprocess`) are allowed. Workers
  contain crashes and hangs, not malicious code. Tightening this changes the `invalid_code` rate,
  so decide before the final runs.
- **The universe has hindsight bias.** It includes names known *today* to have done well (NVDA,
  TSLA). Always report buy-and-hold on the same names.
- **Colab/Kaggle install from GitHub.** Push `trading_rl/` changes before running there.
- **Untested on GPU.** `trading_rl/model.py` and `scripts/main.py train` haven't been run since
  these changes. Do a short smoke run (a few steps) before a long one.
