# What to know going forward

## Old results you can't reuse

- **GRPO adapters trained before `c23f1f9`.** About 75% of their rewards came from backtests on
  the wrong stock. Retrain them, then redo base vs LoRA.
- **`ladder_results/` from before `b6df070`.** They include DOW (bad data), use the old universe,
  and were scored without trading costs. Rerun them on `EVAL_SYMBOLS`.

## Decisions before the final runs

1. **Reward.** Set `REWARD` in `scripts/main.py`, and use the same one for training and
   evaluation.
   - Default: `RewardConfig()`, which is average annual return with 1 bp commission + 2 bps
     slippage per side.
   - Report CAGR rather than `avg_annual`, which doesn't compound.
   - To reward beating buy-and-hold instead of plain profit:
     `RewardConfig('excess_cagr', floor=None, loss=None, clip=(-0.99, None))`.
2. **GRPO group size.** With `num_generations=2`, TRL's group scaling turns every reward pair into
   ±0.707: a +60% vs +1% pair trains exactly like 0 vs −1, and equal pairs train nothing.
   - Use `num_generations=8`.
   - Consider `scale_rewards="batch"`, so the size of the reward gap counts.
3. **Training variety.** There are 35 prompts on one window (2016–2019, a bull market). Each prompt
   is seen about 11 times in 100 steps, and long-biased strategies win almost by default. Consider
   several random 1–2 year sub-windows per symbol.
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
