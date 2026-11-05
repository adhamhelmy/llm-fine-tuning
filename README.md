# llm-fine-tuning

Fine-tuning LLMs with reinforcement learning (GRPO via [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl)) to write profitable [Backtrader](https://www.backtrader.com/) trading strategies.

The model is prompted to write a `class Strategy(bt.Strategy)`. The code is extracted, safety-checked and backtested on historical [Alpaca](https://alpaca.markets/) data, and the result becomes the reward:

| Outcome | Reward |
|---|---|
| Missing `__init__` / `next` | -10 |
| Invalid or unsafe code | -3 |
| Runtime error / timeout (10s) | -2 |
| Never trades | -1 |
| Loses money | 0 |
| Profitable (after costs) | `max(avg annual return %, 1)` (configurable: `RewardConfig`) |

## Layout

| Path | What |
|---|---|
| `trading_rl/` | Shared package: prompt, context/RAG, code extraction, backtester, rewards, evaluation, ladder ablation, Unsloth wrapper |
| `scripts/main.py` | Try / test / compare models, GRPO training |
| `tests/` | Offline smoke test under pytest (synthetic bars) |
| `report.md` | Review findings and open decisions before the final runs |
| `docs/paper/` | Revised methodology (LLM-engineering ladder) |

## Setup

```bash
pip install -e .            # backtesting + evaluation (no GPU)
pip install -e ".[train]"   # + Unsloth/TRL for GRPO training (CUDA GPU)
pip install -e ".[dev]"     # + pytest / ruff:  ruff check . && pytest
```

On Colab / Kaggle, install the package straight from GitHub:

```
%pip install "trading-rl @ git+https://github.com/adhamhelmy/llm-fine-tuning.git"
```

Credentials are not stored in the repo. Set `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (and `HF_TOKEN` for training) as environment variables or in a local `.env`. Downloaded bars are cached in `~/.cache/trading_rl/bars` (override with `TRADING_RL_CACHE`).

## Usage

Edit the settings at the top of `scripts/main.py` (models, ladder configs, split, symbols), then:

```bash
python scripts/main.py try       # one strategy, printed in full
python scripts/main.py test      # first model on every config, with a report
python scripts/main.py compare   # all models on the same configs (e.g. base vs. GRPO adapter)
python scripts/main.py train     # GRPO (CUDA)
```

Models can be Ollama, MLX or Unsloth. `REWARD` in the settings picks the reward (metric + transaction costs) for both training and evaluation:

```python
from trading_rl import RewardConfig
RewardConfig()                                        # avg annual return, 1 + 2 bps per side (default)
RewardConfig.original()                               # frictionless, as in the first experiments
RewardConfig('cagr')                                  # or 'sharpe', 'calmar', 'excess_cagr', ...
RewardConfig(lambda r: r.cagr_pct - 0.5 * r.max_drawdown_pct, floor=None, loss=None)
``` See `docs/paper/methodology.md` for the ladder.

**Robustness report** (costs, vs buy-and-hold, per-asset tests, by-year): `trading_rl.robustness.full_report(results_csv, backtester, out_dir=...)`.

**Use the package directly:**

```python
from trading_rl import Backtester, make_prompt, evaluate_sample

bt = Backtester()                                 # reads ALPACA_* env vars
rec = evaluate_sample(my_generate_fn, bt, 'AAPL', '2022-01-01', '2023-12-31')
print(rec['status'], rec['reward_score'])
```
