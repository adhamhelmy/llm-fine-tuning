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
| Profitable | `max(avg annual return %, 1)` |

## Layout

| Path | What |
|---|---|
| `trading_rl/` | Shared package: prompt, context/RAG, code extraction, backtester, rewards, evaluation, ladder ablation, Unsloth wrapper |
| `unsloth/` | Ladder experiment notebook, `main.py` GRPO training |
| `docs/paper/` | Revised methodology (LLM-engineering ladder) |

## Setup

```bash
pip install -e .            # backtesting + evaluation (no GPU)
pip install -e ".[train]"   # + Unsloth/TRL for GRPO training (CUDA GPU)
```

On Colab / Kaggle, the notebooks install the package straight from GitHub:

```
%pip install "trading-rl @ git+https://github.com/adhamhelmy/llm-fine-tuning.git"
```

Credentials are not stored in the repo. Fill in the empty `HF_TOKEN` / `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` placeholders in a notebook, or set them as environment variables (or in a local `.env` for `strategies/tester.py`).

## Usage

Edit the settings at the top of `unsloth/main.py` (models, ladder configs, split, symbols), then:

```bash
python unsloth/main.py try       # one strategy, printed in full
python unsloth/main.py test      # first model on every config, with a report
python unsloth/main.py compare   # all models on the same configs (e.g. base vs. GRPO adapter)
python unsloth/main.py train     # GRPO (CUDA)
```

Models can be Ollama, MLX or Unsloth. See `docs/paper/methodology.md` for the ladder.

**Robustness report** (costs, vs buy-and-hold, per-asset tests, by-year): `trading_rl.robustness.full_report(results_csv, backtester, out_dir=...)`.

**Use the package directly:**

```python
from trading_rl import Backtester, make_prompt, evaluate_sample

bt = Backtester()                                 # reads ALPACA_* env vars
rec = evaluate_sample(my_generate_fn, bt, 'AAPL', '2022-01-01', '2023-12-31')
print(rec['status'], rec['reward_score'])
```
