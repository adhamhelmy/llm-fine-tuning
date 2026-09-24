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
| `trading_rl/` | Shared package: prompt, code extraction, backtester, rewards, evaluation, Unsloth wrapper |
| `unsloth/` | Training (`main.py`, `kaggle_train.py`, `strategy_generator.ipynb`) and evaluation notebooks |
| `strategies/` | Hand-written strategies + `tester.py` CLI backtester |
| `models/` | Evaluation summaries per model |

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

**Train** (CUDA): run `unsloth/strategy_generator.ipynb` on Colab, push `unsloth/kaggle_train.py` as a Kaggle kernel, or run:

```bash
python unsloth/main.py
```

**Evaluate** a model by sampling strategies and backtesting them:

- `strategy_tester.ipynb` (Unsloth) or `strategy_tester_ollama.ipynb` (local Ollama) for a single model
- `model_comparison.ipynb` (Colab) or `model_comparison_mlx.ipynb` (Apple Silicon; run `convert_lora_adapters.ipynb` first) for base vs. LoRA across many symbols
- `strategy_benchmark.ipynb` to re-backtest the best strategies on a common benchmark

**Backtest a hand-written strategy:**

```bash
cd strategies
python tester.py dow30_momentum --symbols dow30 --start 2020-01-01 --end 2024-12-31
```

**Use the package directly:**

```python
from trading_rl import Backtester, make_prompt, evaluate_sample

bt = Backtester()                                 # reads ALPACA_* env vars
rec = evaluate_sample(my_generate_fn, bt, 'AAPL', '2022-01-01', '2023-12-31')
print(rec['status'], rec['reward_score'])
```
