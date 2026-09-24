"""GRPO training / quick evaluation entry point (requires a CUDA GPU).

    pip install -e ".[train]"     # from the repo root
    python unsloth/main.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from trading_rl.model import Unsloth  # import first: unsloth must load before trl/transformers

import pandas as pd

from trading_rl import (TRAIN_END, TRAIN_START, TRAINING_SYMBOLS, Backtester, evaluate_sample,
                        make_strategy_reward, summarize)
from trading_rl.dataset import build_dataset

HF_TOKEN = os.environ.get('HF_TOKEN', '')
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')


def test(model_name, samples=10, lora_adapter_path=None):
    """Generate `samples` strategies with `model_name`, backtest them, and save a summary CSV."""
    backtester = Backtester(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    backtester.load_bars(TRAINING_SYMBOLS, TRAIN_START, TRAIN_END)
    model = Unsloth(model_name=model_name, max_seq_length=1024,
                    lora_adapter_path=lora_adapter_path, adapter_trainable=False, hf_token=HF_TOKEN)

    results = []
    for i in range(samples):
        symbol = random.choice(TRAINING_SYMBOLS)
        print(f"\n{'='*50}\nSample {i+1}/{samples} | {symbol}  {TRAIN_START} → {TRAIN_END}")
        rec = evaluate_sample(model.generate, backtester, symbol, TRAIN_START, TRAIN_END,
                              save_dir="successful_strategies")
        rec['sample'] = i + 1
        print(f"{rec['status']} (reward={rec['reward_score']})")
        results.append(rec)

    df = pd.DataFrame(results)
    summarize(df)

    save_dir = f"models/{model_name.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)
    df.drop(columns=["strategy_code"]).to_csv(f"{save_dir}/Summary.csv", index=False)
    print(f"\nSaved to {save_dir}/Summary.csv")
    return df


def main():
    backtester = Backtester(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    backtester.load_bars(TRAINING_SYMBOLS, TRAIN_START, TRAIN_END)

    model = Unsloth(model_name="unsloth/Ministral-3-3B-Instruct-2512", max_seq_length=1024)
    steps = 100
    model.train(
        steps=steps,
        reward_functions=[make_strategy_reward(backtester, save_dir="successful_strategies")],
        dataset=build_dataset(),
        hub_model_id=model.hf_repo_name(steps),
        hf_token=HF_TOKEN,
    )


if __name__ == "__main__":
    main()
