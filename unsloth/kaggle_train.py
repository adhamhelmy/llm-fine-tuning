"""Kaggle script kernel: GRPO-train Ministral-8B (see kernel-metadata.json)."""

import subprocess, sys

def pip_install(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)

pip_install(
    "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git",
    "trl==0.22.2",
    "huggingface_hub",
    "trackio",
    "trading-rl @ git+https://github.com/adhamhelmy/llm-fine-tuning.git",
)

import os

from trading_rl.model import Unsloth  # import first: unsloth must load before trl/transformers

from trading_rl import TRAIN_END, TRAIN_START, TRAINING_SYMBOLS, Backtester, make_prompt, make_strategy_reward
from trading_rl.dataset import build_dataset

# ── Kaggle secrets ────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    ALPACA_API_KEY    = _secrets.get_secret("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = _secrets.get_secret("ALPACA_SECRET_KEY")
    HF_TOKEN          = _secrets.get_secret("HF_TOKEN")
except Exception:
    ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
    HF_TOKEN          = os.environ.get("HF_TOKEN", "")

MODEL_NAME  = "unsloth/Ministral-8B-Instruct-2410"
HF_REPO     = "adhamhelmy/ministral-8b-trading-grpo"
TRAIN_STEPS = 500
MAX_SEQ_LEN = 2048
WORK_DIR    = "/kaggle/working"


if __name__ == "__main__":
    backtester = Backtester(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    backtester.load_bars(TRAINING_SYMBOLS, TRAIN_START, TRAIN_END)

    model = Unsloth(model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN)
    # size the prompt budget from the actual prompt
    model.max_prompt_length = model.tokenizer(
        make_prompt("AAPL", TRAIN_START, TRAIN_END), return_tensors="pt").input_ids.shape[-1]

    model.train(
        steps=TRAIN_STEPS,
        reward_functions=[make_strategy_reward(
            backtester, save_dir=f"{WORK_DIR}/successful_strategies", save_plots=False)],
        dataset=build_dataset(),
        output_dir=f"{WORK_DIR}/outputs",
        save_dir=f"{WORK_DIR}/grpo_saved_lora",
        hub_model_id=HF_REPO,
        hf_token=HF_TOKEN,
        report_to="trackio",
        run_name="ministral-8b-trading-grpo-500steps",
        plot=False,
    )
