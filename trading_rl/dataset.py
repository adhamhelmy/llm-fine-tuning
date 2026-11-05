"""GRPO training dataset: one prompt per symbol."""

from .config import TRAINING_SYMBOLS, TRAIN_START, TRAIN_END
from .prompt import make_prompt


def build_rows(symbols=TRAINING_SYMBOLS, start=TRAIN_START, end=TRAIN_END, prompt_fn=make_prompt):
    """Rows with the chat prompt plus symbol/start/end columns. TRL passes extra columns to
    the reward function as kwargs, so each completion is scored on its own window."""
    return [{"prompt": [{"role": "user", "content": prompt_fn(sym, start, end)}],
             "symbol": sym, "start": start, "end": end}
            for sym in symbols]


def build_dataset(symbols=TRAINING_SYMBOLS, start=TRAIN_START, end=TRAIN_END, prompt_fn=make_prompt):
    """prompt_fn(symbol, start, end) -> str. Use trading_rl.ladder.prompt_fn(best_config, backtester)
    to train with the prompt + context chosen by the ladder ablation."""
    from datasets import Dataset
    return Dataset.from_list(build_rows(symbols, start, end, prompt_fn))
