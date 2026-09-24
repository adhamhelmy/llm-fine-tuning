"""GRPO training dataset: one prompt per symbol."""

from .config import TRAINING_SYMBOLS, TRAIN_START, TRAIN_END
from .prompt import make_prompt


def build_dataset(symbols=TRAINING_SYMBOLS, start=TRAIN_START, end=TRAIN_END):
    from datasets import Dataset
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": make_prompt(sym, start, end)}], "answer": 0}
        for sym in symbols
    ])
