"""GRPO training dataset: one prompt per symbol."""

from .config import TRAINING_SYMBOLS, TRAIN_START, TRAIN_END
from .prompt import make_prompt


def build_rows(symbols=TRAINING_SYMBOLS, start=TRAIN_START, end=TRAIN_END, prompt_fn=make_prompt,
               windows=None):
    """Rows with the chat prompt plus symbol/start/end columns. TRL passes extra columns to
    the reward function as kwargs, so each completion is scored on its own window.
    windows: list of (start, end) pairs (one row per symbol x window); default [(start, end)]."""
    windows = windows or [(start, end)]
    return [{"prompt": [{"role": "user", "content": prompt_fn(sym, s, e)}],
             "symbol": sym, "start": s, "end": e}
            for s, e in windows for sym in symbols]


def train_windows(start=TRAIN_START, end=TRAIN_END):
    """The full training window plus its two halves (e.g. 2016-19, 2016-17, 2018-19): more
    prompt variety and a mix of regimes (2018 had a ~20% drawdown) without leaving the split."""
    from datetime import date
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    mid = s.year + (e.year - s.year + 1) // 2
    return [(start, end), (start, f"{mid - 1}-12-31"), (f"{mid}-01-01", end)]


def build_dataset(symbols=TRAINING_SYMBOLS, start=TRAIN_START, end=TRAIN_END, prompt_fn=make_prompt,
                  windows=None, shuffle_seed=None):
    """prompt_fn(symbol, start, end) -> str. Use trading_rl.ladder.prompt_fn(best_config, backtester)
    to train with the prompt + context chosen by the ladder ablation. windows: see build_rows
    (e.g. train_windows()); shuffle_seed mixes symbols/windows across steps."""
    from datasets import Dataset
    ds = Dataset.from_list(build_rows(symbols, start, end, prompt_fn, windows))
    return ds.shuffle(seed=shuffle_seed) if shuffle_seed is not None else ds
