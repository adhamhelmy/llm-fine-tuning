"""Shared code for GRPO fine-tuning of LLMs that generate Backtrader strategies.

Lightweight modules (no GPU needed): config, prompt, codegen, backtest, rewards, evaluate, dataset.
`trading_rl.model` (Unsloth wrapper) needs unsloth + CUDA and is not imported here.
"""

from .backtest import Backtester, BacktestResult, call_with_timeout, resolve_symbols
from .codegen import (extract_function, extract_strategy, function_works,
                      has_required_functions, strip_thinking)
from .config import (DOW30, MAG7, SYMBOL_SETS, TESTER_SYMBOLS, TRAIN_END, TRAIN_START,
                     TRAINING_SYMBOLS)
from .evaluate import evaluate_sample, evaluate_symbol, summarize
from .prompt import extract_trading_parameters, make_prompt
from .rewards import REWARDS, Score, make_strategy_reward, save_strategy, score_strategy

__all__ = [
    'Backtester', 'BacktestResult', 'call_with_timeout', 'resolve_symbols',
    'extract_function', 'extract_strategy', 'function_works', 'has_required_functions', 'strip_thinking',
    'DOW30', 'MAG7', 'SYMBOL_SETS', 'TESTER_SYMBOLS', 'TRAIN_END', 'TRAIN_START', 'TRAINING_SYMBOLS',
    'evaluate_sample', 'evaluate_symbol', 'summarize',
    'extract_trading_parameters', 'make_prompt',
    'REWARDS', 'Score', 'make_strategy_reward', 'save_strategy', 'score_strategy',
]
