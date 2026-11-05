"""Shared code for GRPO fine-tuning of LLMs that generate Backtrader strategies.

Lightweight modules (no GPU needed): config, prompt, context, knowledge, codegen, backtest, rewards,
evaluate, baselines, ladder, robustness, dataset.
`trading_rl.model` (Unsloth wrapper) needs unsloth + CUDA and is not imported here.
"""

from .backtest import Backtester, BacktestResult, call_with_timeout, resolve_symbols
from .baselines import BASELINES, baseline_table
from .codegen import (extract_function, extract_strategy, function_works,
                      has_required_functions, strip_thinking, validate_code)
from .config import (DEV_END, DEV_START, DOW30, FEEDBACK_YEARS, MAG7, SYMBOL_SETS, TEST_END,
                     TEST_START, TESTER_SYMBOLS, TRAIN_END, TRAIN_START, TRAINING_SYMBOLS)
from .evaluate import evaluate_sample, evaluate_symbol, summarize
from .knowledge import KnowledgeBase
from .ladder import (STAGES, LadderConfig, build_prompt, dump_prompts, format_report,
                     generate_strategy, paired_delta, prompt_fn, report, run_ablation,
                     select_best, stage_configs)
from .prompt import extract_trading_parameters, make_prompt
from .robustness import (COST_LEVELS, REGIMES, cost_report, duplicates, format_cost_report,
                         full_report, mcnemar, paired_wilcoxon, per_asset, rebacktest,
                         signed_rank_test, year_report)
from .rewards import REWARDS, Score, make_strategy_reward, save_strategy, score_strategy

__all__ = [
    'Backtester', 'BacktestResult', 'call_with_timeout', 'resolve_symbols',
    'BASELINES', 'baseline_table',
    'extract_function', 'extract_strategy', 'function_works', 'has_required_functions', 'strip_thinking',
    'validate_code',
    'DEV_END', 'DEV_START', 'DOW30', 'FEEDBACK_YEARS', 'MAG7', 'SYMBOL_SETS', 'TEST_END', 'TEST_START',
    'TESTER_SYMBOLS', 'TRAIN_END', 'TRAIN_START', 'TRAINING_SYMBOLS',
    'evaluate_sample', 'evaluate_symbol', 'summarize',
    'KnowledgeBase',
    'STAGES', 'LadderConfig', 'build_prompt', 'dump_prompts', 'format_report', 'generate_strategy',
    'paired_delta', 'prompt_fn', 'report', 'run_ablation', 'select_best', 'stage_configs',
    'extract_trading_parameters', 'make_prompt',
    'COST_LEVELS', 'REGIMES', 'cost_report', 'duplicates', 'format_cost_report', 'full_report',
    'mcnemar', 'paired_wilcoxon', 'per_asset', 'rebacktest', 'signed_rank_test', 'year_report',
    'REWARDS', 'Score', 'make_strategy_reward', 'save_strategy', 'score_strategy',
]
