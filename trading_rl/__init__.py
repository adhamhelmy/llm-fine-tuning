"""Shared code for GRPO fine-tuning of LLMs that generate Backtrader strategies.

Lightweight modules (no GPU needed): config, prompt, context, knowledge, codegen, backtest, rewards,
scoring, evaluate, baselines, ladder, robustness, dataset.
`trading_rl.model` (Unsloth wrapper) needs unsloth + CUDA and is not imported here.
"""

from .backtest import (Backtester, BacktestResult, bar_issues, call_with_timeout, resolve_symbols,
                       timeframe_name)
from .baselines import BASELINES, baseline_table
from .codegen import (describe_exception, extract_function, extract_strategy, function_works,
                      has_required_functions, strip_thinking, validate_code)
from .config import (COST_LEVELS, DEFAULT_COSTS, DEV_END, DEV_START, DOW30, EVAL_SYMBOLS, FEEDBACK_YEARS, MAG7,
                     SYMBOL_ISSUES, SYMBOL_SETS, TEST_END, TEST_START, TRAIN_END, TRAIN_START,
                     TRAINING_SYMBOLS, usable_symbols)
from .evaluate import evaluate_sample, evaluate_symbol, summarize
from .knowledge import KnowledgeBase
from .ladder import (STAGES, LadderConfig, append_csv, build_prompt, dump_prompts, format_report,
                     generate_strategy, paired_delta, prompt_fn, read_results, report, run_ablation,
                     select_best, stage_configs)
from .prompt import extract_trading_parameters, make_prompt
from .robustness import (REGIMES, cost_report, duplicates, format_cost_report, full_report, mcnemar,
                         paired_wilcoxon, per_asset, rebacktest, signed_rank_test, year_report)
from .rewards import (DEFAULT_REWARD, METRICS, REWARDS, RewardConfig, Score, benchmark_result,
                      make_strategy_reward, save_strategy, score_many, score_strategy)
from .scoring import ScoringPool

__all__ = [
    'Backtester', 'BacktestResult', 'bar_issues', 'call_with_timeout', 'resolve_symbols', 'timeframe_name',
    'BASELINES', 'baseline_table',
    'describe_exception', 'extract_function', 'extract_strategy', 'function_works',
    'has_required_functions', 'strip_thinking', 'validate_code',
    'COST_LEVELS', 'DEFAULT_COSTS', 'DEV_END', 'DEV_START', 'DOW30', 'EVAL_SYMBOLS', 'FEEDBACK_YEARS', 'MAG7',
    'SYMBOL_ISSUES', 'SYMBOL_SETS', 'TEST_END', 'TEST_START', 'TRAIN_END', 'TRAIN_START',
    'TRAINING_SYMBOLS', 'usable_symbols',
    'evaluate_sample', 'evaluate_symbol', 'summarize',
    'KnowledgeBase',
    'STAGES', 'LadderConfig', 'append_csv', 'build_prompt', 'dump_prompts', 'format_report',
    'generate_strategy', 'paired_delta', 'prompt_fn', 'read_results', 'report', 'run_ablation',
    'select_best', 'stage_configs',
    'extract_trading_parameters', 'make_prompt',
    'REGIMES', 'cost_report', 'duplicates', 'format_cost_report', 'full_report', 'mcnemar',
    'paired_wilcoxon', 'per_asset', 'rebacktest', 'signed_rank_test', 'year_report',
    'DEFAULT_REWARD', 'METRICS', 'REWARDS', 'RewardConfig', 'Score', 'benchmark_result',
    'make_strategy_reward', 'save_strategy', 'score_many', 'score_strategy',
    'ScoringPool',
]
