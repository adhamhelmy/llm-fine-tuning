"""The reward ladder, shared by GRPO training and every evaluation notebook.

    missing __init__/next   -10   missing_methods
    unparseable / unsafe     -3   invalid_code
    exception / timeout      -2   exception / timeout
    never trades             -1   no_trades
    negative return           0   loss
    positive return          max(avg_annual_return_pct, 1)   profitable
"""

import json
import os
from datetime import datetime
from typing import NamedTuple, Optional

from alpaca_trade_api.rest import TimeFrame

from .backtest import BacktestResult, Backtester
from .codegen import extract_function, extract_strategy, function_works, has_required_functions
from .prompt import extract_trading_parameters

REWARDS = {
    'missing_methods': -10,
    'invalid_code': -3,
    'exception': -2,
    'timeout': -2,
    'no_trades': -1,
    'loss': 0,
}

_RED, _RESET = "\033[91m", "\033[0m"


class Score(NamedTuple):
    status: str
    reward: float
    result: Optional[BacktestResult] = None
    error: Optional[str] = None


def score_strategy(code, backtester: Backtester, symbol, start, end,
                   timeframe=TimeFrame.Day, cash=10_000, timeout=10) -> Score:
    """Validate and backtest extracted Strategy code, returning its status and reward."""
    if not has_required_functions(code):
        return Score('missing_methods', REWARDS['missing_methods'])
    if not function_works(code):
        return Score('invalid_code', REWARDS['invalid_code'])
    try:
        strategy = extract_strategy(code)
        result = backtester.run_with_timeout(strategy, symbol, start, end,
                                             timeframe=timeframe, cash=cash, timeout=timeout)
    except TimeoutError:
        return Score('timeout', REWARDS['timeout'], error='timeout')
    except Exception as e:
        return Score('exception', REWARDS['exception'], error=str(e)[:200])

    if not result.traded:
        return Score('no_trades', REWARDS['no_trades'], result)
    if result.return_pct > 0:
        return Score('profitable', max(result.avg_annual_return_pct, 1), result)
    return Score('loss', REWARDS['loss'], result)


def save_strategy(code, result: BacktestResult, symbol, start, end,
                  save_dir="successful_strategies", backtester: Backtester = None, extra=None):
    """Save strategy.py + stats.json to save_dir/<timestamp>_<symbol>/.
    If a backtester is given, the backtest is re-run to save plots alongside."""
    strategy_dir = os.path.join(save_dir, f"{datetime.now():%Y%m%d_%H%M%S}_{symbol}")
    os.makedirs(strategy_dir, exist_ok=True)

    with open(os.path.join(strategy_dir, "strategy.py"), "w") as f:
        f.write(code)

    stats = {
        **(extra or {}),
        "symbol": symbol, "start": start, "end": end,
        "return_pct": round(result.return_pct, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 4) if result.sharpe_ratio is not None else None,
        "avg_annual_return_pct": round(result.avg_annual_return_pct, 4),
        "max_drawdown_pct": round(result.max_drawdown_pct, 4),
    }
    with open(os.path.join(strategy_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if backtester is not None:
        try:
            backtester.run(extract_strategy(code), symbol, start, end,
                           plot=True, plot_prefix=os.path.join(strategy_dir, "plot"))
        except Exception as e:
            print(f"(plotting failed: {str(e)[:100]})")

    print(f"Saved to {strategy_dir}/")
    return strategy_dir


def make_strategy_reward(backtester: Backtester, save_dir=None, save_plots=True, timeout=10):
    """Build the GRPO reward function (TRL signature: fn(completions, prompts, **kwargs) -> list).

    save_dir:   if set, profitable strategies are saved there during training.
    save_plots: also re-run profitable backtests to save their charts.
    """
    step = 0

    def strategy_succeeds(completions, prompts, **kwargs):
        nonlocal step
        step += 1
        symbol, start, end = extract_trading_parameters(prompts[0][-1]["content"])
        print("=" * 50)
        print(f"Step {step} | {symbol}--{start}--{end}")

        scores = []
        for completion in completions:
            code = extract_function(completion[0]["content"])
            score = score_strategy(code, backtester, symbol, start, end, timeout=timeout)
            if score.status == 'timeout':
                print("Timeout")
            elif score.status == 'exception':
                print(f"{_RED}Exception: {score.error[:100]}{_RESET}")
            elif score.status == 'profitable':
                print(code)
                if save_dir:
                    save_strategy(code, score.result, symbol, start, end, save_dir,
                                  backtester if save_plots else None)
            scores.append(score.reward)

        print(scores)
        return scores

    return strategy_succeeds
