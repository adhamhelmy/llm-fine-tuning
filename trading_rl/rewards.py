"""The reward ladder, shared by GRPO training and every evaluation path.

    missing __init__/next   -10   missing_methods
    unparseable / unsafe     -3   invalid_code
    exception / timeout      -2   exception / timeout
    never trades             -1   no_trades
    negative return           0   loss         (RewardConfig.loss)
    positive return          max(metric, 1)    profitable   (RewardConfig.metric / floor)

What a traded strategy earns is configurable with RewardConfig: the metric ('avg_annual'
(original), 'cagr', 'sharpe', 'calmar', 'excess_cagr' vs buy-and-hold, ... or any callable),
the floor for profitable strategies, the reward for losing ones, clipping, and the
transaction costs the backtest charges. Training and evaluation must use the same config;
`RewardConfig.original()` is the frictionless reward of the first experiments.
"""

import dataclasses
import inspect
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, NamedTuple, Optional, Tuple, Union

from .backtest import BacktestResult, Backtester
from .codegen import (describe_exception, extract_function, extract_strategy,  # noqa: F401
                      has_required_functions, validate_code)
from .config import COST_LEVELS, DEFAULT_COSTS
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


# ── metrics for traded strategies ──────────────────────────────────────────────
# fn(result) or fn(result, benchmark): `benchmark()` returns buy-and-hold's BacktestResult on
# the same symbol/window/costs (computed lazily, only for metrics that call it).

METRICS = {
    'avg_annual': lambda r: r.avg_annual_return_pct,      # original: mean calendar-year return %
    'cagr': lambda r: r.cagr_pct,                         # compound annual growth %
    'return': lambda r: r.return_pct,                     # total return %
    'sharpe': lambda r: r.sharpe_ratio if r.sharpe_ratio is not None else 0.0,
    'calmar': lambda r: r.cagr_pct / max(r.max_drawdown_pct, 1.0),
    'excess_cagr': lambda r, bench: r.cagr_pct - bench().cagr_pct,
    'excess_return': lambda r, bench: r.return_pct - bench().return_pct,
}


def _call_metric(fn, result, benchmark):
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 1
    return fn(result, benchmark) if n >= 2 else fn(result)


@dataclass(frozen=True)
class RewardConfig:
    """How a strategy that traded is rewarded, and what the backtest charges.

    metric: name in METRICS or a callable fn(result) / fn(result, benchmark) -> float.
    floor:  profitable strategies earn max(metric, floor); None = no floor.
    loss:   reward of a losing strategy (return <= 0 after costs); None = the metric itself.
    clip:   (lo, hi) bounds on the reward of traded strategies (None = unbounded side),
            e.g. clip=(-0.99, None) keeps a losing trader above a non-trader (-1).
    costs:  COST_LEVELS name or (commission_bps, slippage_bps) per side.

    Examples:
        RewardConfig()                                    # avg annual return, 'low' costs
        RewardConfig.original()                           # frictionless, as first trained
        RewardConfig('cagr')
        RewardConfig('sharpe', floor=0.1, loss=None, clip=(-0.99, 5))
        RewardConfig('excess_cagr', floor=None, loss=None, clip=(-0.99, None))
        RewardConfig(lambda r: r.cagr_pct - 0.5 * r.max_drawdown_pct, floor=None, loss=None)
    """
    metric: Union[str, Callable] = 'avg_annual'
    floor: Optional[float] = 1.0
    loss: Optional[float] = 0.0
    clip: Optional[Tuple[Optional[float], Optional[float]]] = None
    costs: Union[str, Tuple[float, float]] = DEFAULT_COSTS

    def __post_init__(self):
        if isinstance(self.metric, str) and self.metric not in METRICS:
            raise ValueError(f"unknown metric {self.metric!r}; use one of {list(METRICS)} or a callable")
        if isinstance(self.costs, str) and self.costs not in COST_LEVELS:
            raise ValueError(f"unknown cost level {self.costs!r}; use one of {list(COST_LEVELS)}")

    @classmethod
    def original(cls):
        """The reward the first GRPO runs used: avg annual return, floor 1, no costs."""
        return cls(costs='none')

    @property
    def cost_bps(self):
        return COST_LEVELS[self.costs] if isinstance(self.costs, str) else tuple(self.costs)

    @property
    def commission(self):
        return self.cost_bps[0] / 1e4

    @property
    def slippage(self):
        return self.cost_bps[1] / 1e4

    @property
    def needs_benchmark(self):
        fn = self._metric_fn
        try:
            return len(inspect.signature(fn).parameters) >= 2
        except (TypeError, ValueError):
            return False

    @property
    def _metric_fn(self):
        return METRICS[self.metric] if isinstance(self.metric, str) else self.metric

    @property
    def name(self):
        """Stored in result CSVs: metric and costs always, other fields when non-default."""
        metric = self.metric if isinstance(self.metric, str) else getattr(self.metric, '__name__', 'custom')
        costs = self.costs if isinstance(self.costs, str) else '{}+{}bps'.format(*self.costs)
        parts = [metric, f'costs={costs}']
        for f in ('floor', 'loss', 'clip'):
            v = getattr(self, f)
            if v != next(x.default for x in dataclasses.fields(self) if x.name == f):
                parts.append(f'{f}={v}')
        return ','.join(parts)

    def score(self, result: BacktestResult, benchmark=None):
        """(status, reward) of a backtest result. benchmark: () -> buy-and-hold result."""
        if not result.traded:
            return 'no_trades', REWARDS['no_trades']
        status = 'profitable' if result.return_pct > 0 else 'loss'
        value = None
        if status == 'profitable' or self.loss is None:
            value = _call_metric(self._metric_fn, result, benchmark)
            value = float(value) if value is not None and math.isfinite(value) else 0.0
        if status == 'profitable':
            reward = max(value, self.floor) if self.floor is not None else value
        else:
            reward = self.loss if self.loss is not None else value
        if self.clip:
            lo, hi = self.clip
            reward = max(reward, lo) if lo is not None else reward
            reward = min(reward, hi) if hi is not None else reward
        return status, reward


DEFAULT_REWARD = RewardConfig()


class Score(NamedTuple):
    status: str
    reward: float
    result: Optional[BacktestResult] = None
    error: Optional[str] = None


_BENCHMARKS = {}


def benchmark_result(backtester: Backtester, symbol, start, end, timeframe='day', cash=10_000,
                     reward: RewardConfig = DEFAULT_REWARD) -> BacktestResult:
    """Buy-and-hold on the same symbol/window/costs (cached; trusted code, run in-process)."""
    from .baselines import BASELINES
    key = (symbol, start, end, str(timeframe), cash, reward.cost_bps)
    if key not in _BENCHMARKS:
        _BENCHMARKS[key] = backtester.run(extract_strategy(BASELINES['buy_and_hold']), symbol, start, end,
                                          timeframe=timeframe, cash=cash,
                                          commission=reward.commission, slippage=reward.slippage)
    return _BENCHMARKS[key]


def _precheck(code):
    """Score for code that fails before any backtest, else None."""
    if code is None:
        return Score('missing_methods', REWARDS['missing_methods'],
                     error="No `class Strategy(bt.Strategy):` definition found in the output.")
    if not has_required_functions(code):
        return Score('missing_methods', REWARDS['missing_methods'],
                     error="The Strategy class must define both __init__(self) and next(self).")
    invalid = validate_code(code)
    if invalid:
        return Score('invalid_code', REWARDS['invalid_code'], error=invalid[:300])
    return None


def _finish(outcome, reward, backtester, symbol, start, end, timeframe, cash) -> Score:
    status, payload = outcome
    if status == 'timeout':
        return Score('timeout', REWARDS['timeout'], error='timeout')
    if status == 'exception':
        return Score('exception', REWARDS['exception'], error=payload)
    bench = lambda: benchmark_result(backtester, symbol, start, end, timeframe, cash, reward)  # noqa: E731
    status, value = reward.score(payload, bench)
    return Score(status, value, payload)


def _run_inprocess(code, backtester, symbol, start, end, timeframe, cash, timeout, reward):
    try:
        strategy = extract_strategy(code)
        result = backtester.run_with_timeout(strategy, symbol, start, end, timeframe=timeframe,
                                             cash=cash, timeout=timeout,
                                             commission=reward.commission, slippage=reward.slippage)
        return ('ok', result)
    except TimeoutError:
        return ('timeout', None)
    except Exception as e:
        return ('exception', describe_exception(e))


def score_strategy(code, backtester: Backtester, symbol, start, end, timeframe='day',
                   cash=10_000, timeout=10, reward: RewardConfig = None, pool=None) -> Score:
    """Validate and backtest extracted Strategy code, returning its status and reward.
    pool: a trading_rl.scoring.ScoringPool to run the backtest in a killable subprocess."""
    return score_many([(code, symbol, start, end)], backtester, timeframe, cash, timeout,
                      reward, pool)[0]


def score_many(items, backtester: Backtester, timeframe='day', cash=10_000, timeout=10,
               reward: RewardConfig = None, pool=None):
    """score_strategy for many (code, symbol, start, end) items; with a pool the backtests
    run in parallel."""
    reward = reward or DEFAULT_REWARD
    scores = [_precheck(code) for code, *_ in items]
    todo = [i for i, s in enumerate(scores) if s is None]
    if pool is not None and todo:
        outcomes = pool.run_many(backtester, [
            dict(code=items[i][0], symbol=items[i][1], start=items[i][2], end=items[i][3],
                 timeframe=timeframe, cash=cash, commission=reward.commission,
                 slippage=reward.slippage) for i in todo], timeout=timeout)
    else:
        outcomes = [_run_inprocess(items[i][0], backtester, *items[i][1:], timeframe, cash, timeout, reward)
                    for i in todo]
    for i, outcome in zip(todo, outcomes):
        _, symbol, start, end = items[i]
        scores[i] = _finish(outcome, reward, backtester, symbol, start, end, timeframe, cash)
    return scores


def save_strategy(code, result: BacktestResult, symbol, start, end,
                  save_dir="successful_strategies", backtester: Backtester = None, extra=None,
                  reward: RewardConfig = None):
    """Save strategy.py + stats.json to save_dir/<timestamp>_<symbol>/.
    If a backtester is given, the backtest is re-run (same costs) to save plots alongside."""
    reward = reward or DEFAULT_REWARD
    strategy_dir = os.path.join(save_dir, f"{datetime.now():%Y%m%d_%H%M%S_%f}_{symbol}")
    os.makedirs(strategy_dir, exist_ok=True)

    with open(os.path.join(strategy_dir, "strategy.py"), "w") as f:
        f.write(code)

    stats = {
        **(extra or {}),
        "symbol": symbol, "start": start, "end": end, "reward_config": reward.name,
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in result._asdict().items()},
    }
    with open(os.path.join(strategy_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if backtester is not None:
        try:
            backtester.run(extract_strategy(code), symbol, start, end,
                           commission=reward.commission, slippage=reward.slippage,
                           plot=True, plot_prefix=os.path.join(strategy_dir, "plot"))
        except Exception as e:
            print(f"(plotting failed: {str(e)[:100]})")

    print(f"Saved to {strategy_dir}/")
    return strategy_dir


def make_strategy_reward(backtester: Backtester, reward: RewardConfig = None, save_dir=None,
                         save_plots=True, timeout=10, workers=None, pool=None, cache_size=4096):
    """Build the GRPO reward function (TRL signature: fn(completions, prompts, **kwargs) -> list).

    Each completion is scored on its own row's symbol/window: the `symbol`, `start`, `end`
    dataset columns (build_dataset adds them; TRL passes them as kwargs), else parsed from its
    own prompt. A TRL reward call holds a whole generation batch, i.e. several prompts.

    reward:     RewardConfig (default: DEFAULT_REWARD). Use the same one for evaluation.
    save_dir:   if set, profitable strategies are saved there during training.
    save_plots: also re-run profitable backtests to save their charts.
    workers:    backtest subprocesses (default: cpu_count - 1, max 8). They run the batch in
                parallel and enforce a hard timeout. workers=0 runs in-process.
    pool:       an existing ScoringPool to use instead.
    cache_size: identical (code, window) pairs are scored once (GRPO repeats itself a lot).
    """
    reward = reward or DEFAULT_REWARD
    step = 0
    state = {'pool': pool}
    memo = OrderedDict()

    def _pool():
        if state['pool'] is None and workers != 0:
            from .scoring import ScoringPool
            state['pool'] = ScoringPool(workers=workers, timeout=timeout)
        return state['pool']

    def strategy_succeeds(completions, prompts=None, **kwargs):
        nonlocal step
        step += 1
        items = []
        for i, completion in enumerate(completions):
            if all(k in kwargs for k in ('symbol', 'start', 'end')):
                symbol, start, end = kwargs['symbol'][i], kwargs['start'][i], kwargs['end'][i]
            else:
                symbol, start, end = extract_trading_parameters(prompts[i][-1]["content"])
            text = completion[0]["content"] if isinstance(completion, list) else completion
            items.append((extract_function(text), symbol, start, end))

        todo = list(dict.fromkeys(it for it in items if it not in memo))
        for it, score in zip(todo, score_many(todo, backtester, timeout=timeout, reward=reward,
                                              pool=_pool())):
            memo[it] = score
            if len(memo) > cache_size:
                memo.popitem(last=False)
        scores = [memo[it] for it in items]

        windows = sorted({f"{s}--{a}--{b}" for _, s, a, b in items})
        print("=" * 50)
        print(f"Step {step} | {', '.join(windows)} | {len(todo)} new backtests")
        new = set(todo)
        for it, score in zip(items, scores):
            code, symbol, start, end = it
            if score.status == 'timeout':
                print(f"{symbol}: Timeout")
            elif score.status == 'exception':
                print(f"{_RED}{symbol}: Exception: {score.error[:100]}{_RESET}")
            elif score.status == 'profitable' and it in new:
                new.discard(it)
                print(f"{symbol}: profitable ({score.result})")
                if save_dir:
                    save_strategy(code, score.result, symbol, start, end, save_dir,
                                  backtester if save_plots else None, reward=reward)
        rewards = [s.reward for s in scores]
        print(rewards)
        return rewards

    strategy_succeeds.pool = _pool
    return strategy_succeeds
