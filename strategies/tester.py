"""
Strategy Tester — backtest a hand-written strategy from this folder (uses trading_rl.backtest).

Usage (CLI):
    python tester.py dow30_momentum
    python tester.py dow30_momentum --symbols AAPL,MSFT,NVDA --start 2022-01-01 --end 2023-12-31
    python tester.py dow30_momentum --symbols dow30 --interval week --cash 50000

Usage (import):
    from tester import run_backtest
    from dow30_momentum import Strategy
    run_backtest(Strategy, symbols='dow30', start='2020-01-01', end='2024-12-31')
"""

import os
import sys
import importlib
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import backtrader as bt
import matplotlib
matplotlib.use('Agg')

from trading_rl.backtest import INTERVALS, Backtester, resolve_symbols

# Tester reports an annualised daily Sharpe with a 5% risk-free rate
# (the GRPO reward uses backtrader's default Sharpe settings).
SHARPE_KWARGS = dict(timeframe=bt.TimeFrame.Days, compression=1, riskfreerate=0.05, annualize=True)


def run_backtest(strategy, symbols='dow30', start='2020-01-01', end='2024-12-31',
                 interval='day', cash=100_000, plot=True, plot_prefix='backtest'):
    """
    Run a Backtrader strategy against Alpaca data.

    Args:
        strategy   : bt.Strategy subclass (the class itself, not an instance)
        symbols    : list of tickers, comma string, or named set ('dow30', 'mag7', 'training')
        start      : 'YYYY-MM-DD'
        end        : 'YYYY-MM-DD'
        interval   : 'day' | 'week' | 'month' | 'hour' | 'minute'
        cash       : starting portfolio value
        plot       : save backtest chart to PNG when True
        plot_prefix: filename prefix for saved charts

    Returns:
        dict with keys: total_return, sharpe, avg_annual, max_drawdown
    """
    backtester = Backtester(os.environ.get('ALPACA_API_KEY'), os.environ.get('ALPACA_SECRET_KEY'),
                            sharpe_kwargs=SHARPE_KWARGS, verbose=False)
    sym_list = resolve_symbols(symbols)
    backtester.load_bars(sym_list, start, end, timeframe=interval)

    r = backtester.run(strategy, sym_list, start, end, timeframe=interval, cash=cash,
                       plot=plot, plot_prefix=plot_prefix, show_plots=False)

    print()
    print("=" * 50)
    print(f"  Total return  : {r.return_pct:+.2f}%")
    print(f"  Avg annual    : {r.avg_annual_return_pct:+.2f}%")
    print(f"  Sharpe ratio  : {f'{r.sharpe_ratio:.3f}' if r.sharpe_ratio is not None else 'N/A'}")
    print(f"  Max drawdown  : {r.max_drawdown_pct:.2f}%")
    print("=" * 50)
    for path in backtester.last_plots:
        print(f"  Chart → {path}")

    return {
        'total_return': r.return_pct,
        'sharpe':       r.sharpe_ratio,
        'avg_annual':   r.avg_annual_return_pct,
        'max_drawdown': r.max_drawdown_pct,
    }


def load_strategy(strategy_name):
    """Import a Strategy class from the strategies/ folder by file name."""
    sys.path.insert(0, os.path.dirname(__file__))
    if strategy_name.endswith('.py'):
        strategy_name = strategy_name[:-3]
    module = importlib.import_module(strategy_name)
    return module.Strategy


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Backtest a strategy against Alpaca data')
    p.add_argument('strategy',
                   help='Strategy file name, e.g. dow30_momentum or day_trading_orb')
    p.add_argument('--symbols', default='dow30',
                   help='Comma-separated tickers or named set: dow30 | mag7 | training (default: dow30)')
    p.add_argument('--start',    default='2020-01-01', help='Start date YYYY-MM-DD')
    p.add_argument('--end',      default='2024-12-31', help='End date YYYY-MM-DD')
    p.add_argument('--interval', default='day',
                   choices=list(INTERVALS), help='Bar interval (default: day)')
    p.add_argument('--cash',     default=100_000, type=float, help='Starting cash (default: 100000)')
    p.add_argument('--no-plot',  action='store_true', help='Skip saving chart PNGs')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    strategy = load_strategy(args.strategy)
    run_backtest(
        strategy  = strategy,
        symbols   = args.symbols,
        start     = args.start,
        end       = args.end,
        interval  = args.interval,
        cash      = args.cash,
        plot      = not args.no_plot,
        plot_prefix = args.strategy,
    )
