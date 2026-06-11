"""
Strategy Tester — mirrors main.py's Backtrader class without the ML stack.

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
import time
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import backtrader as bt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame

ALPACA_API_KEY    = os.environ.get('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
BASE_URL          = "https://paper-api.alpaca.markets"

# ── Named symbol sets ─────────────────────────────────────────────────────────

SYMBOL_SETS = {
    'dow30': [
        'AAPL', 'AMGN', 'AXP',  'BA',   'CAT',
        'CRM',  'CSCO', 'CVX',  'DIS',  'DOW',
        'GS',   'HD',   'HON',  'IBM',  'INTC',
        'JNJ',  'JPM',  'KO',   'MCD',  'MMM',
        'MRK',  'MSFT', 'NKE',  'PG',   'TRV',
        'UNH',  'V',    'VZ',   'WBA',  'WMT',
    ],
    'mag7': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA'],
    'training': [
        'AAPL', 'AMGN', 'AXP',  'BA',   'CAT',
        'CRM',  'CSCO', 'CVX',  'DIS',  'DOW',
        'GS',   'HD',   'HON',  'IBM',  'JNJ',
        'JPM',  'KO',   'MCD',  'MMM',  'MRK',
        'MSFT', 'NKE',  'NVDA', 'PG',   'TRV',
        'UNH',  'V',    'VZ',   'WBA',  'WMT',
        'AMZN', 'COIN', 'GE',   'GOOGL','NFLX',
        'NIO',  'TSLA', 'UVV',
    ],
}

INTERVAL_MAP = {
    'day':   TimeFrame.Day,
    'week':  TimeFrame.Week,
    'month': TimeFrame.Month,
    'hour':  TimeFrame.Hour,
    'minute': TimeFrame.Minute,
}

# ── Data layer (same caching pattern as main.py) ───────────────────────────

_cache = {}

# Intraday intervals that need chunked fetching to avoid Alpaca rate limits
_INTRADAY = {str(TimeFrame.Minute), str(TimeFrame.Hour)}

def _get_bars(rest_api, symbol, timeframe, start, end):
    key = (symbol, str(timeframe), start, end)
    if key not in _cache:
        if str(timeframe) in _INTRADAY:
            _cache[key] = _get_bars_chunked(rest_api, symbol, timeframe, start, end)
        else:
            _cache[key] = rest_api.get_bars(symbol, timeframe, start, end, adjustment='all').df
    return _cache[key]

def _get_bars_chunked(rest_api, symbol, timeframe, start, end, chunk_months=3):
    """Fetch intraday bars in monthly chunks to avoid Alpaca pagination rate limits."""
    import pandas as pd
    chunks = []
    cur = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    while cur < end_date:
        # advance by chunk_months months
        month = cur.month - 1 + chunk_months
        next_cur = cur.replace(year=cur.year + month // 12, month=month % 12 + 1, day=1)
        chunk_end = min(next_cur - timedelta(days=1), end_date)
        for attempt in range(3):
            try:
                df = rest_api.get_bars(
                    symbol, timeframe,
                    cur.isoformat(), chunk_end.isoformat(),
                    adjustment='all'
                ).df
                chunks.append(df)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        cur = next_cur
    return pd.concat(chunks) if chunks else pd.DataFrame()

def _resolve_symbols(symbols):
    """Accept a list, a comma string, or a named set key."""
    if isinstance(symbols, (list, tuple)):
        return list(symbols)
    if isinstance(symbols, str):
        if symbols in SYMBOL_SETS:
            return SYMBOL_SETS[symbols]
        return [s.strip().upper() for s in symbols.split(',') if s.strip()]
    raise ValueError(f"symbols must be a list, comma string, or one of {list(SYMBOL_SETS)}")


# ── Core backtest (mirrors main.py's Backtrader.run_backtest) ──────────────

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
        dict with keys: total_return, sharpe, avg_annual_return, max_drawdown
    """
    rest_api  = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)
    timeframe = INTERVAL_MAP.get(interval, TimeFrame.Day)
    sym_list  = _resolve_symbols(symbols)

    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.setcash(cash)
    cerebro.addstrategy(strategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name='sharpe',
                        timeframe=bt.TimeFrame.Days, compression=1,
                        riskfreerate=0.05, annualize=True)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn,  _name='annual')
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name='drawdown')

    print(f"Loading {len(sym_list)} symbol(s)  {start} → {end}  [{interval}]")
    for i, sym in enumerate(sym_list, 1):
        df = _get_bars(rest_api, sym, timeframe, start, end)
        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=sym))
        print(f"  [{i:>2}/{len(sym_list)}] {sym} ✓")

    initial = cerebro.broker.getvalue()
    results = cerebro.run()
    final   = cerebro.broker.getvalue()

    strat          = results[0]
    total_return   = (final / initial - 1) * 100
    sharpe         = strat.analyzers.sharpe.get_analysis().get('sharperatio')
    annual_raw     = strat.analyzers.annual.get_analysis()
    avg_annual     = (sum(annual_raw.values()) / len(annual_raw) * 100) if annual_raw else 0.0
    max_drawdown   = strat.analyzers.drawdown.get_analysis()['max']['drawdown']

    print()
    print("=" * 50)
    print(f"  Total return  : {total_return:+.2f}%")
    print(f"  Avg annual    : {avg_annual:+.2f}%")
    print(f"  Sharpe ratio  : {f'{sharpe:.3f}' if sharpe is not None else 'N/A'}")
    print(f"  Max drawdown  : {max_drawdown:.2f}%")
    print("=" * 50)

    if plot:
        cerebro.plot(iplot=False)
        for i, fig_num in enumerate(plt.get_fignums(), start=1):
            fname = f'{plot_prefix}_{i}.png'
            plt.figure(fig_num)
            plt.savefig(fname, dpi=140, bbox_inches='tight')
            print(f"  Chart → {fname}")
        plt.close('all')

    return {
        'total_return':    total_return,
        'sharpe':          sharpe,
        'avg_annual':      avg_annual,
        'max_drawdown':    max_drawdown,
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
                   help='Strategy file name, e.g. dow30_momentum or strategy_1')
    p.add_argument('--symbols', default='dow30',
                   help='Comma-separated tickers or named set: dow30 | mag7 | training (default: dow30)')
    p.add_argument('--start',    default='2020-01-01', help='Start date YYYY-MM-DD')
    p.add_argument('--end',      default='2024-12-31', help='End date YYYY-MM-DD')
    p.add_argument('--interval', default='day',
                   choices=list(INTERVAL_MAP), help='Bar interval (default: day)')
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
