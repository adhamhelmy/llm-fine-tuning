"""Backtrader + Alpaca backtesting with a shared bar cache and a time limit."""

import os
import signal
import threading
import time
from datetime import date, timedelta
from typing import NamedTuple, Optional

import backtrader as bt
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame

from .config import ALPACA_BASE_URL, SYMBOL_SETS

INTERVALS = {
    'day': TimeFrame.Day,
    'week': TimeFrame.Week,
    'month': TimeFrame.Month,
    'hour': TimeFrame.Hour,
    'minute': TimeFrame.Minute,
}
_INTRADAY = {str(TimeFrame.Minute), str(TimeFrame.Hour)}


class BacktestResult(NamedTuple):
    return_pct: float
    sharpe_ratio: Optional[float]
    avg_annual_return_pct: float
    max_drawdown_pct: float

    @property
    def traded(self) -> bool:
        # A strategy that never trades ends flat and backtrader reports no Sharpe.
        return not (self.return_pct == 0 and self.sharpe_ratio is None)

    def __str__(self):
        sharpe = f"{self.sharpe_ratio:.2f}" if self.sharpe_ratio is not None else "N/A"
        return (f"return={self.return_pct:.2f}% sharpe={sharpe} "
                f"annual={self.avg_annual_return_pct:.2f}% drawdown={self.max_drawdown_pct:.2f}%")


def resolve_symbols(symbols):
    """Accept a ticker, a list, a comma-separated string, or a named set ('dow30', 'mag7', 'training')."""
    if isinstance(symbols, (list, tuple, set)):
        return list(symbols)
    if isinstance(symbols, str):
        if symbols in SYMBOL_SETS:
            return list(SYMBOL_SETS[symbols])
        return [s.strip().upper() for s in symbols.split(',') if s.strip()]
    raise ValueError(f"symbols must be a list, comma string, or one of {list(SYMBOL_SETS)}")


def call_with_timeout(fn, seconds, *args, **kwargs):
    """Run fn with a time limit, raising TimeoutError.

    Uses SIGALRM when on the main thread (interrupts the backtest, same as unsloth's
    execute_with_time_limit); otherwise falls back to a worker thread (the caller gets the
    TimeoutError, but the runaway thread is not killed).
    """
    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        def _handler(signum, frame):
            raise TimeoutError(f"exceeded {seconds}s limit")
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    # Daemon thread so a runaway backtest can't block interpreter exit.
    out = {}

    def _target():
        try:
            out['value'] = fn(*args, **kwargs)
        except BaseException as e:
            out['error'] = e

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"exceeded {seconds}s limit")
    if 'error' in out:
        raise out['error']
    return out['value']


class Backtester:
    """Runs Backtrader strategies on Alpaca bars. The bar cache is shared by all instances."""

    _cache = {}

    def __init__(self, api_key=None, secret_key=None, base_url=ALPACA_BASE_URL,
                 sharpe_kwargs=None, verbose=True):
        """
        api_key / secret_key: default to the ALPACA_API_KEY / ALPACA_SECRET_KEY env vars.
        sharpe_kwargs: extra args for bt.analyzers.SharpeRatio (default: backtrader defaults,
                       which is what the GRPO reward was trained with).
        """
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY', '')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY', '')
        self.base_url = base_url
        self.sharpe_kwargs = sharpe_kwargs or {}
        self.verbose = verbose
        self._rest = None

    @property
    def rest(self):
        if self._rest is None:
            self._rest = REST(self.api_key, self.secret_key, self.base_url)
        return self._rest

    # ── data ──────────────────────────────────────────────────────────────────

    def get_bars(self, symbol, start, end, timeframe=TimeFrame.Day):
        timeframe = INTERVALS.get(timeframe, timeframe)
        key = (symbol, str(timeframe), start, end)
        if key not in Backtester._cache:
            if str(timeframe) in _INTRADAY:
                Backtester._cache[key] = self._get_bars_chunked(symbol, timeframe, start, end)
            else:
                Backtester._cache[key] = self.rest.get_bars(
                    symbol, timeframe, start, end, adjustment='all').df
        return Backtester._cache[key]

    def _get_bars_chunked(self, symbol, timeframe, start, end, chunk_months=3):
        """Fetch intraday bars in chunks to avoid Alpaca pagination rate limits."""
        chunks = []
        cur, end_date = date.fromisoformat(start), date.fromisoformat(end)
        while cur < end_date:
            month = cur.month - 1 + chunk_months
            nxt = cur.replace(year=cur.year + month // 12, month=month % 12 + 1, day=1)
            chunk_end = min(nxt - timedelta(days=1), end_date)
            for attempt in range(3):
                try:
                    chunks.append(self.rest.get_bars(
                        symbol, timeframe, cur.isoformat(), chunk_end.isoformat(),
                        adjustment='all').df)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5)
            cur = nxt
        return pd.concat(chunks) if chunks else pd.DataFrame()

    def load_bars(self, symbols, start, end, timeframe=TimeFrame.Day):
        """Pre-fetch and cache bars (call once before training/evaluation)."""
        symbols = resolve_symbols(symbols)
        print(f"Pre-fetching bar data for {len(symbols)} symbols ({start} → {end})...")
        for i, symbol in enumerate(symbols, 1):
            self.get_bars(symbol, start, end, timeframe)
            print(f"  [{i}/{len(symbols)}] {symbol} ✓")
        print(f"Done. {len(Backtester._cache)} entries cached.")

    # ── backtesting ───────────────────────────────────────────────────────────

    def run(self, strategy, symbols, start, end, timeframe=TimeFrame.Day, cash=10_000,
            plot=False, plot_prefix='backtest_plot', show_plots=True) -> BacktestResult:
        """Backtest a bt.Strategy subclass. With plot=True, charts are saved to
        f"{plot_prefix}_{i}.png" and their paths stored in self.last_plots."""
        cerebro = bt.Cerebro(stdstats=True)
        cerebro.broker.setcash(cash)
        cerebro.addstrategy(strategy)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', **self.sharpe_kwargs)
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        for sym in resolve_symbols(symbols):
            cerebro.adddata(bt.feeds.PandasData(
                dataname=self.get_bars(sym, start, end, timeframe), name=sym))

        initial = cerebro.broker.getvalue()
        strat = cerebro.run()[0]
        annual = strat.analyzers.annual.get_analysis()
        result = BacktestResult(
            return_pct=(cerebro.broker.getvalue() / initial - 1) * 100,
            sharpe_ratio=strat.analyzers.sharpe.get_analysis().get('sharperatio'),
            avg_annual_return_pct=(sum(annual.values()) / len(annual) * 100) if annual else 0.0,
            max_drawdown_pct=strat.analyzers.drawdown.get_analysis()['max']['drawdown'],
        )
        if self.verbose:
            print(result)

        self.last_plots = []
        if plot:
            self.last_plots = self._save_plots(strat, plot_prefix, show_plots)
        return result

    def run_with_timeout(self, strategy, symbols, start, end, timeframe=TimeFrame.Day,
                         cash=10_000, timeout=10) -> BacktestResult:
        """Same as run() (without plotting) but raises TimeoutError after `timeout` seconds."""
        return call_with_timeout(self.run, timeout, strategy, symbols, start, end,
                                 timeframe=timeframe, cash=cash)

    @staticmethod
    def _save_plots(strat, prefix, show):
        # Use backtrader's Plot directly: cerebro.plot() calls plt.show(), which blocks in
        # scripts and closes the figures in inline notebooks before they can be saved.
        import matplotlib.pyplot as plt
        from backtrader.plot import Plot
        paths = []
        for i, fig in enumerate(Plot().plot(strat, iplot=False), start=1):
            path = f'{prefix}_{i}.png'
            fig.savefig(path, dpi=140, bbox_inches='tight')
            plt.close(fig)
            paths.append(path)
        if show:
            try:
                from IPython import get_ipython
                from IPython.display import Image, display
                if get_ipython() is not None:
                    for path in paths:
                        display(Image(path))
            except ImportError:
                pass
        return paths
