"""Backtrader + Alpaca backtesting with a shared (memory + disk) bar cache and a time limit."""

import os
import signal
import threading
import time
from datetime import date, timedelta
from typing import NamedTuple, Optional

import backtrader as bt
import pandas as pd

from .config import SYMBOL_ISSUES, SYMBOL_SETS

# Bar sizes, by name. Legacy alpaca TimeFrame objects are accepted too (see timeframe_name).
INTERVALS = ('day', 'week', 'month', 'hour', 'minute')
_INTRADAY = {'hour', 'minute'}
_ALPACA_NAMES = {'1Day': 'day', '1Week': 'week', '1Month': 'month', '1Hour': 'hour', '1Min': 'minute'}


def timeframe_name(timeframe) -> str:
    """'day' / 'hour' / ... from a name or an alpaca(-py / -trade-api) TimeFrame object."""
    name = timeframe if isinstance(timeframe, str) else _ALPACA_NAMES.get(str(timeframe))
    if name not in INTERVALS:
        raise ValueError(f"unknown timeframe {timeframe!r}; use one of {INTERVALS}")
    return name


class BacktestResult(NamedTuple):
    return_pct: float
    sharpe_ratio: Optional[float]
    avg_annual_return_pct: float      # mean of calendar-year returns (the original reward)
    max_drawdown_pct: float
    cagr_pct: float = 0.0             # compound annual growth rate over the bar span
    n_orders: int = 0                 # completed (filled) orders
    # Jensen's alpha: OLS of per-bar strategy returns on the traded asset's per-bar returns
    # (buy-and-hold of the same data, no costs; risk-free rate 0). See jensen_alpha().
    alpha_pct: float = 0.0            # annualised intercept, % per year
    beta: float = 0.0                 # market exposure vs buy-and-hold
    alpha_t: float = 0.0              # t-statistic of the intercept

    @property
    def traded(self) -> bool:
        return self.n_orders > 0

    def __str__(self):
        sharpe = f"{self.sharpe_ratio:.2f}" if self.sharpe_ratio is not None else "N/A"
        return (f"return={self.return_pct:.2f}% cagr={self.cagr_pct:.2f}% sharpe={sharpe} "
                f"annual={self.avg_annual_return_pct:.2f}% drawdown={self.max_drawdown_pct:.2f}% "
                f"alpha={self.alpha_pct:.2f}% beta={self.beta:.2f} orders={self.n_orders}")


def jensen_alpha(strategy_values, benchmark_prices, periods_per_year=252.0, min_periods=20):
    """Jensen's alpha of an equity curve against a benchmark price series (same bars).

    Regresses per-bar strategy returns on per-bar benchmark returns (risk-free rate 0):
        r_p = a + b * r_m + e
    Returns (alpha_pct, beta, alpha_t): `a` annualised (x periods_per_year, in %), `b`, and the
    t-statistic of `a`. Fewer than `min_periods` returns gives (0.0, 0.0, 0.0); an equity curve
    that reaches <= 0 gives (-100.0, 0.0, 0.0).
    """
    import numpy as np
    v = np.asarray(strategy_values, dtype=float)
    m = np.asarray(benchmark_prices, dtype=float)
    if len(v) and (v <= 0).any():
        return -100.0, 0.0, 0.0     # account wiped out (e.g. leveraged short): worst alpha, not 0
    if len(v) != len(m) or len(v) <= min_periods or (m[:-1] <= 0).any():
        return 0.0, 0.0, 0.0
    y, x = v[1:] / v[:-1] - 1, m[1:] / m[:-1] - 1
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < min_periods:
        return 0.0, 0.0, 0.0
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    beta = float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else 0.0
    a = ym - beta * xm
    resid = y - a - beta * x
    s2 = (resid ** 2).sum() / (n - 2)
    se = np.sqrt(s2 * (1 / n + (xm ** 2 / sxx if sxx > 0 else 0.0)))
    t = float(a / se) if se > 1e-12 else 0.0
    return float(a * periods_per_year * 100), beta, t


def resolve_symbols(symbols):
    """Accept a ticker, a list, a comma-separated string, or a named set ('eval', 'dow30',
    'mag7', 'training'). Returns upper-case tickers, de-duplicated, in the given order."""
    if isinstance(symbols, str):
        if symbols in SYMBOL_SETS:
            symbols = SYMBOL_SETS[symbols]
        else:
            symbols = symbols.split(',')
    elif not isinstance(symbols, (list, tuple, set)):
        raise ValueError(f"symbols must be a list, comma string, or one of {list(SYMBOL_SETS)}")
    return list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))


def call_with_timeout(fn, seconds, *args, **kwargs):
    """Run fn with a time limit, raising TimeoutError.

    Uses SIGALRM when on the main thread; otherwise falls back to a worker thread (the caller
    gets the TimeoutError, but the runaway thread is not killed). For a hard limit that also
    stops C-level loops, score through trading_rl.scoring.ScoringPool (subprocesses).
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


def bar_issues(bars: pd.DataFrame, max_flat_days=20, start=None, max_late_days=7):
    """Data-quality problems of a bar frame: empty, starting more than `max_late_days` after
    `start` (listed later), or a long run of flat zero-volume bars (Alpaca forward-fills
    inactive tickers). Returns a list of human-readable strings."""
    if bars is None or len(bars) == 0:
        return ['no bars']
    issues = []
    if start is not None:
        first = bars.index[0].date()
        if (first - date.fromisoformat(start)).days > max_late_days:
            issues.append(f'first bar {first} (window starts {start})')
    flat = (bars['volume'] == 0) & (bars['close'].diff() == 0)
    longest = int(flat.groupby((~flat).cumsum()).sum().max()) if flat.any() else 0
    if longest >= max_flat_days:
        issues.append(f'{longest} consecutive flat zero-volume bars')
    return issues


class Backtester:
    """Runs Backtrader strategies on Alpaca bars. The in-memory bar cache is shared by all
    instances; bars are also cached on disk (cache_dir) so later runs never hit Alpaca."""

    _cache = {}
    # Disk cache location; set TRADING_RL_CACHE='' (or cache_dir=None) to disable.
    DEFAULT_CACHE_DIR = os.environ.get('TRADING_RL_CACHE',
                                       os.path.join(os.path.expanduser('~'), '.cache', 'trading_rl', 'bars'))

    def __init__(self, api_key=None, secret_key=None, sharpe_kwargs=None, verbose=True,
                 cache_dir='default', feed=None):
        """
        api_key / secret_key: default to the ALPACA_API_KEY / ALPACA_SECRET_KEY env vars.
        sharpe_kwargs: extra args for bt.analyzers.SharpeRatio (default: backtrader defaults,
                       which is what the GRPO reward was trained with).
        cache_dir:     directory for pickled bars ('default': DEFAULT_CACHE_DIR; None: memory only).
        feed:          alpaca data feed ('sip' / 'iex'); None = account default.
        """
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY', '')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY', '')
        self.sharpe_kwargs = sharpe_kwargs or {}
        self.verbose = verbose
        self.cache_dir = (Backtester.DEFAULT_CACHE_DIR if cache_dir == 'default' else cache_dir) or None
        self.feed = feed
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not (self.api_key and self.secret_key):
                raise RuntimeError("Alpaca keys missing: pass api_key/secret_key or set ALPACA_API_KEY / "
                                   "ALPACA_SECRET_KEY (bars already in the cache need no keys).")
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._client

    # ── data ──────────────────────────────────────────────────────────────────

    @staticmethod
    def cache_key(symbol, start, end, timeframe='day'):
        return (symbol, timeframe_name(timeframe), start, end)

    def get_bars(self, symbol, start, end, timeframe='day'):
        key = self.cache_key(symbol, start, end, timeframe)
        if key not in Backtester._cache:
            path = os.path.join(self.cache_dir, '_'.join(key) + '.pkl') if self.cache_dir else None
            if path and os.path.exists(path):
                bars = pd.read_pickle(path)
            elif (sup := self._from_superset(*key)) is not None:
                bars = sup          # sub-window of bars already cached: same prices, no download
            else:
                bars = self._fetch_bars(symbol, start, end, key[1])
                if path:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    bars.to_pickle(path + '.tmp')
                    os.replace(path + '.tmp', path)
            Backtester._cache[key] = bars
        return Backtester._cache[key]

    def _from_superset(self, symbol, timeframe, start, end):
        """Bars for [start, end] cut from a cached window that covers it (memory or disk)."""
        keys = [k for k in Backtester._cache if k[:2] == (symbol, timeframe)]
        if self.cache_dir and os.path.isdir(self.cache_dir):
            prefix = f"{symbol}_{timeframe}_"
            for f in os.listdir(self.cache_dir):
                parts = f[len(prefix):-4].split('_') if f.startswith(prefix) and f.endswith('.pkl') else []
                if len(parts) == 2:
                    keys.append((symbol, timeframe, *parts))
        for k in sorted(set(keys), key=lambda k: (k[2], k[3])):
            if k[2] <= start and k[3] >= end and (k[2], k[3]) != (start, end):
                sup = Backtester._cache.get(k)
                if sup is None:
                    sup = pd.read_pickle(os.path.join(self.cache_dir, '_'.join(k) + '.pkl'))
                days = sup.index.normalize().tz_localize(None) if sup.index.tz is not None else sup.index.normalize()
                sub = sup[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
                return sub if len(sub) else None
        return None

    def _fetch_bars(self, symbol, start, end, timeframe):
        """Download split/dividend-adjusted bars from Alpaca (intraday in 3-month chunks)."""
        if timeframe not in _INTRADAY:
            return self._fetch_range(symbol, start, end, timeframe)
        chunks = []
        cur, end_date = date.fromisoformat(start), date.fromisoformat(end)
        while cur <= end_date:
            month = cur.month - 1 + 3
            nxt = cur.replace(year=cur.year + month // 12, month=month % 12 + 1, day=1)
            chunk_end = min(nxt - timedelta(days=1), end_date)
            chunks.append(self._fetch_range(symbol, cur.isoformat(), chunk_end.isoformat(), timeframe))
            cur = nxt
        chunks = [c for c in chunks if len(c)]
        return pd.concat(chunks) if chunks else pd.DataFrame()

    def _fetch_range(self, symbol, start, end, timeframe, attempts=3):
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        tf = {'day': TimeFrame.Day, 'week': TimeFrame.Week, 'month': TimeFrame.Month,
              'hour': TimeFrame.Hour, 'minute': TimeFrame.Minute}[timeframe]
        # Alpaca's `end` is exclusive at midnight; include the whole end day.
        end_excl = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end_excl,
                               adjustment=Adjustment.ALL, **({'feed': self.feed} if self.feed else {}))
        for attempt in range(attempts):
            try:
                df = self.client.get_stock_bars(req).df
                break
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(5)
        if len(df) and isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')
        return df

    def load_bars(self, symbols, start, end, timeframe='day'):
        """Pre-fetch and cache bars (call once before training/evaluation). Warns about
        symbols with missing or forward-filled data (see config.SYMBOL_ISSUES)."""
        symbols = resolve_symbols(symbols)
        print(f"Pre-fetching bar data for {len(symbols)} symbols ({start} → {end})...")
        for i, symbol in enumerate(symbols, 1):
            bars = self.get_bars(symbol, start, end, timeframe)
            warn = bar_issues(bars, start=start)
            known = SYMBOL_ISSUES.get(symbol)
            if known and known[0] > start:
                warn.append(f'unreliable before {known[0]}: {known[1]}')
            print(f"  [{i}/{len(symbols)}] {symbol} " + (f"⚠ {'; '.join(warn)}" if warn else "✓"))
        print(f"Done. {len(Backtester._cache)} entries cached.")

    # ── backtesting ───────────────────────────────────────────────────────────

    def _cerebro(self, strategy, symbols, start, end, timeframe, cash,
                 commission=0.0, slippage=0.0, detailed=False, plot=False):
        # The Trades/BuySell observers only matter for plots; skipping them speeds up every
        # backtest. AnnualReturn reads the Broker observer, so that one is always added.
        # preload=False: bars are loaded one at a time, so a strategy indexing the future
        # (self.data.close[1]) gets an IndexError instead of tomorrow's price (lookahead).
        # Same results for honest strategies, ~1.5x slower.
        cerebro = bt.Cerebro(stdstats=plot, preload=False)
        if not plot:
            cerebro.addobserver(bt.observers.Broker)
        cerebro.broker.setcash(cash)
        if commission:
            cerebro.broker.setcommission(commission=commission)   # fraction of traded value
        if slippage:
            # Fills move against the order by `slippage` (fraction of price); slip_out lets
            # fills go outside the bar's high/low so the cost is always paid.
            cerebro.broker.set_slippage_perc(slippage, slip_open=True, slip_limit=True,
                                             slip_match=True, slip_out=True)
        cerebro.addstrategy(strategy)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', **self.sharpe_kwargs)
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(_Activity, _name='activity')
        if detailed:
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='daily', timeframe=bt.TimeFrame.Days)
        for sym in resolve_symbols(symbols):
            cerebro.adddata(bt.feeds.PandasData(
                dataname=self.get_bars(sym, start, end, timeframe), name=sym))
        return cerebro

    @staticmethod
    def _result(cerebro, strat, initial) -> BacktestResult:
        annual = strat.analyzers.annual.get_analysis()
        act = strat.analyzers.activity.get_analysis()
        final = cerebro.broker.getvalue()
        growth = final / initial
        cagr = (growth ** (1 / act['years']) - 1) * 100 if growth > 0 else -100.0
        n_bars = len(act['values'])
        alpha, beta, alpha_t = jensen_alpha(act['values'], act['prices'],
                                            periods_per_year=max(n_bars - 1, 1) / act['years'])
        return BacktestResult(
            return_pct=(growth - 1) * 100,
            sharpe_ratio=strat.analyzers.sharpe.get_analysis().get('sharperatio'),
            avg_annual_return_pct=(sum(annual.values()) / len(annual) * 100) if annual else 0.0,
            max_drawdown_pct=strat.analyzers.drawdown.get_analysis()['max']['drawdown'],
            cagr_pct=cagr,
            n_orders=act['orders'],
            alpha_pct=alpha,
            beta=beta,
            alpha_t=alpha_t,
        )

    def run(self, strategy, symbols, start, end, timeframe='day', cash=10_000,
            commission=0.0, slippage=0.0,
            plot=False, plot_prefix='backtest_plot', show_plots=True) -> BacktestResult:
        """Backtest a bt.Strategy subclass. commission / slippage are fractions per side
        (0.0001 = 1 bp); the reward passes its RewardConfig costs here. With plot=True, charts
        are saved to f"{plot_prefix}_{i}.png" and their paths stored in self.last_plots."""
        cerebro = self._cerebro(strategy, symbols, start, end, timeframe, cash,
                                commission=commission, slippage=slippage, plot=plot)
        initial = cerebro.broker.getvalue()
        strat = cerebro.run()[0]
        result = self._result(cerebro, strat, initial)
        if self.verbose:
            print(result)

        self.last_plots = []
        if plot:
            self.last_plots = self._save_plots(strat, plot_prefix, show_plots)
        return result

    def run_detailed(self, strategy, symbols, start, end, timeframe='day', cash=10_000,
                     commission=0.0, slippage=0.0, periods_per_year=252):
        """Backtest with transaction costs and return (BacktestResult, extra metrics dict).

        commission / slippage: fractions per side (0.0005 = 5 bps).
        extras: n_trades (closed + open), turnover (annualised traded value / avg equity),
        volatility_pct (annualised stdev of per-bar returns), commission_paid,
        rejected_orders (margin/rejected: costs can make all-in orders unaffordable) and
        year_<YYYY> (calendar-year return in %).
        """
        cerebro = self._cerebro(strategy, symbols, start, end, timeframe, cash,
                                commission=commission, slippage=slippage, detailed=True)
        initial = cerebro.broker.getvalue()
        strat = cerebro.run()[0]
        result = self._result(cerebro, strat, initial)

        tot = strat.analyzers.trades.get_analysis().get('total', {})
        rets = pd.Series(list(strat.analyzers.daily.get_analysis().values()), dtype=float)
        act = strat.analyzers.activity.get_analysis()
        extras = {
            'n_trades': int(tot.get('closed', 0)) + int(tot.get('open', 0)),
            'turnover': act['turnover'],
            'volatility_pct': float(rets.std() * periods_per_year ** 0.5 * 100) if len(rets) > 1 else 0.0,
            'commission_paid': act['commission'],
            'rejected_orders': act['rejected'],
        }
        for year, r in strat.analyzers.annual.get_analysis().items():
            extras[f'year_{year}'] = r * 100
        return result, extras

    def run_with_timeout(self, strategy, symbols, start, end, timeframe='day',
                         cash=10_000, timeout=10, commission=0.0, slippage=0.0) -> BacktestResult:
        """Same as run() (without plotting) but raises TimeoutError after `timeout` seconds."""
        return call_with_timeout(self.run, timeout, strategy, symbols, start, end,
                                 timeframe=timeframe, cash=cash,
                                 commission=commission, slippage=slippage)

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


class _Activity(bt.Analyzer):
    """Filled orders, traded value, commission, rejected orders, the calendar span in years
    (turnover and CAGR are annualised by calendar time), and per-bar equity + first data feed
    close (for Jensen's alpha)."""

    def start(self):
        self.traded = self.comm = 0.0
        self.rejected = self.orders = 0
        self.values, self.prices, self.first, self.last = [], [], None, None

    def notify_order(self, order):
        if order.status == order.Completed:
            self.traded += abs(order.executed.size * order.executed.price)
            self.comm += order.executed.comm
            self.orders += 1
        elif order.status in (order.Margin, order.Rejected):
            self.rejected += 1

    def next(self):
        self.values.append(self.strategy.broker.getvalue())
        self.prices.append(self.strategy.datas[0].close[0])
        d = self.strategy.datetime.date(0)
        self.first = self.first or d
        self.last = d

    def get_analysis(self):
        years = max((self.last - self.first).days / 365.25, 1 / 365.25) if self.first else 1.0
        avg = sum(self.values) / len(self.values) if self.values else 0.0
        return {'turnover': (self.traded / avg / years) if avg else 0.0,
                'commission': self.comm, 'rejected': self.rejected, 'orders': self.orders,
                'years': years, 'values': self.values, 'prices': self.prices}
