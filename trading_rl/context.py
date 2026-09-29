"""Context blocks for the strategy prompt (the *context engineering* stage).

    api_reference()        Backtrader API cheat sheet (indicator signatures from the installed
                           backtrader, so names/params/lines are always correct) + idioms/pitfalls
    market_stats(...)      statistics of the stock computed ONLY from bars before the window
    format_market_summary  text block for those stats
    few_shot_examples(k)   k validated example strategies

Retrieval (financial RAG) lives in trading_rl.knowledge.
"""

from datetime import date, timedelta

import backtrader as bt
import numpy as np

# Indicators LLMs reach for most often. Signatures are read from backtrader at runtime.
_COMMON_INDICATORS = [
    'SMA', 'EMA', 'WMA', 'RSI', 'MACD', 'BollingerBands', 'ATR', 'Stochastic', 'CCI',
    'ADX', 'CrossOver', 'Highest', 'Lowest', 'Momentum', 'ROC', 'StdDev', 'WilliamsR',
    'ParabolicSAR',
]
_TWO_INPUTS = {'CrossOver', 'CrossUp', 'CrossDown'}

_IDIOMS = """\
DATA ACCESS
- self.data is the single data feed. Fields: self.data.open/high/low/close/volume.
- Index lines with [0] (current bar), [-1] (previous bar). Never use [1] (future) or Python slicing.
- Compare values, not line objects, in next():  `if self.data.close[0] > self.sma[0]:`
- Crossovers: create `self.cross = bt.indicators.CrossOver(fast, slow)` in __init__, then
  `self.cross[0] > 0` (up) / `self.cross[0] < 0` (down) in next().
- Parameters: `params = dict(fast=20, slow=50)` then `self.p.fast` (or self.params.fast).

ORDERS & POSITION
- self.position.size (0 when flat); `if not self.position:` checks for no position.
- self.order_target_percent(target=0.95) buys to 95% of equity; target=0.0 exits.
- self.buy(size=n) / self.sell(size=n) / self.close(). Don't compute sizes from cash
  unless you guard against size 0.
- Orders fill on the next bar; track a pending order (`self.order`) if you place several per bar.

COMMON RUNTIME ERRORS TO AVOID
- No imports (bt is already available; numpy/pandas/talib are NOT).
- bt.talib.* needs TA-Lib and is NOT available; use bt.indicators.*.
- Indicators must be created in __init__, never in next().
- Warm-up is automatic: next() starts once all indicators have enough bars. Do not index further
  back than your longest indicator period.
- Don't name attributes after Strategy methods (self.stop, self.start, self.next, self.prenext,
  self.buy, self.sell, self.close, self.position, self.notify_order): it breaks the backtest.
  Use names like self.stop_price.
- Division by zero: guard denominators (e.g. ATR or StdDev can be 0).
- MACD lines are .macd and .signal; BollingerBands lines are .mid, .top, .bot;
  Stochastic lines are .percK and .percD. Use the exact line names listed above.
- Never call self.data.close.get(...) / len(self.data) inside __init__ to make decisions."""


def _indicator_signature(name):
    cls = getattr(bt.indicators, name)  # AttributeError = typo in _COMMON_INDICATORS
    params = ', '.join(f"{k}={v!r}" for k, v in cls.params._getitems()
                       if not k.startswith('_') and not isinstance(v, type))
    lines = ', '.join(f".{l}" for l in cls.lines.getlinealiases())
    data = 'data0, data1' if name in _TWO_INPUTS else 'data'
    return f"- bt.indicators.{name}({data}{', ' + params if params else ''})  -> lines: {lines}"


def api_reference() -> str:
    """Backtrader cheat sheet built from the installed backtrader version."""
    sigs = '\n'.join(map(_indicator_signature, _COMMON_INDICATORS))
    return (f"BACKTRADER API REFERENCE (backtrader {bt.__version__}):\n"
            f"INDICATORS (exact names, default params, output lines)\n{sigs}\n\n{_IDIOMS}")


# ── market summary (no lookahead: bars strictly before `start`) ────────────────

def market_stats(backtester, symbol, start, lookback_years=2):
    """Descriptive statistics of `symbol` over the `lookback_years` before `start`."""
    end_d = date.fromisoformat(start) - timedelta(days=1)
    start_d = end_d.replace(year=end_d.year - lookback_years) + timedelta(days=1)
    bars = backtester.get_bars(symbol, start_d.isoformat(), end_d.isoformat())
    close = bars['close'].astype(float)
    if len(close) < 60:
        return None
    ret = close.pct_change().dropna()
    years = len(close) / 252
    sma200 = close.rolling(200).mean()
    sma50 = close.rolling(50).mean()
    tr = np.maximum(bars['high'] - bars['low'],
                    np.maximum(abs(bars['high'] - close.shift()), abs(bars['low'] - close.shift())))
    stats = {
        'window': f"{start_d} to {end_d}",
        'annual_return_pct': ((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100,
        'annual_volatility_pct': ret.std() * np.sqrt(252) * 100,
        'max_drawdown_pct': ((close / close.cummax()) - 1).min() * -100,
        'pct_days_above_sma200': (close > sma200)[sma200.notna()].mean() * 100 if sma200.notna().any() else None,
        'last_close_vs_sma200_pct': (close.iloc[-1] / sma200.iloc[-1] - 1) * 100 if sma200.notna().any() else None,
        'sma50_above_sma200': bool(sma50.iloc[-1] > sma200.iloc[-1]) if sma200.notna().any() else None,
        'return_autocorr_lag1': ret.autocorr(1),
        'avg_true_range_pct': (tr / close).mean() * 100,
    }
    stats['regime'] = _regime_tags(stats)
    return stats


def _regime_tags(s):
    tags = []
    if s['annual_return_pct'] > 10 and (s['last_close_vs_sma200_pct'] or 0) > 0:
        tags.append('uptrend')
    elif s['annual_return_pct'] < -5:
        tags.append('downtrend')
    else:
        tags.append('range-bound')
    vol = s['annual_volatility_pct']
    tags.append('high volatility' if vol > 35 else 'low volatility' if vol < 20 else 'moderate volatility')
    if s['return_autocorr_lag1'] < -0.05:
        tags.append('mean-reverting')
    elif s['return_autocorr_lag1'] > 0.05:
        tags.append('momentum')
    return tags


def format_market_summary(stats, symbol=None, show_window=True) -> str:
    if not stats:
        return ""
    def f(v, suffix='%'):
        return 'n/a' if v is None else f"{v:.1f}{suffix}"
    name = symbol or "the stock"
    window = f", {stats['window']}" if show_window else ""
    return (f"MARKET CONTEXT for {name} (computed only from data BEFORE the backtest window{window}):\n"
            f"- Annualized return: {f(stats['annual_return_pct'])}, annualized volatility: "
            f"{f(stats['annual_volatility_pct'])}, max drawdown: {f(stats['max_drawdown_pct'])}\n"
            f"- Days above 200-day SMA: {f(stats['pct_days_above_sma200'])}; last close vs 200-day SMA: "
            f"{f(stats['last_close_vs_sma200_pct'])}; 50-day SMA above 200-day SMA: {stats['sma50_above_sma200']}\n"
            f"- Lag-1 autocorrelation of daily returns: {stats['return_autocorr_lag1']:.3f}; "
            f"average true range: {f(stats['avg_true_range_pct'])} of price\n"
            f"- Regime: {', '.join(stats['regime'])}\n"
            "The future may differ from this history; prefer robust rules over curve fitting.")


# ── few-shot examples (validated by the smoke test) ────────────────────────────

EXAMPLES = [
    '''class Strategy(bt.Strategy):
    params = dict(fast=20, slow=100)

    def __init__(self):
        fast = bt.indicators.EMA(self.data.close, period=self.p.fast)
        slow = bt.indicators.EMA(self.data.close, period=self.p.slow)
        self.cross = bt.indicators.CrossOver(fast, slow)

    def next(self):
        if not self.position and self.cross[0] > 0:
            self.order_target_percent(target=0.95)
        elif self.position and self.cross[0] < 0:
            self.close()''',
    '''class Strategy(bt.Strategy):
    params = dict(rsi_period=14, oversold=30, overbought=70, trend=200)

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.trend = bt.indicators.SMA(self.data.close, period=self.p.trend)

    def next(self):
        if not self.position:
            if self.rsi[0] < self.p.oversold and self.data.close[0] > self.trend[0]:
                self.order_target_percent(target=0.95)
        elif self.rsi[0] > self.p.overbought:
            self.close()''',
    '''class Strategy(bt.Strategy):
    params = dict(entry=55, exit=20, atr_period=14, stop_atr=3.0)

    def __init__(self):
        self.upper = bt.indicators.Highest(self.data.high, period=self.p.entry)
        self.lower = bt.indicators.Lowest(self.data.low, period=self.p.exit)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.stop_price = None

    def next(self):
        if not self.position:
            if self.data.close[0] > self.upper[-1]:
                self.order_target_percent(target=0.95)
                self.stop_price = self.data.close[0] - self.p.stop_atr * self.atr[0]
        elif self.data.close[0] < self.lower[-1] or self.data.close[0] < self.stop_price:
            self.close()''',
]


def few_shot_examples(k=2) -> str:
    if k <= 0:
        return ""
    body = '\n\n'.join(f"Example {i + 1}:\n{code}" for i, code in enumerate(EXAMPLES[:k]))
    return ("EXAMPLES of valid strategies (they show the format and correct API usage; do not copy "
            f"them, design a strategy suited to this stock):\n\n{body}")
