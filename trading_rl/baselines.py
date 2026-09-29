"""Non-LLM reference strategies, run through the same backtester and scoring as generated code."""

import pandas as pd

from .backtest import Backtester, resolve_symbols
from .rewards import score_strategy

BASELINES = {
    'buy_and_hold': '''class Strategy(bt.Strategy):
    def __init__(self):
        pass

    def next(self):
        if not self.position:
            self.order_target_percent(target=0.95)''',
    'sma_50_200': '''class Strategy(bt.Strategy):
    def __init__(self):
        self.cross = bt.indicators.CrossOver(bt.indicators.SMA(self.data.close, period=50),
                                             bt.indicators.SMA(self.data.close, period=200))

    def next(self):
        if not self.position and self.cross[0] > 0:
            self.order_target_percent(target=0.95)
        elif self.position and self.cross[0] < 0:
            self.close()''',
}


def baseline_table(backtester: Backtester, symbols, start, end, cash=10_000, timeout=30):
    """One row per (baseline, symbol) with the same columns as evaluation records."""
    rows = []
    for name, code in BASELINES.items():
        for sym in resolve_symbols(symbols):
            s = score_strategy(code, backtester, sym, start, end, cash=cash, timeout=timeout)
            rows.append({'config': name, 'symbol': sym, 'start': start, 'end': end,
                         'status': s.status, 'reward_score': s.reward, 'error': s.error,
                         **(s.result._asdict() if s.result else {})})
    return pd.DataFrame(rows)
