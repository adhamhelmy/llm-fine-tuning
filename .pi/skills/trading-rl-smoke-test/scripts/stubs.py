"""Offline stubs for trading_rl: synthetic Alpaca bars + a fake Ollama server.

    import stubs; stubs.install_fake_bars()           # every Backtester gets random-walk bars
    stubs.install_fake_ollama([text1, text2, ...])    # requests.post cycles through these replies

Import this AFTER the repo root is on sys.path (run_notebook.py / smoke.py handle that).
"""

import itertools
import zlib

import numpy as np
import pandas as pd

# Example model outputs, one per reward status (keys = expected status).
SAMPLE_OUTPUTS = {
    'profitable': (
        "Sure!\n```python\n"
        "class Strategy(bt.Strategy):\n"
        "    def __init__(self):\n"
        "        self.sma = bt.indicators.SMA(self.data.close, period=20)\n"
        "    def next(self):\n"
        "        if not self.position and self.data.close[0] > self.sma[0]:\n"
        "            self.order_target_percent(target=0.95)\n"
        "        elif self.position and self.data.close[0] < self.sma[0]:\n"
        "            self.close()\n"
        "```"
    ),
    'no_trades': "class Strategy(bt.Strategy):\n    def __init__(self):\n        pass\n    def next(self):\n        pass\n",
    'missing_methods': "class Strategy(bt.Strategy):\n    def __init__(self): pass\n",
    'invalid_code': "class Strategy(bt.Strategy):\n    def __init__(self):\n        import numpy\n    def next(self):\n        pass\n",
    'exception': "class Strategy(bt.Strategy):\n    def __init__(self):\n        pass\n    def next(self):\n        1/0\n",
    'timeout': "class Strategy(bt.Strategy):\n    def __init__(self):\n        pass\n    def next(self):\n        while True: pass\n",
}


def fake_frame(symbol, start, end, drift=0.0008, vol=0.02):
    """Deterministic (per symbol) business-day random walk in Alpaca's bar format."""
    idx = pd.date_range(start, end, freq='B', tz='UTC')
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))
    return pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99,
                         'close': close, 'volume': 1e6}, index=idx)


def install_fake_bars(**frame_kwargs):
    """Replace Backtester.get_bars so no Alpaca call is ever made (fills the shared cache)."""
    from trading_rl.backtest import Backtester

    def get_bars(self, symbol, start, end, timeframe=None):
        key = (symbol, str(timeframe), start, end)
        if key not in Backtester._cache:
            Backtester._cache[key] = fake_frame(symbol, start, end, **frame_kwargs)
        return Backtester._cache[key]

    Backtester.get_bars = get_bars


def install_fake_ollama(outputs=None, models=('llama3.2:3b',)):
    """Patch requests.get/post to mimic Ollama's /api/tags and /api/chat."""
    import requests
    replies = itertools.cycle(outputs or list(SAMPLE_OUTPUTS.values()))

    class _Resp:
        def __init__(self, payload): self._payload = payload
        def raise_for_status(self): pass
        def json(self): return self._payload

    requests.get = lambda *a, **k: _Resp({'models': [{'name': m} for m in models]})
    requests.post = lambda *a, **k: _Resp({'message': {'content': next(replies)}})
