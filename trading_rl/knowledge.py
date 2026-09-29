"""Financial RAG: a small strategy knowledge base + BM25 retrieval (no extra dependencies).

    kb = KnowledgeBase.default()                  # curated corpus below
    kb = KnowledgeBase.from_dir('my_corpus/')     # your own .md/.txt files, split on blank lines
    kb.retrieve('uptrend high volatility', k=3)   # -> [Doc, ...]
    kb.context_block(query, k=3)                  # -> prompt section

The default corpus is deliberately generic (textbook strategy families and when they tend to
work). It contains no dated market facts, so retrieval cannot leak information about the
backtest window.
"""

import math
import os
import re
from collections import Counter
from typing import NamedTuple


class Doc(NamedTuple):
    title: str
    text: str
    tags: str = ""


DEFAULT_DOCS = [
    Doc("Moving-average trend following",
        "Go long when a fast moving average crosses above a slow one (e.g. EMA 20/100 or SMA 50/200) "
        "and exit on the opposite cross. Works in persistent trends; loses money through whipsaws in "
        "range-bound markets. Longer windows mean fewer trades and lower turnover. Adding a filter such "
        "as ADX > 20 reduces whipsaws.",
        "trend uptrend downtrend momentum moving average crossover sma ema"),
    Doc("Time-series momentum",
        "Hold the stock when its return over the past 3-12 months is positive and stay flat otherwise "
        "(bt.indicators.ROC or Momentum with period 63-252). Momentum is one of the most persistent "
        "documented anomalies, but it suffers sharp crashes at trend reversals. Rebalancing monthly "
        "rather than daily reduces trading noise.",
        "momentum trend uptrend roc rate of change lookback"),
    Doc("RSI mean reversion with trend filter",
        "Buy when RSI(2-14) falls below 30 (oversold) while price stays above its 200-day SMA, and "
        "sell when RSI rises above 50-70. The trend filter avoids buying into collapses. Works best on "
        "large, liquid stocks with negative short-term return autocorrelation.",
        "mean-reverting mean reversion rsi oversold range-bound pullback"),
    Doc("Bollinger Band reversion",
        "Buy when the close drops below the lower band (20-day, 2 standard deviations) and exit at "
        "the middle band. Suits range-bound, mean-reverting stocks. In strong trends price can ride "
        "the band, so add a stop-loss or a trend filter.",
        "mean-reverting range-bound bollinger bands volatility reversion"),
    Doc("Donchian channel breakout",
        "Enter when price closes above the highest high of the last N days (20-55) and exit below "
        "the lowest low of the last M days (10-20), as in the Turtle rules. Captures large trends; "
        "win rate is low (30-40%) but winners are large. Suits trending, volatile stocks.",
        "breakout trend uptrend high volatility donchian highest lowest channel"),
    Doc("Volatility-scaled position sizing",
        "Size positions inversely to recent volatility, e.g. target = min(0.95, 0.15 / annualized "
        "volatility), where volatility comes from ATR or StdDev of returns. This keeps risk roughly "
        "constant and reduces drawdowns in high-volatility regimes without changing the signal.",
        "high volatility risk sizing atr stddev drawdown position sizing"),
    Doc("ATR trailing stop",
        "Exit a long position when price falls more than k x ATR (k = 2-4) below the highest close "
        "since entry. This limits losses in reversals while letting trends run. Track the highest "
        "close in next() with a member variable.",
        "risk stop loss atr trailing drawdown high volatility"),
    Doc("MACD momentum",
        "Go long when the MACD line crosses above its signal line, ideally only when MACD > 0 "
        "(momentum confirmation), and exit on the reverse cross. MACD lags; it suits smooth trends "
        "and produces false signals in choppy markets.",
        "momentum trend macd signal crossover uptrend"),
    Doc("Regime filter: stay in cash in downtrends",
        "Only allow long entries when the close is above the 200-day SMA (or the 50-day SMA is above "
        "the 200-day). Historically this avoids most of the large drawdowns of equities, at the cost "
        "of some upside after V-shaped recoveries.",
        "downtrend regime filter drawdown sma 200 risk"),
    Doc("Buy-and-hold benchmark awareness",
        "For stocks in a strong long-term uptrend, buy-and-hold is hard to beat after costs. Active "
        "rules should aim to cut drawdowns (e.g. a regime filter or stops) while remaining invested "
        "most of the time; strategies that trade rarely or sit in cash too long usually underperform.",
        "uptrend low volatility benchmark buy and hold exposure"),
    Doc("Stochastic oscillator swing trading",
        "Buy when %K crosses above %D below 20 (oversold) and sell when it crosses below %D above 80. "
        "Effective in range-bound, mean-reverting markets; in trends combine it with a trend filter "
        "and trade only in the trend direction.",
        "mean-reverting range-bound stochastic oscillator swing"),
    Doc("Avoiding overfitting",
        "Use few parameters with round, conventional values (14, 20, 50, 200). Strategies with many "
        "tuned thresholds fit noise and fail out of sample. Prefer rules that are sensible across "
        "many stocks and periods.",
        "robustness overfitting parameters out of sample"),
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _TOKEN_RE.findall(text.lower())


class KnowledgeBase:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self._tf = [Counter(_tokens(f"{d.title} {d.tags} {d.text}")) for d in self.docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df = Counter(t for tf in self._tf for t in tf)
        n = len(self.docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @classmethod
    def default(cls):
        return cls(DEFAULT_DOCS)

    @classmethod
    def from_dir(cls, path):
        """One Doc per paragraph (blank-line separated) of every .md/.txt file under `path`."""
        docs = []
        for root, _, files in os.walk(path):
            for name in sorted(files):
                if name.endswith(('.md', '.txt')):
                    text = open(os.path.join(root, name), encoding='utf-8').read()
                    for i, para in enumerate(p.strip() for p in re.split(r"\n\s*\n", text)):
                        if len(para) > 40:
                            docs.append(Doc(f"{name}#{i}", para))
        return cls(docs)

    def score(self, query):
        q = _tokens(query)
        out = []
        for tf, length in zip(self._tf, self._len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self._avg))
            out.append(s)
        return out

    def retrieve(self, query, k=3):
        scores = self.score(query)
        order = sorted(range(len(self.docs)), key=lambda i: (-scores[i], i))
        return [self.docs[i] for i in order[:k]]

    def context_block(self, query, k=3):
        if k <= 0:
            return ""
        docs = self.retrieve(query, k)
        body = '\n'.join(f"[{i + 1}] {d.title}: {d.text}" for i, d in enumerate(docs))
        return f"RELEVANT STRATEGY KNOWLEDGE (retrieved for this stock's regime):\n{body}"
