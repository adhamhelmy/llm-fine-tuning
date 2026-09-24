"""The strategy-generation prompt and helpers to parse it back."""

import re

_RULES = """\
- Framework: Backtrader
- Strategy must subclass bt.Strategy
- The strategy will be passed directly into:
run_backtest(StrategyClass, symbols, start, end, timeframe, cash)

STRICT RULES:
1. Output ONLY a single Python class definition (no explanations, no markdown, no comments outside the class).
2. The class MUST be named Strategy.
3. Do NOT include imports (bt is already available).
4. Do NOT reference external data, files, APIs, or indicators outside Backtrader.
5. The strategy MUST work for single-symbol backtests.
6. All indicators must be created in __init__.
7. Trading logic must be implemented in next().
8. Orders must use only: self.buy(), self.sell(), self.close(), self.order_target_percent().
9. No plotting, printing, logging, or analyzers.
10. Strategy must be deterministic and backtest-safe (no lookahead bias)."""

_TEMPLATE = """\
OUTPUT FORMAT:
Return ONLY the Python class exactly like this structure:

class Strategy(bt.Strategy):

    params = dict(
        # parameters here
    )

    def __init__(self):
        # indicator definitions

    def next(self):
        # trading logic

DO NOT output anything else."""

_COMPACT_FOOTER = "Return ONLY the Python class. DO NOT output anything else."

_PARAMS_RE = re.compile(r'trading strategy for (\w+) from (\S+) to (\S+)')


def make_prompt(symbol: str, start: str, end: str, compact: bool = False) -> str:
    """Build the strategy prompt.

    compact=False: full prompt incl. the class skeleton (used for GRPO training).
    compact=True:  same rules without the skeleton (used by the model-comparison notebooks).
    """
    header = (f"Create a trading strategy for {symbol} from {start} to {end} "
              "that is fully compatible with the following backtesting setup:")
    footer = _COMPACT_FOOTER if compact else _TEMPLATE
    return f"{header}\n\n{_RULES}\n\n{footer}"


def extract_trading_parameters(prompt: str):
    """Return (symbol, start, end) embedded in a prompt built by make_prompt."""
    match = _PARAMS_RE.search(prompt)
    if match:
        return match.group(1), match.group(2), match.group(3)
    raise ValueError(f"Could not extract trading parameters from prompt: {prompt[:100]}")
