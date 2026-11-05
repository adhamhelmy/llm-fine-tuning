"""Shared constants: symbol universes, date splits, transaction costs."""

# Dow Jones Industrial Average as of 2020-08-31 .. 2024-02-26 (AMGN/CRM/HON in, WBA still in).
DOW30 = [
    'AAPL', 'AMGN', 'AXP',  'BA',   'CAT',
    'CRM',  'CSCO', 'CVX',  'DIS',  'DOW',
    'GS',   'HD',   'HON',  'IBM',  'INTC',
    'JNJ',  'JPM',  'KO',   'MCD',  'MMM',
    'MRK',  'MSFT', 'NKE',  'PG',   'TRV',
    'UNH',  'V',    'VZ',   'WBA',  'WMT',
]

MAG7 = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']

# 38-symbol GRPO training universe: DOW30 with NVDA instead of INTC, plus 8 extra names.
# Membership is unchanged from the original experiments; see SYMBOL_ISSUES before training.
TRAINING_SYMBOLS = [s if s != 'INTC' else 'NVDA' for s in DOW30] + [
    'AMZN', 'COIN', 'GE', 'GOOGL', 'NFLX', 'NIO', 'TSLA', 'UVV',
]

SYMBOL_SETS = {
    'dow30': DOW30,
    'mag7': MAG7,
    'training': TRAINING_SYMBOLS,
}

# Known data problems in Alpaca's daily bars (verified 2016-2024). Alpaca forward-fills
# inactive tickers with flat, zero-volume bars, so these backtests are silently wrong before
# `valid_from`. Backtester.load_bars() warns about them; filter with usable_symbols().
SYMBOL_ISSUES = {
    'COIN': ('2021-04-14', 'IPO 2021-04-14: no bars at all before, every backtest raises'),
    'DOW':  ('2019-04-02', 'Dow Chemical until the 2017 DowDuPont merger, then flat at $44.18 '
                           'with zero volume until the Dow Inc. spin-off (2019-04-02)'),
    'NIO':  ('2018-09-12', 'ticker of a Nuveen muni fund until the NIO Inc. IPO (2018-09-12): '
                           'flat at $14.67 with ~zero volume for 2016-2018'),
}


def usable_symbols(symbols, start):
    """Drop symbols whose data is unreliable for a window starting at `start` (see SYMBOL_ISSUES)."""
    return [s for s in symbols if s not in SYMBOL_ISSUES or SYMBOL_ISSUES[s][0] <= start]


# Time splits (no overlap). Alpaca daily bars start in 2016.
#   train: GRPO training windows
#   dev:   choosing the best option at each ladder stage (prompt, context, harness, ...)
#   test:  final, reported numbers only; never used for any selection
# Harness/loop feedback always comes from the FEEDBACK_YEARS before the evaluated window.
TRAIN_START, TRAIN_END = '2016-01-01', '2019-12-31'
DEV_START, DEV_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'
FEEDBACK_YEARS = 2

# (commission_bps, slippage_bps) per side. Used by the reward (RewardConfig.costs) and by the
# robustness report. 'none' is the original frictionless setting.
COST_LEVELS = {
    'none': (0, 0),
    'low': (1, 2),       # ~3 bps per side: liquid large caps, retail commission-free broker
    'mid': (5, 5),       # 10 bps per side
    'high': (10, 15),    # 25 bps per side: conservative / less liquid names
}
DEFAULT_COSTS = 'low'    # cost level of the default reward (training + evaluation)
