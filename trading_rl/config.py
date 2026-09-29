"""Shared constants: symbol universes, default date ranges, Alpaca endpoint."""

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# 38-symbol universe used for GRPO training and model comparison.
TRAINING_SYMBOLS = [
    'AAPL', 'AMGN', 'AXP',  'BA',   'CAT',
    'CRM',  'CSCO', 'CVX',  'DIS',  'DOW',
    'GS',   'HD',   'HON',  'IBM',  'JNJ',
    'JPM',  'KO',   'MCD',  'MMM',  'MRK',
    'MSFT', 'NKE',  'NVDA', 'PG',   'TRV',
    'UNH',  'V',    'VZ',   'WBA',  'WMT',
    'AMZN', 'COIN', 'GE',   'GOOGL', 'NFLX',
    'NIO',  'TSLA', 'UVV',
]

# First 30 training symbols (Dow-like set used by the strategy testers).
TESTER_SYMBOLS = TRAINING_SYMBOLS[:30]

DOW30 = [
    'AAPL', 'AMGN', 'AXP',  'BA',   'CAT',
    'CRM',  'CSCO', 'CVX',  'DIS',  'DOW',
    'GS',   'HD',   'HON',  'IBM',  'INTC',
    'JNJ',  'JPM',  'KO',   'MCD',  'MMM',
    'MRK',  'MSFT', 'NKE',  'PG',   'TRV',
    'UNH',  'V',    'VZ',   'WBA',  'WMT',
]

MAG7 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']

SYMBOL_SETS = {
    'dow30': DOW30,
    'mag7': MAG7,
    'training': TRAINING_SYMBOLS,
}

# Time splits (no overlap). Alpaca daily bars start in 2016.
#   train: GRPO training windows
#   dev:   choosing the best option at each ladder stage (prompt, context, harness, ...)
#   test:  final, reported numbers only; never used for any selection
# Harness/loop feedback always comes from the FEEDBACK_YEARS before the evaluated window.
TRAIN_START, TRAIN_END = '2016-01-01', '2019-12-31'
DEV_START, DEV_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'
FEEDBACK_YEARS = 2
