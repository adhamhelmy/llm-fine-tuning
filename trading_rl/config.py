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

TRAIN_START = '2016-01-01'
TRAIN_END = '2024-12-31'
