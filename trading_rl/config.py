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

# Known data problems in Alpaca's daily bars (verified 2016-2024). Alpaca forward-fills
# inactive tickers with flat, zero-volume bars, so these backtests are silently wrong before
# `valid_from`. Backtester.load_bars() warns about them.
SYMBOL_ISSUES = {
    'COIN': ('2021-04-14', 'IPO 2021-04-14: no bars at all before, every backtest raises'),
    'DOW':  ('2019-04-02', 'Dow Chemical until the 2017 DowDuPont merger, then flat at $44.18 '
                           'with zero volume until the Dow Inc. spin-off (2019-04-02)'),
    'NIO':  ('2018-09-12', 'ticker of a Nuveen muni fund until the NIO Inc. IPO (2018-09-12): '
                           'flat at $14.67 with ~zero volume for 2016-2018'),
}

# GRPO training universe: DOW30 with NVDA instead of INTC, plus extra large caps, minus every
# symbol in SYMBOL_ISSUES (their 2016-2019 bars are missing or forward-filled). 35 symbols.
# (The first experiments used 38: these 35 + COIN, DOW, NIO.)
TRAINING_SYMBOLS = [s for s in [x if x != 'INTC' else 'NVDA' for x in DOW30] +
                    ['AMZN', 'COIN', 'GE', 'GOOGL', 'NFLX', 'NIO', 'TSLA', 'UVV']
                    if s not in SYMBOL_ISSUES]

# Dev/test evaluation universe: DOW30 minus SYMBOL_ISSUES (29 symbols). DOW itself is fine from
# 2020, but the dev split's feedback window (2018-2019, used by repair/refine and the market
# summary) is flat for it, so it is dropped everywhere to keep one universe for dev and test.
EVAL_SYMBOLS = [s for s in DOW30 if s not in SYMBOL_ISSUES]

# S&P 100 constituents at the start of the test split (January 2022), point-in-time to limit
# survivorship bias; current tickers (META = FB, ELV = ANTM). GOOG is left out (same company as
# GOOGL). Compiled from memory: check against an official constituent list before publishing.
SP100_2022 = [
    'AAPL', 'ABBV', 'ABT',  'ACN',  'ADBE', 'AIG',  'AMGN', 'AMT',  'AMZN', 'AVGO',
    'AXP',  'BA',   'BAC',  'BIIB', 'BK',   'BKNG', 'BLK',  'BMY',  'BRK.B', 'C',
    'CAT',  'CHTR', 'CL',   'CMCSA', 'COF', 'COP',  'COST', 'CRM',  'CSCO', 'CVS',
    'CVX',  'DHR',  'DIS',  'DOW',  'DUK',  'ELV',  'EMR',  'EXC',  'F',    'FDX',
    'GD',   'GE',   'GILD', 'GM',   'GOOGL', 'GS',  'HD',   'HON',  'IBM',  'INTC',
    'JNJ',  'JPM',  'KHC',  'KO',   'LIN',  'LLY',  'LMT',  'LOW',  'MA',   'MCD',
    'MDLZ', 'MDT',  'MET',  'META', 'MMM',  'MO',   'MRK',  'MS',   'MSFT', 'NEE',
    'NFLX', 'NKE',  'NVDA', 'ORCL', 'PEP',  'PFE',  'PG',   'PM',   'PYPL', 'QCOM',
    'RTX',  'SBUX', 'SO',   'SPG',  'T',    'TGT',  'TMO',  'TMUS', 'TSLA', 'TXN',
    'UNH',  'UNP',  'UPS',  'USB',  'V',    'VZ',   'WBA',  'WFC',  'WMT',  'XOM',
]

# Held-out TEST universe: S&P 100 of January 2022 plus the Dow 30 of the same period (adds TRV),
# both point-in-time at the start of the test split. GRPO never trains on any of these tickers.
TEST_SYMBOLS = list(dict.fromkeys(SP100_2022 + DOW30))

# Old name: test stocks that were not in the legacy 35-symbol TRAINING_SYMBOLS.
UNSEEN_SYMBOLS = [s for s in SP100_2022 if s not in TRAINING_SYMBOLS and s not in SYMBOL_ISSUES]

# GRPO training universe for the final run: large/mid-cap US stocks (mostly S&P 500 members
# outside the S&P 100), minus every TEST_SYMBOLS ticker. The list is NOT a point-in-time index:
# it only has to be liquid US stocks disjoint from the test universe. Tickers that Alpaca doesn't
# know, that listed after the training window started, or whose bars are flat/forward-filled are
# dropped at load time (Backtester.clean_symbols), so the list may contain a few dead names.
_TRAIN_CANDIDATES = """
A AAL AAP ADI ADM ADP ADSK AEE AEP AES AFL AJG AKAM ALB ALGN ALK ALL ALLE AMAT AMCR AMD AME AMP
ANET ANSS AON AOS APA APD APH APTV ARE ATO AVB AVY AWK AZO BALL BAX BBY BDX BEN BF.B BIO BKR BR
BRO BSX BWA BXP CAG CAH CB CBOE CBRE CCI CCL CDNS CDW CE CF CFG CHD CHRW CI CINF CLX CMA CME CMG
CMI CMS CNC CNP COO COR CPB CPRT CRL CSX CTAS CTRA CTSH CZR D DAL DD DE DFS DG DGX DHI DLR DLTR
DOV DPZ DRI DTE DVA DVN DXC DXCM EA EBAY ECL ED EFX EIX EL EMN ENPH EOG EQIX EQR ES ESS ETN ETR
EVRG EW EXPD EXPE EXR FANG FAST FCX FDS FE FFIV FIS FITB FMC FRT FTNT GIS GL GLW GNRC GPC GPN
GRMN GWW HAL HAS HBAN HCA HES HIG HII HLT HOLX HPE HPQ HRL HSIC HST HSY HUM IDXX IEX IFF ILMN
INCY INTU IP IPG IQV IR IRM ISRG IT ITW IVZ J JBHT JCI JKHY JNPR K KEY KEYS KIM KLAC KMB KMI KMX
KR L LDOS LEN LH LHX LKQ LNC LNT LRCX LUV LVS LW LYB LYV MAA MAR MAS MCHP MCK MCO MGM MHK MKC
MKTX MLM MMC MNST MOS MPC MPWR MSCI MSI MTB MTD MU NCLH NDAQ NEM NI NOC NOW NRG NSC NTAP NTRS NUE
NVR NWSA O ODFL OKE OMC ORLY OXY PAYC PAYX PCAR PEG PFG PGR PH PHM PKG PLD PNC PNR PNW POOL PPG
PPL PRU PSA PSX PTC PWR RCL REG REGN RF RHI RJF RL RMD ROK ROL ROP ROST RSG SBAC SCHW SEE SHW SJM
SLB SNA SNPS SPGI SRE STE STT STX STZ SWK SWKS SYF SYK SYY TAP TDG TDY TEL TER TFC TFX TPR TRMB
TROW TSCO TSN TT TTWO TXT TYL UAL UDR UHS ULTA URI VFC VLO VMC VRSK VRSN VRTX VTR WAB WAT WDC WEC
WELL WHR WM WMB WRB WST WY WYNN XEL XRAY XYL YUM ZBH ZBRA ZION ZTS
UVV
""".split()
TRAIN_UNIVERSE = [s for s in dict.fromkeys(_TRAIN_CANDIDATES)
                  if s not in TEST_SYMBOLS and s not in SYMBOL_ISSUES]

SYMBOL_SETS = {
    'eval': EVAL_SYMBOLS,
    'sp100': SP100_2022,
    'unseen': UNSEEN_SYMBOLS,
    'test': TEST_SYMBOLS,
    'train_universe': TRAIN_UNIVERSE,
    'dow30': DOW30,            # the index as it was (includes DOW: check SYMBOL_ISSUES)
    'mag7': MAG7,
    'training': TRAINING_SYMBOLS,
}


def usable_symbols(symbols, start):
    """Drop symbols whose data is unreliable for a window starting at `start` (see SYMBOL_ISSUES)."""
    return [s for s in symbols if s not in SYMBOL_ISSUES or SYMBOL_ISSUES[s][0] <= start]


# Time splits. Alpaca daily bars start in 2016.
#   train: GRPO training windows. Since 2026-11-14 (final run) it absorbs the old dev years
#          (2020-21): every setting was already fixed, so dev had nothing left to choose.
#   dev:   kept for the ladder/config experiments that were run on it (docs/paper/configs_tested.md).
#          It is INSIDE the final GRPO training window: never report a GRPO adapter on it.
#   test:  final, reported numbers only; never used for any selection
# Harness/loop feedback always comes from the FEEDBACK_YEARS before the evaluated window.
TRAIN_START, TRAIN_END = '2016-01-01', '2021-12-31'
DEV_START, DEV_END = '2020-01-01', '2021-12-31'
# TEST was extended from 2024-12-31 to 2025-12-31 on 2026-11-14, before any alpha-reward result
# on TEST was looked at: 4 calendar years / regimes instead of 3.
TEST_START, TEST_END = '2022-01-01', '2025-12-31'
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
