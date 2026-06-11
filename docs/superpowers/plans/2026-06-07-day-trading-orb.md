# Day Trading ORB Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `strategies/day_trading_orb.py` — an Opening Range Breakout strategy with VWAP filter, ATR stops, fixed R:R target, and hard EOD close.

**Architecture:** Single `bt.Strategy` subclass. Session state (VWAP, OR levels, trade flags) is reset on every new calendar day inside `next()`. Stop and target are tracked as instance attributes and checked on each bar — no bracket orders.

**Tech Stack:** Python 3, Backtrader, Alpaca (1-min bars via `tester.py`). No test suite in this repo — verification is via `tester.py` smoke runs.

> **Bar size note:** Alpaca's REST API only exposes `TimeFrame.Minute` (1-min bars). The strategy uses `bar_minutes=1` so `or_minutes=30` correctly captures 30 one-minute bars as the opening range window.

---

### Task 1: Class skeleton — params and `__init__`

**Files:**
- Create: `strategies/day_trading_orb.py`

- [ ] **Step 1: Create the file with params and `__init__`**

```python
import backtrader as bt


class Strategy(bt.Strategy):
    """
    Opening Range Breakout with VWAP filter.

    Builds the opening range from the first `or_minutes` bars, then enters
    long on a break above OR high (if price > VWAP) or short below OR low
    (if price < VWAP). Stop is ATR-based; target is stop_distance × rr_ratio.
    All positions are force-closed at close_hour:close_minute.
    """

    params = dict(
        or_minutes=30,       # opening range window in minutes
        bar_minutes=1,       # bar size in minutes (Alpaca minimum = 1)
        atr_period=14,       # ATR lookback
        atr_stop_mult=1.5,   # stop distance in ATR units
        rr_ratio=2.0,        # reward / risk ratio for take-profit
        risk_pct=0.01,       # fraction of portfolio risked per trade
        close_hour=15,       # force-flat hour (ET)
        close_minute=45,     # force-flat minute (ET)
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # session state — all reset daily
        self.current_day  = None
        self.bars_today   = 0
        self.or_high      = None
        self.or_low       = None
        self.vwap_num     = 0.0   # cumulative (typical_price × volume)
        self.vwap_den     = 0.0   # cumulative volume
        self.vwap         = 0.0
        self.trade_taken  = False
        self.direction    = 0     # 1 = long, -1 = short, 0 = flat
        self.stop_price   = None
        self.target_price = None
```

- [ ] **Step 2: Verify the file parses cleanly**

```bash
cd /path/to/llm-fine-tuning
python3 -c "import sys; sys.path.insert(0,'strategies'); from day_trading_orb import Strategy; print('OK')"
```

Expected output: `OK`

---

### Task 2: Session reset and position-sizing helpers

**Files:**
- Modify: `strategies/day_trading_orb.py` — add two methods to the class

- [ ] **Step 1: Add `_reset_session` and `_position_size` inside the `Strategy` class, after `__init__`**

```python
    def _reset_session(self, dt):
        self.current_day  = dt.date()
        self.bars_today   = 0
        self.or_high      = None
        self.or_low       = None
        self.vwap_num     = 0.0
        self.vwap_den     = 0.0
        self.vwap         = 0.0
        self.trade_taken  = False
        self.direction    = 0
        self.stop_price   = None
        self.target_price = None

    def _position_size(self, entry, stop):
        risk_per_share = abs(entry - stop)
        if risk_per_share == 0:
            return 0
        dollar_risk = self.broker.getvalue() * self.p.risk_pct
        return int(dollar_risk / risk_per_share)
```

- [ ] **Step 2: Verify the file still parses**

```bash
python3 -c "import sys; sys.path.insert(0,'strategies'); from day_trading_orb import Strategy; print('OK')"
```

Expected output: `OK`

---

### Task 3: VWAP accumulator and opening range builder

**Files:**
- Modify: `strategies/day_trading_orb.py` — add `next` with steps 1–3 of the bar flow

- [ ] **Step 1: Add `next` method — day reset, VWAP update, and OR window**

```python
    def next(self):
        dt = self.data.datetime.datetime(0)

        # 1. Detect new session
        if dt.date() != self.current_day:
            self._reset_session(dt)

        # 2. Update intraday VWAP
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        self.vwap_num += tp * self.data.volume[0]
        self.vwap_den += self.data.volume[0]
        self.vwap = self.vwap_num / self.vwap_den if self.vwap_den > 0 else tp

        # 3. Build opening range — return early until window is complete
        self.bars_today += 1
        or_bars = self.p.or_minutes // self.p.bar_minutes
        if self.bars_today <= or_bars:
            if self.or_high is None:
                self.or_high = self.data.high[0]
                self.or_low  = self.data.low[0]
            else:
                self.or_high = max(self.or_high, self.data.high[0])
                self.or_low  = min(self.or_low,  self.data.low[0])
            return
```

- [ ] **Step 2: Verify the file parses**

```bash
python3 -c "import sys; sys.path.insert(0,'strategies'); from day_trading_orb import Strategy; print('OK')"
```

Expected output: `OK`

---

### Task 4: Entry signals

**Files:**
- Modify: `strategies/day_trading_orb.py` — extend `next` with force-flat and entry logic

- [ ] **Step 1: Append force-flat check and entry logic inside `next` (after the OR return)**

```python
        # 4. Force-flat: close any position by close_hour:close_minute
        past_close = (
            dt.hour > self.p.close_hour or
            (dt.hour == self.p.close_hour and dt.minute >= self.p.close_minute)
        )
        if past_close:
            if self.position:
                self.close()
                self.trade_taken = True
            return

        # 5. Skip if already traded today or in a position
        if self.trade_taken or self.position:
            return

        # 6. Entry signals
        price = self.data.close[0]
        atr   = self.atr[0]

        if price > self.or_high and price > self.vwap:
            stop   = price - atr * self.p.atr_stop_mult
            target = price + (price - stop) * self.p.rr_ratio
            size   = self._position_size(price, stop)
            if size > 0:
                self.buy(size=size)
                self.stop_price   = stop
                self.target_price = target
                self.direction    = 1
                self.trade_taken  = True

        elif price < self.or_low and price < self.vwap:
            stop   = price + atr * self.p.atr_stop_mult
            target = price - (stop - price) * self.p.rr_ratio
            size   = self._position_size(price, stop)
            if size > 0:
                self.sell(size=size)
                self.stop_price   = stop
                self.target_price = target
                self.direction    = -1
                self.trade_taken  = True
```

- [ ] **Step 2: Verify the file parses**

```bash
python3 -c "import sys; sys.path.insert(0,'strategies'); from day_trading_orb import Strategy; print('OK')"
```

Expected output: `OK`

---

### Task 5: Stop and target management

**Files:**
- Modify: `strategies/day_trading_orb.py` — insert stop/target check **before** the "skip if already traded" guard in `next`

The full `next` execution order must be:
1. Day reset  →  2. VWAP  →  3. OR build (return)  →  4. Force-flat (return)  →  **5. Stop/target** →  6. Skip guard  →  7. Entry

- [ ] **Step 1: Insert stop/target block between force-flat and the skip guard**

Replace the `# 5. Skip if already traded today or in a position` comment and everything after it with:

```python
        # 5. Stop / target management (fires while in position, before entry check)
        if self.position:
            price = self.data.close[0]
            hit_stop = (
                (self.direction == 1  and price <= self.stop_price) or
                (self.direction == -1 and price >= self.stop_price)
            )
            hit_target = (
                (self.direction == 1  and price >= self.target_price) or
                (self.direction == -1 and price <= self.target_price)
            )
            if hit_stop or hit_target:
                self.close()
            return  # stay in this branch; no new entries while in position

        # 6. Skip if already traded today
        if self.trade_taken:
            return

        # 7. Entry signals
        price = self.data.close[0]
        atr   = self.atr[0]

        if price > self.or_high and price > self.vwap:
            stop   = price - atr * self.p.atr_stop_mult
            target = price + (price - stop) * self.p.rr_ratio
            size   = self._position_size(price, stop)
            if size > 0:
                self.buy(size=size)
                self.stop_price   = stop
                self.target_price = target
                self.direction    = 1
                self.trade_taken  = True

        elif price < self.or_low and price < self.vwap:
            stop   = price + atr * self.p.atr_stop_mult
            target = price - (stop - price) * self.p.rr_ratio
            size   = self._position_size(price, stop)
            if size > 0:
                self.sell(size=size)
                self.stop_price   = stop
                self.target_price = target
                self.direction    = -1
                self.trade_taken  = True
```

- [ ] **Step 2: Verify the file parses**

```bash
python3 -c "import sys; sys.path.insert(0,'strategies'); from day_trading_orb import Strategy; print('OK')"
```

Expected output: `OK`

---

### Task 6: Smoke test with tester.py

**Files:**
- Read only: `strategies/tester.py`

- [ ] **Step 1: Run backtest on SPY, 2023 (one calendar year, fast)**

```bash
cd /path/to/llm-fine-tuning/strategies
python3 tester.py day_trading_orb --symbols SPY --start 2023-01-01 --end 2023-12-31 --interval minute --no-plot
```

Expected: Script fetches bars, prints per-bar progress, then prints the results block:
```
==================================================
  Total return  : <any number>%
  Avg annual    : <any number>%
  Sharpe ratio  : <number or N/A>
  Max drawdown  : <number>%
==================================================
```

If it crashes with `AttributeError` or `KeyError`, re-check that `_reset_session` initialises every attribute accessed in `next`.

- [ ] **Step 2: Verify at least some trades fired**

Add `--no-plot` output will not show trade count directly. Confirm trades happened by checking that `Total return` is not exactly `0.00%` (which would mean no trades at all, or the strategy bought and sold at identical prices every time).

If return is exactly 0.00%, run with a shorter window to debug:
```bash
python3 tester.py day_trading_orb --symbols AAPL --start 2023-06-01 --end 2023-09-30 --interval minute --no-plot
```

- [ ] **Step 3: Run on Mag 7, 2022–2024 for a fuller picture**

```bash
python3 tester.py day_trading_orb --symbols mag7 --start 2022-01-01 --end 2024-12-31 --interval minute --no-plot
```

Expected: completes without exception, prints results block.
