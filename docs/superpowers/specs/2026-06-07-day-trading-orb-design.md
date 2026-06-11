# Day Trading Strategy — Opening Range Breakout (ORB) with VWAP Filter

**Date:** 2026-06-07
**File:** `strategies/day_trading_orb.py`

---

## Overview

A Backtrader intraday strategy that trades the momentum burst following the opening 30 minutes of the US equity session. Entries are filtered by VWAP direction to avoid false breakouts. Risk is managed with ATR-based stops, a fixed reward/risk target, and a hard end-of-day close at 15:45 ET.

Compatible with `strategies/tester.py` out of the box.

---

## Signal Logic

**Data:** 5-minute bars via Alpaca (`--interval minute` passed to tester).

**Opening Range (OR):** The high and low of the first 30 minutes of the session (6 × 5-min bars). No trades fire during this window; the strategy only observes.

**VWAP:** Cumulative intraday VWAP — `sum(typical_price × volume) / sum(volume)` — calculated from the first bar of each session and reset every day.

**Entry rules (one trade per session per symbol):**
- **Long:** close > OR high AND close > VWAP
- **Short:** close < OR low AND close < VWAP

Once a trade is taken (or stopped/targeted), no further entries that day.

---

## Risk Management

**Stop loss:** `entry ± (ATR₁₄ × atr_stop_mult)` — long stops below entry, short stops above.

**Take profit:** `entry ± (stop_distance × rr_ratio)` — fixed risk/reward, default 2:1.

**Position sizing:** `size = (portfolio_value × risk_pct) / stop_distance` — risks a fixed percentage of portfolio per trade, keeping drawdowns proportional to account size.

**Force-flat:** At 15:45 ET any open position is closed at market. Stop/target tracking is cleared. Prevents overnight exposure.

**Stop/target management:** Checked manually each bar in `next()` — no bracket orders. More portable across Backtrader versions.

### Tunable Parameters

| Param           | Default | Purpose                          |
|-----------------|---------|----------------------------------|
| `or_minutes`    | 30      | Opening range window (minutes)   |
| `bar_minutes`   | 5       | Bar size (must match data feed)  |
| `atr_period`    | 14      | ATR lookback                     |
| `atr_stop_mult` | 1.5     | Stop distance in ATR units       |
| `rr_ratio`      | 2.0     | Reward / risk ratio              |
| `risk_pct`      | 0.01    | Portfolio fraction risked/trade  |
| `close_hour`    | 15      | Force-flat hour (ET)             |
| `close_minute`  | 45      | Force-flat minute (ET)           |

---

## Implementation Structure

**File:** `strategies/day_trading_orb.py`

### `__init__`
- Instantiate `bt.indicators.ATR(period=atr_period)`
- Initialise all session state: `or_high`, `or_low`, `or_set`, `vwap_num`, `vwap_den`, `bars_today`, `trade_taken`, `current_day`, `entry_price`, `stop_price`, `target_price`, `direction`

### `_reset_session(dt)`
- Called on the first bar of each new calendar date
- Clears all daily state listed above

### `next` — sequential bar flow
1. Detect new day → `_reset_session()`
2. Update VWAP accumulator (`vwap_num`, `vwap_den`)
3. Extend OR window (track high/low); return early until `bars_today >= or_bars`
4. Force-flat check: if time ≥ 15:45 and in position → cancel tracking, `self.close()`
5. Skip if `self.position` or `trade_taken`
6. **Long breakout:** `close > or_high and close > vwap` → size → `self.buy()`
7. **Short breakout:** `close < or_low and close < vwap` → size → `self.sell()`
8. **Stop/target management (when in position):** check `close ≤ stop_price` (long) or `close ≥ stop_price` (short) → `self.close()`; check `close ≥ target_price` (long) or `close ≤ target_price` (short) → `self.close()`

> **Note:** Stop/target fills are evaluated on bar close (not intra-bar). This is standard for OHLCV backtesting and slightly optimistic on stop accuracy — acceptable for strategy validation.
> **Note:** Force-flat also fires on the last available bar of the day if 15:45 is not present in the feed (e.g. data ends at 15:55).

---

## Usage

```bash
# SPY, 2022-2024
python strategies/tester.py day_trading_orb --symbols SPY --start 2022-01-01 --end 2024-12-31 --interval minute

# Multi-symbol (Mag 7)
python strategies/tester.py day_trading_orb --symbols mag7 --start 2023-01-01 --end 2024-12-31 --interval minute
```

```python
from tester import run_backtest, load_strategy
Strategy = load_strategy('day_trading_orb')
run_backtest(Strategy, symbols='SPY', start='2022-01-01', end='2024-12-31', interval='minute')
```
