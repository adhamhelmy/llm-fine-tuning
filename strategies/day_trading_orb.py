import backtrader as bt
from zoneinfo import ZoneInfo

_ET = ZoneInfo('America/New_York')


class Strategy(bt.Strategy):
    """
    Opening Range Breakout with VWAP filter — v4.

    Key design choices:
      - All time checks in ET (Alpaca bars are UTC)
      - OR built from regular market hours only (9:30–16:00 ET)
      - Long-only: short ORB consistently loses on most US equities
      - Position cap: prevents over-leverage when ATR is small vs price
      - Volume confirmation: breakout bar must have volume > vol_mult × SMA
      - Buffer entry: close must exceed OR by half a stop to filter fakes
    """

    params = dict(
        or_minutes=45,          # opening range window
        bar_minutes=1,          # bar size in minutes
        atr_period=100,         # ATR lookback (longer = smoother on 1-min bars)
        atr_stop_mult=1.0,      # stop = ATR × this
        min_stop_pct=0.003,     # stop floor = 0.3% of price
        rr_ratio=2.0,           # reward / risk ratio
        risk_pct=0.01,          # portfolio fraction risked per trade
        max_pos_pct=0.95,       # position cap (no leverage)
        close_hour=15,          # force-flat hour (ET)
        close_minute=45,        # force-flat minute (ET)
        min_or_range_pct=0.002, # skip days where OR < 0.2% of price
        vol_period=50,          # volume SMA period for breakout confirmation
        vol_mult=1.2,           # breakout bar must have volume > vol_mult × SMA
        max_gap_pct=0.02,       # skip days where open gaps > 2% from prior close
    )

    def __init__(self):
        self.atr    = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.vol_period)

        self.current_day  = None
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
        self.prior_close  = None   # last market-hours close of the previous day
        self.gap_skip     = False  # True = skip today due to large gap

    def _et(self):
        return self.data.datetime.datetime(0).astimezone(_ET)

    def _in_market_hours(self, dt_et):
        return (dt_et.hour, dt_et.minute) >= (9, 30) and dt_et.hour < 16

    def _reset_session(self, day, open_price):
        gap_skip = False
        if self.prior_close and self.prior_close > 0:
            gap = abs(open_price - self.prior_close) / self.prior_close
            gap_skip = gap > self.p.max_gap_pct
        self.current_day  = day
        self.bars_today   = 0
        self.or_high      = None
        self.or_low       = None
        self.vwap_num     = 0.0
        self.vwap_den     = 0.0
        self.vwap         = 0.0
        self.trade_taken  = gap_skip   # skip entry if gap is too large
        self.gap_skip     = gap_skip
        self.direction    = 0
        self.stop_price   = None
        self.target_price = None

    def _position_size(self, entry, stop):
        risk_per_share = abs(entry - stop)
        if risk_per_share == 0:
            return 0
        portfolio     = self.broker.getvalue()
        size_by_risk  = int(portfolio * self.p.risk_pct / risk_per_share)
        size_by_cap   = int(portfolio * self.p.max_pos_pct / entry)
        return min(size_by_risk, size_by_cap)

    def _stop_distance(self, price):
        atr      = self.atr[0]
        atr_stop = atr * self.p.atr_stop_mult if atr == atr else 0
        pct_stop = price * self.p.min_stop_pct
        return max(atr_stop, pct_stop)

    def next(self):
        dt_et = self._et()

        # 1. Reset on new ET calendar day
        if dt_et.date() != self.current_day:
            self._reset_session(dt_et.date(), self.data.open[0])

        # 2. Skip pre-market / after-hours; capture prior close at last market bar
        if not self._in_market_hours(dt_et):
            return
        self.prior_close = self.data.close[0]

        # 3. Update intraday VWAP (skip zero-volume bars)
        tp  = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        vol = self.data.volume[0]
        if vol > 0:
            self.vwap_num += tp * vol
            self.vwap_den += vol
        self.vwap = self.vwap_num / self.vwap_den if self.vwap_den > 0 else tp

        # 4. Build opening range
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

        # 5. Force-flat at close_hour:close_minute ET
        if dt_et.hour > self.p.close_hour or (
                dt_et.hour == self.p.close_hour and dt_et.minute >= self.p.close_minute):
            if self.position:
                self.close()
                self.trade_taken = True
            return

        # 6. Stop / target management
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
                self.direction = 0
            return

        # 7. Skip if already traded today
        if self.trade_taken:
            return

        price     = self.data.close[0]
        stop_dist = self._stop_distance(price)
        if stop_dist == 0:
            return

        # 8. Quality filters
        or_range = self.or_high - self.or_low
        if or_range < price * self.p.min_or_range_pct:
            return

        vol_ma = self.vol_ma[0]
        if vol_ma != vol_ma or vol == 0:  # NaN or zero
            return

        # 9. Long breakout: close above OR + buffer, VWAP above, volume expanding
        buffer = stop_dist * 0.5
        if (price > self.or_high + buffer
                and price > self.vwap
                and vol > vol_ma * self.p.vol_mult):
            stop   = price - stop_dist
            target = price + stop_dist * self.p.rr_ratio
            size   = self._position_size(price, stop)
            if size > 0:
                self.buy(size=size)
                self.stop_price   = stop
                self.target_price = target
                self.direction    = 1
                self.trade_taken  = True
