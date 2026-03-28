class Strategy(bt.Strategy):
    params = dict(
        cash_usage=0.8,           # Use 80% of cash for trading
        stop_loss_pct=-0.05,      # -5% stop loss
        take_profit_pct=0.10,     # 10% take profit
        window_length=200,        # Lookback period for SMA
        rsi_period=14,            # RSI period
        trailing_stop_trigger=0.02 # 2% trail trigger
    )

    def __init__(self):
        # Core indicators
        self.data_close = self.datas[0].close
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low

        # Trend indicators
        self.sma = bt.indicators.SMA(period=self.p.window_length)

        # Momentum indicators
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)

        # Volatility and position sizing
        self.atr = bt.indicators.ATR(period=self.p.window_length)

        # Current position and protection tracking
        self.current_trailing_stop = None
        self.in_long = False

    def next(self):
        if not self.position:
            # Entry conditions (RSI + SMA + ATR)
            if (self.rsi[0] < 30) and (self.datas[0].close[0] > self.sma[0]):
                self.in_long = True
                self.current_trailing_stop = self.data_low[0] - (self.atr[0] * 1.5)
                self.order_target_percent(target=self.p.cash_usage)
        else:
            # Exit conditions with trailing stop (no fixed exit)
            current_low = self.data_low[0]
            new_trailing_stop = current_low - (self.atr[0] * 1.5)

            if current_low < self.current_trailing_stop or \
               current_low < new_trailing_stop:
                self.close()

                # Reset tracking for next opportunity
                self.in_long = False
                self.current_trailing_stop = None

            # Dynamic position management
            elif self.rsi[0] > 70:
                self.order_target_percent(target=0)