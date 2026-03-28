class Strategy(bt.Strategy):

    params = dict(
        short_window=50,
        long_window=200
    )

    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_window)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_window)

    def next(self):
        if self.sma_short[0] > self.sma_long[0] and not self.position:
            self.order_target_percent(target=1.0)
        elif self.sma_short[0] < self.sma_long[0] and self.position:
            self.close()