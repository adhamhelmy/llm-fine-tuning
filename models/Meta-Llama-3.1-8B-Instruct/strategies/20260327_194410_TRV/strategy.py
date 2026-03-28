class Strategy(bt.Strategy):
    params = dict(
        short_window=20,
        long_window=50,
        risk_alloc=0.8
    )

    def __init__(self):
        self.fast_ma = bt.indicators.MovingAverageSimple(period=self.p.short_window)
        self.slow_ma = bt.indicators.MovingAverageSimple(period=self.p.long_window)

    def next(self):
        if self.data.close > self.fast_ma and self.data.close > self.slow_ma:
            if self.position:
                if self.position.size == 0:
                    self.close()
            self.order_target_percent(target=self.p.risk_alloc)
        elif self.data.close <= self.fast_ma and self.data.close <= self.slow_ma:
            if not self.position:
                self.close()
            self.order_target_percent(valuess=0)