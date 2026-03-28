class Strategy(bt.Strategy):
    params = dict(
        n1=20, n2=50, threshold=0.05
    )

    def __init__(self):
        self.data.close = self.datas[0].close
        self.sma1 = bt.indicators.SimpleMovingAverage(self.data, period=self.params.n1)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.data, period=self.params.n2)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

    def next(self):
        if not self.position:
            if self.sma1 > self.sma2 and self.rsi < 30:
                self.buy(exectype=bt.Order.Limit, price=self.data.close)
        else:
            if self.sma1 < self.sma2 and self.rsi > 70:
                self.close()