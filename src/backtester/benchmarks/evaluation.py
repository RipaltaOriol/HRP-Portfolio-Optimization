from src.backtester.benchmarks.base import Benchmark


class PNL(Benchmark):

    def __init__(self, freq='D'):
        super(PNL, self).__init__(name="PNL ($)", freq=freq)

    def calculate(self, weight_allocations, ticker_list, **kwargs):
        pass

class Sharpe(Benchmark):

    def __init__(self, freq='D'):
        super(Sharpe, self).__init__(name="Sharpe", freq=freq)

    def calculate(self, weight_allocations, ticker_list, **kwargs):
        pass

class MaxDrawdown(Benchmark):

    def __init__(self, freq='D'):
        super(MaxDrawdown, self).__init__(name="MaxDrawdown", freq=freq)

    def calculate(self, weight_allocations, ticker_list, **kwargs):
        pass
