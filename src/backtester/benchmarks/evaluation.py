from src.backtester.benchmarks.base import Benchmark


class PNL(Benchmark):

    def __init__(self, freq='D'):
        super(PNL, self).__init__(name="PNL ($)", freq=freq)

    def calculate(self, weight_allocations, ticker_list, data, **kwargs):

        daily_returns = data.pct_change().fillna(0)

        weight_allocations = weight_allocations.reindex(daily_returns.index, method='ffill').fillna(0)

        portfolio_pnl = (weight_allocations * daily_returns).sum(axis=1)
        portfolio_pnl_cumsum = portfolio_pnl.cumsum()
        portfolio_pnl_df = portfolio_pnl_cumsum.to_frame(name="Portfolio PnL")

        grouped_pnl = self.groupby_freq(portfolio_pnl_df, self.freq).sum().round(0)

        return grouped_pnl

class Sharpe(Benchmark):

    def __init__(self, freq='D'):
        super(Sharpe, self).__init__(name="Sharpe", freq=freq)

    def calculate(self, weight_allocations, ticker_list, data, **kwargs):
        pass

class MaxDrawdown(Benchmark):

    def __init__(self, freq='D'):
        super(MaxDrawdown, self).__init__(name="MaxDrawdown", freq=freq)

    def calculate(self, weight_allocations, ticker_list, data, **kwargs):
        pass
