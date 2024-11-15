from src.backtester.benchmarks.base import Benchmark


class PNL(Benchmark):

    """"
    First 3 lines of calculate function should be implemented in every calculation function, in order to be able to compare weight_allocations and daily_returns
    """
    def __init__(self, freq='D'):
        super(PNL, self).__init__(name="PNL ($)", freq=freq)


    def calculate(self, weight_weight_predictions, ticker_list, data, **kwargs):

        daily_returns = data.pct_change().fillna(0)
        weight_weight_predictions = weight_weight_predictions.resample('D').ffill().dropna(how='all')
        weight_weight_predictions = weight_weight_predictions.reindex(daily_returns.index).ffill().fillna(0)

        portfolio_pnl = (weight_weight_predictions * daily_returns).sum(axis=1)
        #portfolio_pnl_cumsum = portfolio_pnl.cumsum()
        portfolio_pnl_df = portfolio_pnl.to_frame(name="Portfolio PnL")

        grouped_pnl = self.groupby_freq(portfolio_pnl_df, self.freq).sum()

        return (grouped_pnl)

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
