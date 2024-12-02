from src.metrics.Metric import Metric

class MaxDrawdown(Metric):

    def __init__(self, freq: str = 'D'):
        super().__init__(name = "Max. Drawdown", freq = freq)

    def calculate(self, weights, tickers, data, **kwargs):
        """    
        Parameters
        ----------
        weights: pd.DataFrame
            The dataframe of weights provided by the model
        tikcers: List[str]
            A list of tickers used in the model
        data: pd.DataFrame
            A dataframe of the data used in the backtesting
        -------
        Returns the dradown for a given iteration
        """

        portfolio_returns = (weights * data).sum(axis=1)

        cumulative_returns = (1 + portfolio_returns).cumprod()

        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max

        drawdown_df = drawdowns.to_frame(name="Drawdown")

        return drawdown_df.reindex(data.index).fillna(0)