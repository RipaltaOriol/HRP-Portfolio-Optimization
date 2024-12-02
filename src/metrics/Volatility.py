
from src.metrics.Metric import Metric

class Volatility(Metric):

    def __init__(self, freq: str = 'D', window: int = 20):
        super().__init__(name = "Volatility", freq = freq)
        self.window = window  # Default rolling window is 20 days

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
        Returns the volatility for a given iteration
        """
        # TODO: remove tickers if not used

        portfolio_returns = (weights * data).sum(axis=1)

        rolling_volatility = portfolio_returns.rolling(self.window).std()

        volatility_df = rolling_volatility.to_frame(name="Volatility")

        return volatility_df.reindex(data.index).fillna(0)