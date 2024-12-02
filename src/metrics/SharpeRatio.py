from metrics.Metric import Metric

class SharpeRatio(Metric):

    def __init__(self, freq: str = 'D', rf: float = 0):
        super().__init__(name="Sharpe Ratio", freq = freq)
        self.rf = rf

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
        Returns the Sharpe Ratio for a given iteration
        """
        # TODO: remove tickers if not used
        
        portfolio_returns = (weights * data).sum(axis=1)

        excess_returns = portfolio_returns - self.risk_free_rate
        grouped_returns = self.groupby_freq(excess_returns, self.freq)
        mean = grouped_returns.mean()
        std = grouped_returns.std()
        sharpe_ratio = mean / std

        sharpe_ratio_df = self.to_frame_and_indexing(sharpe_ratio, self.freq, self.name)
        #sharpe_ratio_df = sharpe_ratio_df.replace([float('inf'), float('-inf')], 0).fillna(0)
        return sharpe_ratio_df