from metrics.Metric import Metric
import pandas as pd

class PNL(Metric):
    def __init__(self, freq: str = 'D'):
        super().__init__(name="PNL (%)", freq = freq)

    def calculate(self, weights, returns, **kwargs):
        """    
        Parameters
        ----------
        weights: pd.DataFrame
            The dataframe of weights provided by the model
        data: pd.DataFrame
            A dataframe of the data used in the backtesting
        -------
        Returns the PnL for a given iteration
        EJ lecture: log(1+x) = x approximately, for small x. SO if we use cumsum() on returns we approximate the cumulative return
        more precisely: log(total_ret) = log((1+r1)*(1+r2)*...*(1_r_n)) = log(1+r1) + log(1+r2) + ... + log(1+r_n) = r1 + r2 + rn returns
        """
        # TODO: review note in docstring
        # TODO: remove tickers if not used
        
        # weights = weights.reindex(returns.index, method='ffill')
        print("____________")
        print(weights)
        print(returns)
        common_tickers = weights.columns.intersection(returns.columns)
        print(common_tickers)
        weights = weights[common_tickers]
        returns = returns[common_tickers]
        print(weights * returns)
        
        return weights
        print("____________")
        portfolio_returns = (weights * returns).sum(axis=1)
        # print(portfolio_returns, type(portfolio_returns))
        return True


        # Set initial capital
        # initial_capital = 1_000_000  # Example: $1,000,000

        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Calculate portfolio values
        portfolio_values = self.starting_capital * cumulative_returns

        # Calculate PnL (Profit and Loss)
        pnl = portfolio_values - self.starting_capital

        results_df = pd.DataFrame({
            'Portfolio_Return': portfolio_returns,
            'Cumulative_Return': cumulative_returns,
            'Portfolio_Value': portfolio_values,
            'PnL': pnl
        }, index=returns.index)
        print(results_df)

        # portfolio_returns_cumsum = portfolio_pnl.cumsum()

        # continue here
        # print(type(portfolio_returns))
        # print(portfolio_returns)
        return True
        # portfolio_returns_df = portfolio_returns.to_frame(name=self.name)

        # grouped_pnl = self.groupby_freq(portfolio_returns_df, self.freq).sum()*100
        # grouped_pnl = self.to_frame_and_indexing(grouped_pnl, self.freq, self.name)

        # return (grouped_pnl)