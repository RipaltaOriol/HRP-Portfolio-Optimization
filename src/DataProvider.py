import yfinance as yf
import pandas as pd
from typing import List

class DataProvider:
    def __init__(self, tickers: List[str], start: str, end: str, target: str = "Adj Close") -> None:
        """
        List[str]: tickers
        str: start date for data
        str: end date for data
        str: target column to use for returns
        """
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = pd.DataFrame()  # initialize to empty
        self.target = target

    def provide(self) -> pd.DataFrame:
        """
        Main class function which returns the ticker data requested.
        ----
        Returns pd.DataFrame with ticker returns for the class date range.
        """
        self.fetch()
        self.clean()
        self.calc_returns()
        return self.data


    def fetch(self) -> pd.DataFrame:
        """
        Fetches historical for the specified asset classes.
        ----
        Returns pd.DataFrame with ticker data.
        """
        self.data = yf.download(self.tickers, self.start, self.end)
        self.data = self.data[self.target] if self.target else self.data



    def clean(self, brute = True) -> None:
        """
        Clean up the data
        """
        if self.data.isnull().values.any():
            print("The dataset contains null or empty values")
            print("Pefroming cleaning")
            if brute:
                self.data = self.data.dropna()
            else: # TODO: the code below has not been tested
                missing_fractions = self.data.isnull().mean().sort_values(ascending=False)
                missing_fractions.head(10)
                drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
                self.data.drop(labels=drop_list, axis=1, inplace=True)
                # fill the missing values with the last value available in the dataset.
                self.data = self.data.fillna(method='ffill')
        else:
            print("The dataset contains no null values")
        return True


    def calc_returns(self):
        """
        Computes returns for the given data.
        ----
        Returns pd.DataFrame with ticker returns.
        """
        self.data = self.data.pct_change()
        self.data = self.data.dropna()