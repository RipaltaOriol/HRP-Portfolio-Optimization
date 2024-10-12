import yfinance as yf
import pandas as pd
from typing import List

class DataProvider:
    def __init__(self, tickers: List[str], start: str, end: str, target: str = "Adj Close") -> None:
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = pd.DataFrame() # initialize to empty
        self.target = target

    def fetch(self) -> pd.DataFrame:
        """
        Fetches historical for the specified asset classes.
        """
        self.data = yf.download(self.tickers, self.start, self.end)
        self.data = self.data[self.target] if self.target else self.data
        self.clean()
        return self.data

    def clean(self) -> None:
        """
        Clean up the data
        """
        self.data = self.data.dropna()
        return True
