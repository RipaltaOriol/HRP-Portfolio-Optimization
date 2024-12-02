from typing import List

from deepdiff import DeepHash
import pandas as pd
from typing import List


class Metric:
    def __init__(self, name: str, freq: str, starting_capital: int = 10000):
        self.name = name
        self.freq = freq
        self.starting_capital = starting_capital
    
    def calculate(self, weights: pd.DataFrame, tickers: List[str], data: pd.DataFrame, **kwargs):
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
        Returns the calculation for the target metric
        """
        return NotImplementedError

    def __hash__(self):
        # TODO: missing docs
        return DeepHash(self)[self]

    def groupby_freq(dataframe, freq):
        # TODO: missing docs
        if freq == 'D':
            return dataframe.groupby([dataframe.index.date])
        elif freq == 'W':
            return dataframe.groupby([dataframe.index.year, dataframe.index.isocalendar().week])
        elif freq == 'M':
            return dataframe.groupby([dataframe.index.month])
        elif freq == 'Y':
            return dataframe.groupby([dataframe.index.year])
        elif freq == 'YM':
            return dataframe.groupby([dataframe.index.year, dataframe.index.month])
        elif freq == 'P':
            return dataframe

    
    def to_frame_and_indexing(data, freq, name):
        # TODO: missing docs
        # TODO: should be added for non linear metrics
        if isinstance(data, pd.DataFrame):
            return data

        if freq == 'P':
            if isinstance(data, pd.Series):
                data.index = ['period']
                return data
            else:
                return pd.DataFrame({name: [data]}, index=['period'])
        return data




