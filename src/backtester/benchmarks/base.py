from deepdiff import DeepHash


class Benchmark:

    def __init__(self, name='Base_Benchmark', freq='M', starting_capital=10000000):
        self.name = name
        self.freq = freq

    def calculate(self, weight_allocations, ticker_list, data, **kwargs):
        pass

    def __hash__(self):
        return DeepHash(self)[self]

    @staticmethod
    def groupby_freq(dataframe, freq):
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

