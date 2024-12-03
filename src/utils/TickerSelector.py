import pandas as pd
import ssl
import random
import yfinance as yf

class TickerSelector:
    def __init__(self):
        """
        initialize and fetch S&P 500 tickers from Wikipedia
        """
        self.tickers = self._fetch_sp500_tickers()

    def _fetch_sp500_tickers(self):
        """
        Fetch the S&P 500 tickers from Wikipedia.
        Returns:
            List of tickers (str): A list of all S&P 500 tickers.
        """
        try:
            # Disable SSL verification
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Fetch S&P 500 data from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_table = pd.read_html(url)[0]  # The first table contains the tickers
            return sp500_table['Symbol'].tolist()  # Extract tickers as a list
        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            return []
        
    def random_selection(self, num_stocks=100):
        """random tickers"""
        return random.sample(self.tickers, num_stocks)
    
    def calculate_ticker_returns(self, start_date, end_date, num_stocks=100):
        """ calc returns of sp500 in order to choose top/bottom"""
        ticker_data = {}
        for ticker in self.tickers:
            ticker = ticker.replace('.', '-') # different convention on wiki VS yfinance
            try:
                stock = yf.Ticker(ticker)
                # Fetch historical data for the custom date range
                history = stock.history(start=start_date, end=end_date)
                if not history.empty:
                    # Calculate total return over the period
                    returns = (history["Close"].iloc[-1] - history["Close"].iloc[0]) / history["Close"].iloc[0]
                    ticker_data[ticker] = returns
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                print(ticker)
        return ticker_data
    
    def select_top_return_tickers(self, ticker_data, num_stocks=100):
        """
        top tickers by returns 
        """
        # sort descending order
        sorted_tickers = sorted(ticker_data.items(), key=lambda x: x[1], reverse=True)
        return [ticker[0] for ticker in sorted_tickers[:num_stocks]]
    
    def select_bottom_tickers(self, ticker_returns, num_stocks=100):
        """
            Select the bottom tickers 
        """
        # Sort tickers by returns in ascending order
        sorted_tickers = sorted(ticker_returns.items(), key=lambda x: x[1])

        # Select the bottom tickers
        return [ticker[0] for ticker in sorted_tickers[:num_stocks]]



    