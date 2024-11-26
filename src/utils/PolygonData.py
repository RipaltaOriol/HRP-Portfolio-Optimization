import aiohttp
import asyncio
import pandas as pd
import requests
import datetime


class MarketCapFetcher:
    def __init__(self):
        self.polygon_api_key = 'LTLVSbi7rBjyjJtCpmLuTDPPhFsNSCyy'

    def get_next_trading_day(self, date):
        """
        Check if the given date is a trading day. If not (weekend or holiday), return the next trading day.
        """
        url = "https://api.polygon.io/v1/marketstatus/upcoming"
        params = {"apiKey": self.polygon_api_key}

        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            holidays = response.json()

            # Convert the date to datetime.date for easier comparison
            date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()

            # Loop to find the next valid trading day
            while True:
                # Skip weekends
                if date_obj.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    date_obj += datetime.timedelta(days=1)
                    continue

                # Skip holidays
                holiday_dates = [datetime.datetime.strptime(holiday["date"], "%Y-%m-%d").date() for holiday in holidays]
                if date_obj in holiday_dates:
                    date_obj += datetime.timedelta(days=1)
                    continue

                # Found a valid trading day
                return date_obj.strftime("%Y-%m-%d")
        except requests.exceptions.RequestException as e:
            print(f"Error checking next trading day: {e}")
            return date  # Fallback to the given date

    async def fetch_price_data_async(self, session: aiohttp.ClientSession, ticker, date):
        """
        Fetch historical price data for a specific date asynchronously using Polygon.io.
        """
        url = f"https://api.polygon.io/v1/open-close/{ticker}/{date}"
        params = {"apiKey": self.polygon_api_key}

        try:
            async with session.get(url, params=params, timeout=5) as response:
                response.raise_for_status()
                json_response = await response.json()
                if "close" in json_response:
                    return json_response["close"]
                else:
                    print(f"No price data available for {ticker} on {date}")
                    return None
        except aiohttp.ClientError as e:
            print(f"Error fetching price data for {ticker} on {date}: {e}")
            return None

    async def fetch_shares_outstanding_async(self, session: aiohttp.ClientSession, ticker):
        """
        Fetch shares outstanding for a ticker asynchronously using Polygon.io.
        """
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        params = {"apiKey": self.polygon_api_key}

        try:
            async with session.get(url, params=params, timeout=5) as response:
                response.raise_for_status()
                json_response = await response.json()
                if "results" in json_response and "share_class_shares_outstanding" in json_response["results"]:
                    return json_response["results"]["share_class_shares_outstanding"]
                else:
                    print(f"No shares outstanding data available for {ticker}")
                    return None
        except aiohttp.ClientError as e:
            print(f"Error fetching shares outstanding for {ticker}: {e}")
            return None

    async def fetch_market_cap_for_ticker(self, session: aiohttp.ClientSession, ticker, date):
        """
        Fetch price data and shares outstanding asynchronously for a specific date,
        and calculate market cap for a ticker.
        """
        close_price = await self.fetch_price_data_async(session, ticker, date)
        shares_outstanding = await self.fetch_shares_outstanding_async(session, ticker)

        if close_price is not None and shares_outstanding is not None:
            market_cap = close_price * shares_outstanding
            return {"date": date, "ticker": ticker, "market_cap": market_cap}
        else:
            return {"date": date, "ticker": ticker, "market_cap": None}

