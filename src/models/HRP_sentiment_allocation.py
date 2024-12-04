import datetime
import numpy as np
import urllib3
from .HRP_calculator import HRP_Calculator, HRP_Calculator_2, HRP_Calculator_3
import pandas as pd
from .base import WeightAllocationModel
import matplotlib.pyplot as plt
from models import plot_hrp_weights
from models import SentimentAnalyzer
import asyncio


class HRP_Sentiment(WeightAllocationModel):
    def __init__(self, months_back=3, include_sentiment=False, async_getter=True, is_shrinkage = True):
        super(HRP_Sentiment, self).__init__()
        self.months_back = months_back
        self.include_sentiment = include_sentiment
        self.async_getter = async_getter
        self.sentiment_analyzer = SentimentAnalyzer()
        self.is_shrinkage = is_shrinkage

    def __str__(self):
        if self.include_sentiment:
            return f"HRP_WithSentiment"
        else:
            return f"HRP_NoSentiment"

    def __hash__(self):
        if self.include_sentiment:
            return f"HRP_WithSentiment".__hash__()
        else:
            return f"HRP_NoSentiment_NoSentiment".__hash__()

    def date_data_needed(self, date_from, date_to):
        return date_from - pd.DateOffset(months=self.months_back)

    def weights_allocate(self, date_from, date_to, ticker_list, data, **params):
        weights_list = []

        for rebalance_date in pd.date_range(start=date_from, end=date_to, freq='MS'):

            start_date = rebalance_date - pd.DateOffset(months=self.months_back)
            end_date = rebalance_date - pd.DateOffset(days=1)
            past_data = data.loc[start_date:end_date, ticker_list]

            if past_data.empty or len(past_data) < 2:
                continue

            hrp_calculator = HRP_Calculator(past_data, self.is_shrinkage)
            hrp_weights = hrp_calculator.weights_allocate()


            if self.include_sentiment is False:
                if isinstance(hrp_weights, pd.Series):
                    weights_data = [hrp_weights.values]
                else:
                    weights_data = [hrp_weights.values()]

                weights_df = pd.DataFrame(data=weights_data, index=[rebalance_date], columns=hrp_weights.keys())
                weights_df = weights_df[data.columns]

            else:
                weights_df = self.add_sentiment(start_date, end_date, data, hrp_weights, rebalance_date, linear_adjustment=False)

            weights_list.append(weights_df)

            plot_hrp_weights(weights_df.T.squeeze(), len(weights_list))

        weight_predictions = pd.concat(weights_list)
        weight_predictions = weight_predictions.sort_index()

        return weight_predictions

    def add_sentiment(self, start_date, end_date, ticker_list, hrp_weights: pd.Series, rebalance_date: datetime.date, linear_adjustment=False, **params):

        # if statement to get polygon data asynchronously or not
        if self.async_getter:
            aggregated_sentiments = self.async_sentiment_getter(start_date, end_date, ticker_list)
        else:
            aggregated_sentiments = self.sentiment_getter(start_date, end_date, ticker_list)

        # think about the adjustment here. There is crazy bias. if we have small weights, but crazy sentiment, there won't be a difference
        if linear_adjustment:
            adjusted_weights = {ticker: hrp_weights.get(ticker, 0) * (1 + aggregated_sentiments.get(ticker, 0)) for ticker in ticker_list}
        else:
            k = 2.5  # You can adjust k to control the impact of sentiment
            adjusted_weights = {ticker: hrp_weights.get(ticker, 0) * np.exp(k * aggregated_sentiments.get(ticker, 0)) for ticker in ticker_list}

        # normalize adjusted weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:  # avoid division by zero, else statement here should never be triggered
            normalized_weights = {ticker: weight / total_weight for ticker, weight in adjusted_weights.items()}
        else:
            normalized_weights = {ticker: 0 for ticker in ticker_list}

        weights_df = pd.DataFrame(data=[normalized_weights.values()], index=[rebalance_date], columns=normalized_weights.keys())

        return weights_df

    def sentiment_getter(self, start_date, end_date, ticker_list, **params):

        sentiment_scores = {}
        aggregated_sentiments = {}

        for ticker in ticker_list:
            ticker_news = self.sentiment_analyzer.fetch_ticker_news_with_retries(start_date, end_date, ticker)
            filtered_news = [article for article in ticker_news if article['publisher']['homepage_url'] != 'https://www.zacks.com/']
            ticker_news = filtered_news

            sentiment_scores[ticker] = self.sentiment_analyzer.calculate_finbert_sentiment(ticker_news)
            aggregated_sentiments[ticker] = self.sentiment_analyzer.calculate_finbert_aggregate_sentiment(sentiment_scores[ticker])

        return aggregated_sentiments

    def async_sentiment_getter(self, start_date, end_date, ticker_list, **params):

        sentiment_scores = {}
        aggregated_sentiments = {}

        news = asyncio.run(self.sentiment_analyzer.fetch_all_ticker_news(start_date, end_date, ticker_list))
        for ticker, ticker_news in zip(ticker_list, news):
            filtered_news = [article for article in ticker_news if article['publisher']['homepage_url'] != 'https://www.zacks.com/']
            ticker_news = filtered_news

            sentiment_scores[ticker] = self.sentiment_analyzer.calculate_finbert_sentiment(ticker_news)
            aggregated_sentiments[ticker] = self.sentiment_analyzer.calculate_finbert_aggregate_sentiment(sentiment_scores[ticker])

        return aggregated_sentiments




