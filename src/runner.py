from random import Random
from agents.main import Agent
from backtester.main import Backtester
from datetime import date
# from models.HRP_allocation import HRP
from models.HRP_calculator import HRP_Calculator_3 as HRPModel # TODO: rename this
from models.HRP_sentiment_allocation import HRP_Sentiment
from models.other_models import EqualWeights, MarketCapWeights
from metrics import PNL, SharpeRatio
from utils.Frequency import Frequency
from tickers import tickers # TODO: fix this import

start_date = date(2024, 1, 1)
simulate_date = date(2024, 2, 29)
end_date = date(2024, 4, 29)

frequency = Frequency.MONTH_END


metrics = [
    PNL.PNL("P"),
    # SharpeRatio.SharpeRatio("P"),
    # PNL.PNL("YM"),
    # SharpeRatio.SharpeRatio("YM")
]

# agents = [Agent(HRP(months_back=1))] # TODO: Look at the initialization
agents = [
    Agent(HRPModel)
]

backtester = Backtester(
    start_date = start_date,
    end_date = end_date,
    simulation_date = simulate_date,
    frequency = frequency,
    tickers = tickers,
    metrics = metrics)

# Add the agents to the backcaster one by one.
for agent in agents:
    backtester.add_agent(agent)

# run the backtester

backtester.execute()

# Export the results to an excel file. Display parameter is for printing results to console as well.
# backtester.results_to_excel2(
#     filename='backtesting_simulation_{}-{}.xlsx'.format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')),
#     save_dir='results',
#     disp=True
# )

