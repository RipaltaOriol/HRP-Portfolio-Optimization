from random import Random
from src.agents import Agent
from src.backtester import Backtester
from datetime import date
import src.backtester.benchmarks.evaluation as b
from src.models.random_allocation import RandomAllocation
from src.models.HRP_allocation import HRP
from ticker_codes import tickers

# make sure to pip install -r requirements.txt

start_date = date(2024, 1, 1)
end_date = date(2024, 10, 29)

benchmarks = [b.PNL('P'),b.Sharpe('P')]

#agents = [Agent(RandomAllocation(months_back=1))]
agents = [Agent(HRP(months_back=2))]

back_tester = Backtester(start_date=start_date,
                         end_date=end_date,
                         ticker_list= tickers,
                         benchmarks=benchmarks)

# Add the agents to the backcaster one by one.
for agent in agents:
    back_tester.add_agent(agent)

# Expected to run for about 5 minutes for all models.
back_tester.run_n_evaluate()

# Export the results to an excel file. Display parameter is for printing results to console as well.
back_tester.results_to_excel2(
    filename='backtesting_simulation_{}-{}.xlsx'.format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')),
    save_dir='results',
    disp=True
)

x=2
# call save once on exit, even if multiple files were created during the simulation.




