import pandas as pd
from agents.main import Agent
import copy
from datetime import datetime
from utils.DataProvider import DataProvider
from utils.Frequency import Frequency
import os
# from backtester import WeightAllocationModel
from models.base import WeightAllocationModel
from typing import List

class Backtester:

    def __init__(self, start_date: datetime, simulation_date: datetime, end_date: datetime, frequency: Frequency, tickers: List[str], metrics, save=False):
        self.start_date = start_date
        self.simulation_date = simulation_date
        self.end_date = end_date
        self.frequency = frequency # TODO: make this a enum
        self.tickers = sorted(tickers)
        self.metrics = metrics

        self.agents = []
        self.data = None

        self.results = {}
        self.excel_writer = None
        self.save = save
        WeightAllocationModel.save = save

    
    def execute(self):
        """
        Main function to execute the backtest class
        """
        self.fetch_data()
        self.simulate()
        self.evaluate()

        return True
        
        
        return results

    def fetch_data(self):
        """
        Parameters
        ----------
        Assigns the fetch tickers data to a class variable: self.data
        """
        # TODO: CHECK THAT END DATE IS NOT GREATER THAT TODAY'S DATE
        if self.agents: # TODO: rethink this condition
            data_provider = DataProvider(self.start_date, self.end_date, self.tickers)
            self.data = data_provider.provide()

            if self.tickers != self.data.columns.to_list():
                print(self.tickers, self.data.columns.to_list())
                # self.tickers = self.data.columns.values
                print("Something went wrong")
                print('Tickers succesfully set to data_columns, because tickers didnt match data.columns. Check data')


    def add_agent(self, agent: List[Agent]) -> None:
        """    
        Parameters
        ----------
        agents: List[Agent]
            A list of agents/models
        -------
        Adds the given agents to the backtester
        """
        self.agents.append(agent)


    def allocate_agents(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        training_data: pd.DataFrame
            A dataframe with date rows and columns tickers with returns as cells
        ----------
        Exectues the each of the agents/models for the given training data
        """
        for agent in self.agents:
            agent.allocate(training_data)
    
    def simulate(self):
        # TODO: missing docstring
        for rebalance_date in pd.date_range(start = self.simulation_date, end = self.end_date, freq = self.frequency.value):
            print(rebalance_date, "____")
            # print(rebalance_date, "rebalance date")
            training_data = self.data.loc[self.start_date : rebalance_date]
            # TODO: check that data is not empty
            self.allocate_agents(training_data)
            # print(type(allocation))
            # now I must store the allocations
            
        # return NotImplementedError
    def evaluate(self):
        # TODO: ensure the agents have been exeucted
        # TODO: refactor code & docstring
        """
        This is where the agents are evaluated based on specified benchmarks. Returns a dictionary that has as keys,
        the frequencies of the benchmarks, e.g. "D" for daily benchmarks, "W" for weekly etc. and as values the
        DataFrames of the specified frequencies with all the benchmarks associated with them.
        :param benchmarks: Benchmarks that the agents will be evaluated at.
        :return: Dictionary with the specified format.
        """
        for agent in self.agents:
            history_weights = agent.get_allocations()
            history_weights_T = history_weights.T

            # Ensure index are of type datetime
            history_weights_T.index = pd.to_datetime(history_weights_T.index)
            self.data.index = pd.to_datetime(self.data.index)

            for metric in self.metrics:
                return metric.calculate(history_weights_T, self.data)
                pass
                # considerations: capital


        # try:

        #     weight_predictions = self.agents[0].weight_predictions
        # except:
        #     raise ValueError(
        #         "Agents haven't decided their weights for the whole period yet, please run agent.weights_allocate first!"
        #     )

        # results = {
        #     "D": pd.DataFrame(index=weight_predictions.groupby([weight_predictions.index.date]).sum().index),
        #     "W": pd.DataFrame(index=weight_predictions.groupby([weight_predictions.index.year, weight_predictions.index.isocalendar().week]).sum().index),
        #     "M": pd.DataFrame(index=weight_predictions.groupby([weight_predictions.index.month]).sum().index),
        #     "Y": pd.DataFrame(index=weight_predictions.groupby([weight_predictions.index.year]).sum().index),
        #     "YM": pd.DataFrame(index=weight_predictions.groupby([weight_predictions.index.year, weight_predictions.index.month]).sum().index),
        #     "P": pd.DataFrame(),
        # }

        # self.data = self.data[(self.data.index.date >= self.start_date) & (self.data.index.date <= self.end_date)]
        # for agent in self.agents:
        #     self.results[agent.sheet_name()] = copy.deepcopy(results)
        #     for benchmark in self.metrics:
        #         benchmark_result = benchmark.calculate(agent.weight_predictions, self.tickers, self.data)
        #         self.results[agent.sheet_name()][benchmark.freq][benchmark.name] = benchmark_result
        # return self.results


    # TODO: refactor this
    def results_to_excel2(self, filename: str, save_dir=".", disp=False):
        """
        Export the results of the simulation to an Excel file and display them in the console.
        :param filename: Filename of the Excel file.
        :param save_dir: Directory in which the file will be saved relative to the backtesting project.
        :param disp: Boolean parameter to print results in the console.
        """
        if not self.results:
            raise ValueError("Please run evaluate_agents first to generate the results.")

        # Construct file path
        filepath = os.path.abspath(os.path.join(save_dir, filename))
        count = 1
        while os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            filepath = os.path.abspath(os.path.join(save_dir, f"{name} ({count}){ext}"))
            count += 1

        if self.excel_writer is None:
            self.excel_writer = pd.ExcelWriter(filepath)
        else:
            if filepath != self.excel_writer.path:
                self.excel_writer.save()
                self.excel_writer = pd.ExcelWriter(filepath)

        # Create a new Excel writer
        with pd.ExcelWriter(filepath) as writer:
            for agent, agent_results in self.results.items():
                if disp:
                    print(f"Agent: {agent}")

                row = 0
                for frequency, frequency_results in agent_results.items():
                    if frequency_results.empty:
                        continue

                    if disp:
                        print(f"\nFrequency: {frequency}")
                        print(frequency_results)

                    try:
                        # write results to Excel
                        frequency_results.to_excel(writer, sheet_name=agent[:31],  # Excel sheet name limit is 31 characters
                            startrow=row, float_format="%.6f", )
                        row += frequency_results.shape[0] + 2  # Leave a gap between sections
                    except Exception as ex:
                        raise ValueError(f"Failed to write to Excel: {ex}")

                print('\n')

        if disp:
            print(f"\nResults successfully saved to: {filepath}")
