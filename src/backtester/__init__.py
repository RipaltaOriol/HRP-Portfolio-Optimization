import pandas as pd
from agent import  Agent
import copy



class Backtester:

    def __init__(self,start_date, end_date, ticker_list, benchmarks, save=False):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.benchmarks = benchmarks
        self.save = save

        self.data_from = None
        self.agents = []
        self.new_agents = []
        self.changed_agents = []
        self.benchmarks = benchmarks

    def data_date_from(self):

        date_from = self.start_date
        for agent in self.agents:
            temp = agent.date_data_needed(self.start_date, self.end_date)
            if pd.Timestamp(temp) < pd.Timestamp(date_from):
                date_from = temp

        return date_from

    def get_data(self):

        if self.new_agents:
            self.data_from = self.data_date_from()

            if self.data_from is None:
                raise ValueError("You have to provide agents for evaluations")

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the simulation/
        :param agent: Agent object to be used for simulation
        """
        self.agents.append(agent)
        self.new_agents.append(agent)

    def remove_agent(self, agent):
        """
        Removes an already added agent from the simulation.
        :param agent: Agent instance of the agent you want to be removed.
        """
        try:
            self.agents.remove(agent)
        except ValueError:
            raise Warning("Model was not found in the evaluator")

    def clear_agents(self):
        """
        Deletes all agents from the simulation. Results will be still be available.
        """
        self.agents = []


    def agents_predict(self):
        """
        In this method all agents, one by one, predict their backcasts for the simulation period.
        """
        for agent in self.new_agents:
            print(f"Predictions for {agent}, are being calculated.")
            agent.predict(self.start_date, self.end_date, self.ticker_list)
            print(f"Predictions for {agent}, done.\n")

    def evaluate_agents(self, benchmarks=None):
        """
        This is where the agents are evaluated based on specified benchmarks. Returns a dictionary that has as keys,
        the frequencies of the benchmarks, e.g. "D" for daily benchmarks, "W" for weekly etc. and as values the
        DataFrames of the specified frequencies with all the benchmarks associated with them.
        :param benchmarks: Benchmarks that the agents will be evaluated at.
        :return: Dictionary with the specified format.
        """
        try:
            quantities = self.agents[0].quantities
        except:
            raise ValueError(
                "Agents haven't decided their quantities yet please run agents_decide first!"
            )

        results = {
            "D": pd.DataFrame(index=quantities.groupby([quantities.index.date]).sum().index),
            "W": pd.DataFrame(index=quantities.groupby([quantities.index.year, quantities.index.isocalendar().week]).sum().index),
            "M": pd.DataFrame(index=quantities.groupby([quantities.index.month]).sum().index),
            "Y": pd.DataFrame(index=quantities.groupby([quantities.index.year]).sum().index),
            "YM": pd.DataFrame(index=quantities.groupby([quantities.index.year, quantities.index.month]).sum().index),
            "P": pd.DataFrame(),
        }
        for agent in self.agents:
            self.results[agent.sheet_name()] = copy.deepcopy(results)
            for benchmark in benchmarks:
                benchmark_result = benchmark.calculate(agent.weight_predictions, self.ticker_list)
                self.results[agent.sheet_name()][benchmark.freq][benchmark.name] = benchmark_result
        return self.results