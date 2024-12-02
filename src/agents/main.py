# from agents import WeightAllocationModel
import pandas as pd

class Agent:
    # TODO: rename WeightAllocationModel to AllocationModel
    def __init__(self, model):
        # TODO: add typehiints to the model
        self.model = model
        self.history_weights = pd.DataFrame()

    def allocate(self, data: pd.DataFrame):

        # initializae the model
        model_instance = self.model(data)
        # TODO: wrap this in a try block
        weights = model_instance.weights_allocate() # TODO: rename this
    
        self.history_weights[data.index[-1]] = weights
        # df_weights = pd.DataFrame(data=weights, index=[data.index[-1]])
        # print(self.history_weights)
        # print(df_weights, df_weights.index)
        # print(df_weights[data.columns])
        # weights_df =  weights_df[data.columns]
        # weights_list.append(weights_df)
        # self.history_weights.

        # self.weight_predictions = self.weight_predictions.resample('D').ffill().dropna(how='all')
        # self.weight_predictions = self.weight_predictions.reindex(data.index).ffill().fillna(0)
        # self.weight_predictions = self.weight_predictions[(self.weight_predictions.index.date>=from_date) & (self.weight_predictions.index.date<=to_date)]

        # WeightAllocationModel.ticker_list =  self.ticker_list

        # self.weight_predictions = self.model.weights_allocate(from_date, to_date, ticker_list, data)

        # # adjusting weight_prediction frequency and indexing to our data.
        # self.weight_predictions = self.weight_predictions.resample('D').ffill().dropna(how='all')
        # self.weight_predictions = self.weight_predictions.reindex(data.index).ffill().fillna(0)
        # self.weight_predictions = self.weight_predictions[(self.weight_predictions.index.date>=from_date) & (self.weight_predictions.index.date<=to_date)]

    def get_allocations(self):
        return self.history_weights

    def date_data_needed(self, date_from, date_to=None):
        return self.model.date_data_needed(date_from, date_to)

    def __str__(self):
        return "Agent with weights allocation model: {}".format(str(self.model.__class__.__name__))

    def __hash__(self):
        return f"{str(self.model.__class__.__name__)}".__hash__()

    def sheet_name(self):
        return f"{str(self.model.__class__.__name__)}"