from src.models.base import WeightAllocationModel


class Agent:

    def __init__(self, model: WeightAllocationModel):

        self.model = model
        self.ticker_list = None
        self.weight_predictions = None

    def weights_allocate(self, from_date, to_date, ticker_list, data):

        WeightAllocationModel.ticker_list =  self.ticker_list

        self.weight_predictions = self.model.weights_allocate(from_date, to_date, ticker_list, data)

    def date_data_needed(self, date_from, date_to=None):
        return self.model.date_data_needed(date_from, date_to)

    def __str__(self):
        return "Agent with weights allocation model: {}".format(str(self.model))

    def __hash__(self):
        return f"{str(self.model)} {str(self.strategy)}".__hash__()

    def sheet_name(self):
        return f"{str(self.model)}"