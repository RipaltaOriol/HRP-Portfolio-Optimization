import os
import pandas as pd

class WeightAllocationModel:

    ticker_list = None
    save = False

    def __init__(self):
        pass

    def weights_allocate(self, date_from, date_to, ticker_list, **params):
        """
        This is where all predictions are made. The predictions must be returned in a DataFrame format and specifically,
        having as index date datetime, columns the ticker names, and values the weights for each ticker for the current date
        :rtype: object
        :param date_from: First day of predictions.
        :param date_to: Last day of predictions.
        :param params: Left empty, for future development.
        :return: Predictions with the specified format.
        """
        raise NotImplementedError('Every Model must implement its own predict function.')

    def __str__(self):
        """
        Returns models' name with the specified parameters to be used when extracting results.
        :return: str name of model
        """

        return "Weight-Allocation"

    def date_data_needed(self, date_from, date_to):
        """
        :param date_from: Starting day of the simulation
        :param date_to: Last day of the simulation
        :return: Returns the date that the model needs data from
        """

        return NotImplementedError('Every model must implement its own date_data_needed method.')










