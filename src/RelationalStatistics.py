import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
class RelationalStatistics:
    """
    Module for relational statistic
    """
    def __init__(self, data: pd.DataFrame)-> None:
        """
        pd.DataFrame data: data of ticker returns
        """
        self.data = data

    def calc_standard_deviations(self) -> pd.DataFrame:
        """
        Generates standard deviations for class data
        """
        return self.data.std()

    def calc_variances(self) -> pd.DataFrame:
        """
        Generates variances for class data
        """
        return self.data.var()

    def calc_covariance_matrix(self)-> pd.DataFrame:
        """
        Generates the covariance matrix for class data
        """
        return self.data.cov()

    def calc_correlation_matrix(self)-> pd.DataFrame:
        """
        Generates the correlation matrix for class data
        """
        return self.data.corr()

    def calc_distance(self) -> pd.DataFrame:
        """
        Generates correlation distance matrix for the class data
        """
        # calculate the distance matrix
        distance_matrix = (0.5*(1- self.calc_correlation_matrix()))**0.5
        return distance_matrix

    def calc_eucledian_distance(self) -> pd.DataFrame:
        """
        Generates eucledian distance matrix for the class data
        """
        distance_matrix = self.calc_distance()
        euclidean_distance_matrix = euclidean_distances(distance_matrix.values)
        euclidean_distance_matrix = pd.DataFrame(euclidean_distance_matrix, index=distance_matrix.columns, columns=distance_matrix.columns)
        return euclidean_distance_matrix

    def calc_shrinkage_coefficient(self) -> float:
        """
        Generates shrinkage coefficient for the class data using heuristic.
        That is lambda = num of variables/ num of observations
        """
        return self.data.shape[1] / self.data.shape[0]

    # ---- PROBLEMATIC ----
    def calc_average_correlation(self) -> float:
        """
        Generate average correlation amongst variables in the class data
        """
        n = corr_matrix.shape[0]
        corr_matrix = self.calc_correlation_matrix() # exclude diagonal elements

        off_diagonal_sum = np.sum(corr_matrix) - np.sum(np.diag(corr_matrix))

        # compute the average correlation
        num_off_diagonal = n * (n - 1)  # total number of off-diagonal elements
        avg_corr = off_diagonal_sum / num_off_diagonal
        # return off_diagonal_sum, num_off_diagonal
        return avg_corr

    def calc_shrinkage_covariance(self) -> pd.DataFrame:
        """
        Generates the shrinkage covariance method
        """
        shrinkage_coefficient = self.calc_shrinkage_coefficient()
        sample_cov = self.calc_covariance_matrix()

        # set the diagonal of our constant correlation matrix to be equal to the diagonal of our covariance matrix
        identity = np.eye(sample_cov.shape[0])  # Identity matrix of same size
        shrinkage_cov = (1 - shrinkage_coefficient) * sample_cov + shrinkage_coefficient * identity
        return pd.DataFrame(shrinkage_cov, index=self.data.columns, columns=self.data.columns)
