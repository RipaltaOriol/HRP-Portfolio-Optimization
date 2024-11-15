import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
class RelationalStatistics():

    def __init__(self, data)-> None:
        """
            Constructor of Relational Statistic - dataframe data as input
        """
        self.data = data

    def fetch_standard_deviations(self) -> pd.DataFrame:
        """
            fetches standard deviations
        """
        return self.data.std()
    
    def fetch_variances(self) -> pd.DataFrame:
        """
            fetches variances
        """
        return self.data.var()
    
    def fetch_covariance_matrix(self)-> pd.DataFrame:
        """
            Fetches covariance matrix
        """
        return self.data.cov()
    
    def fetch_correlation_matrix(self)-> pd.DataFrame:
        """
            Fetches correlation matrix
        """
        return self.data.corr()
    

    def fetch_distance(self) -> pd.DataFrame:
        """
        Parameters
        ----------
        self

        Returns
        -------
        Correlation-Distance matrix
        """

        # Calculate the distance matrix
        distance_matrix = (0.5*(1- self.fetch_correlation_matrix()))**0.5

        return distance_matrix
    
    def fetch_eucledian_distance(self):
        """
        Parameters
        ----------
        self

        Returns
        -------
        Correlation-Distance matrix
        """

        # Calculate the distance matrix
        distance_matrix = self.fetch_distance() # Possible redundancy, distance matrix is being calculated twice

        # Calculate the eucledian distance matrix
        euclidean_distance_matrix = euclidean_distances(distance_matrix.values)

        # Convert to pandas dataframe
        euclidean_distance_matrix = pd.DataFrame(euclidean_distance_matrix, index=distance_matrix.columns, columns=distance_matrix.columns)

        return euclidean_distance_matrix
    
    def fetch_shrinkage_coefficient(self) -> float:
        """
            calculate shrinkage coefficient by heiristic: lamda = num of variables/ num of observations
        """
        return self.data.shape[1] / self.data.shape[0]
    


############################################ PROBLEMATIC ######################################
    def fetch_average_correlation(self):
        """
            calculate average correlation
        """
        # Exclude diagonal elements
        corr_matrix = self.fetch_correlation_matrix()
        n = corr_matrix.shape[0]
        off_diagonal_sum = np.sum(corr_matrix) - np.sum(np.diag(corr_matrix))
        
        # Compute the average correlation
        num_off_diagonal = n * (n - 1)  # Total number of off-diagonal elements
        avg_corr = off_diagonal_sum / num_off_diagonal
        # return off_diagonal_sum, num_off_diagonal
        return avg_corr
    
    def fetch_shrinkage_covariance(self) -> pd.DataFrame:
        """
        Fetches shrinkage covariance matrix using method above
        Fetches shrinkage coefficient from method above
        Constructs constant correlation matrix

        Returns shrinkage covariance matrix
        """
        # get shrinkage coefficient from method
        shrinkage_coefficient = self.fetch_shrinkage_coefficient()

        # get sample covariance matrix
        sample_cov = self.fetch_covariance_matrix()

        # set the diagonal of our constant correlation matrix to be equal to the diagonal of our covariance matrix


        identity = np.eye(sample_cov.shape[0])  # Identity matrix of same size
        shrinkage_cov = (1 - shrinkage_coefficient) * sample_cov + shrinkage_coefficient * identity
        return pd.DataFrame(shrinkage_cov, index=self.data.columns, columns=self.data.columns)
    

