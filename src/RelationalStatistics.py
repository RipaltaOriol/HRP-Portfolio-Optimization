import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
class RelationalStatistics:

    def __init__(self, data)-> None:
        """
            Constructor of Relational Statistic - dataframe data as input
        """
        self.data = data

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
    
    def fetch_shrinkage_covariance(self, shrinkage=0.1) -> pd.DataFrame:
        """
        Fetches shrinkage covariance matrix using Ledoit-Wolf shrinkage method.
        
        :param shrinkage: Shrinkage coefficient (default: 0.1)
        :return: Shrinkage covariance matrix as a DataFrame
        """
        sample_cov = self.data.cov().values  # Sample covariance matrix
        identity = np.eye(sample_cov.shape[0])  # Identity matrix of same size
        shrinkage_cov = (1 - shrinkage) * sample_cov + shrinkage * identity
        return pd.DataFrame(shrinkage_cov, index=self.data.columns, columns=self.data.columns)