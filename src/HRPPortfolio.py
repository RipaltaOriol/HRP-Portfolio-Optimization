# Necessary Dependancies
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances


"""
This file contains the HRPPortfolio class. This class will have the following attributes:
1. 
"""

class HRPPortfolio:

    def __init__(self, correlation_matrix):
        self.correlation_matrix = correlation_matrix
    
    def distance_matrix(self, correlation_matrix):
        """
        Parameters
        ----------
        correlation_matrix : pd.DataFrame 
            Datafrae containing the correlation matrix of the assets

        Returns
        -------
        Correlation-Distance matrix
        """
        distance = (0.5*(1- correlation_matrix))**0.5
        return distance
    
    def eucledian_distance_matrix(self, correlation_matrix):
        """
        Parameters
        ----------
        correlation_matrix : pd.DataFrame 
            Datafrae containing the correlation matrix of the assets

        Returns
        -------
        Correlation-Distance matrix
        """

        # Calculate the distance matrix
        distance = self.distance_matrix(correlation_matrix) # Possible redundancy, distance matrix is being calculated twice

        # Calculate the eucledian distance matrix
        euclidean_distance = euclidean_distances(distance.values)

        return euclidean_distance 

    def hierarchical_clustering(self):
        return linkage_matrix

    def quasi_diagonalization(self, linkage_matrix):
        return clustered_correlations
    
    def recursive_bisection(self, clustered_correlations):
        return weights

    

    
        