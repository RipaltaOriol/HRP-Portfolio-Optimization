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

    def __init__(self, correlation_matrix, covariance_matrix):
        self.covariance_matrix = covariance_matrix
        self.corelation_matrix = correlation_matrix
    
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

    def quasi_diagonalization(self, cluster_order, covariance_matrix):
        '''Takes the linkage matrix and cluster order and orders the matrix so that 
        the highest correlations are along the diagonal'''
        matrix = covariance_matrix
        reordered_matrix = matrix[np.ix_(cluster_order, cluster_order)]
        return reordered_matrix
    
    def hrp_recursive_bisection(self, reordered_matrix, cluster_order, linkage_matrix):
        n = len(cluster_order)

        if n == 1:
            return {cluster_order[0]: 1}
        
        node = linkage_matrix[n - 2]

        left_child = [int(node[0])]
        right_child = [int(node[1])]

        left_cluster = [cluster_order[i] for i in left_child]
        right_cluster = [cluster_order[i] for i in right_child]

        cov_left = reordered_matrix[np.ix_(left_cluster, left_cluster)]
        cov_right = reordered_matrix[np.ix_(right_cluster, right_cluster)]

        inv_cov_left = np.linalg.inv(cov_left)
        inv_cov_right = np.linalg.inv(cov_right)

        var_left = np.trace(inv_cov_left)
        var_right = np.trace(inv_cov_right)

        alpha_left = 1 - (var_right / (var_left + var_right))
        alpha_right = 1 - alpha_left

        weights_left = self.hrp_recursive_bisection(reordered_matrix, left_cluster, linkage_matrix)
        weights_right = self.hrp_recursive_bisection(reordered_matrix, right_cluster, linkage_matrix)

        for assets in weights_left:
            weights_left[assets] *= alpha_left
        for assets in weights_right:
            weights_right[assets] *= alpha_right
        
        weights = {**weights_left, **weights_right}

        return weights



# """
# Parameters
# ----------
# input1 : dtype 
#     what is it?
# input2 : dtype 
#     what is it?
# input3 : dtype 
#     what is it?

# Returns
# -------
# What does the function return.
# """
    

    
        