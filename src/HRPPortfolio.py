# Necessary Dependancies
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from src import RelationalStatistics


"""
This file contains the HRPPortfolio class. This class will have the following attributes:
1. 
"""

class HRPPortfolio():

    def __init__(self):
        self.stats_module = RelationalStatistics()

    def hierarchical_clustering(self):
        """
        Parameters
        ----------
        self

        Returns
        -------
        Correlation-Distance matrix
        """

        # Calculate the eucledian distance between the stocks - will be using centroid linkage method
        linkage_matrix = linkage(self.stats_module.fetch_eucledian_distance(), 'centroid')

        return linkage_matrix

    def quasi_diagonalization(self, cluster_order, covariance_matrix):
        '''Takes the linkage matrix and cluster order and orders the matrix so that 
        the highest correlations are along the diagonal'''
        matrix = covariance_matrix
        reordered_matrix = matrix[np.ix_(cluster_order, cluster_order)]
        return reordered_matrix
    
    def recursive_bisection(self, reordered_matrix):
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
    

    
        