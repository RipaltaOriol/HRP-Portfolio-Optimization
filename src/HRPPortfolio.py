# Necessary Dependancies
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from src import RelationalStatistics
from src.RelationalStatistics import RelationalStatistics



"""
This file contains the HRPPortfolio class. This class will have the following attributes:
1. 
"""

class HRPPortfolio:

    def __init__(self, data):
        self.data = data
        self.stats_module = RelationalStatistics(data)

    def hierarchical_clustering(self):
        """
        Parameters
        ----------
        self

        Returns
        -------
        Correlation-Distance matrix
        """
        eucledian_df = self.stats_module.fetch_eucledian_distance()

        # Calculate the eucledian distance between the stocks - will be using centroid linkage method
        linkage_matrix = linkage(eucledian_df, 'centroid')

        return linkage_matrix
    
    def get_cluster_order(self):
        '''Takes the linkage matrix and returns the cluster order'''


        linkage_matrix = self.hierarchical_clustering()
        n = self.data.shape[1]  # Total number of original assets based on columns in self.data
        merged_assets = []  # List to track original assets that are merged
        merged_set = set()  # Set to track already merged assets

        # Step through each merge (each row of the linkage matrix)
        for i in range(len(linkage_matrix)):
            left, right = int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1])

            # Only add the original assets to the merged list if they haven't already been merged
            if left < n and left not in merged_set:
                merged_assets.append(left)
                merged_set.add(left)

            if right < n and right not in merged_set:
                merged_assets.append(right)
                merged_set.add(right)

        return merged_assets
        

    def quasi_diagonalization(self):
        '''Takes the linkage matrix and cluster order and orders the matrix so that 
        the highest correlations are along the diagonal'''
        cluster_order = self.get_cluster_order()

        # Step 2: Ensure cluster_order is a 1D array or list (flatten if necessary)
        cluster_order = np.ravel(cluster_order)
    
        # Step 3: Fetch the covariance matrix
        matrix = self.stats_module.fetch_covariance_matrix()
    
        # Step 4: Check if the covariance matrix is a pandas DataFrame
        if isinstance(matrix, pd.DataFrame):
            # Use .iloc to index the DataFrame
            reordered_matrix = matrix.iloc[cluster_order, cluster_order].values
        else:
            # If it's a numpy array, use np.ix_ for indexing
            reordered_matrix = matrix[np.ix_(cluster_order, cluster_order)]
    
        return reordered_matrix
    
    def hrp_recursive_bisection(self, reordered_matrix, cluster_order, linkage_matrix, merged_clusters):
        '''This function performs the recursive bisection method on the given ordered matrix and outputs the weights'''
        n = len(cluster_order)

        if n == 1:
            return {cluster_order[0]: 1}

        # Get the node from the linkage matrix (this is the current bisection step)
        node = linkage_matrix[n - 2]  # n-2 because linkage matrix has n-1 merges

        left_child = int(node[0])
        right_child = int(node[1])

        # If the indices are less than the number of original assets, they refer directly to assets
        if left_child < len(cluster_order):
            left_cluster = [cluster_order[left_child]]
        else:
            # Recursively get the cluster members for the merged cluster
            left_cluster = merged_clusters[left_child]

        if right_child < len(cluster_order):
            right_cluster = [cluster_order[right_child]]
        else:
            # Recursively get the cluster members for the merged cluster
            right_cluster = merged_clusters[right_child]

        # Covariance matrices for the left and right clusters
        cov_left = reordered_matrix[np.ix_(left_cluster, left_cluster)]
        cov_right = reordered_matrix[np.ix_(right_cluster, right_cluster)]

        # Inverse covariance matrices for left and right
        inv_cov_left = np.linalg.inv(cov_left)
        inv_cov_right = np.linalg.inv(cov_right)

        # Variance of the left and right clusters
        var_left = np.trace(inv_cov_left)
        var_right = np.trace(inv_cov_right)

        # The alpha values for allocation
        alpha_left = 1 - (var_right / (var_left + var_right))
        alpha_right = 1 - alpha_left

        # Recursive call to the left and right children (subclusters)
        weights_left = self.hrp_recursive_bisection(reordered_matrix, left_cluster, linkage_matrix, merged_clusters)
        weights_right = self.hrp_recursive_bisection(reordered_matrix, right_cluster, linkage_matrix, merged_clusters)

        # Scale the weights by alpha
        for asset in weights_left:
            weights_left[asset] *= alpha_left
        for asset in weights_right:
            weights_right[asset] *= alpha_right
    
        # Combine left and right weights into a single dictionary
        weights = {**weights_left, **weights_right}

        # Update the merged clusters dictionary with the current merge
        merged_clusters[len(cluster_order)] = left_cluster + right_cluster  # Store the merged cluster assets

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
    

    
        