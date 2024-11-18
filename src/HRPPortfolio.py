# Necessary Dependancies
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from src import RelationalStatistics
from src.RelationalStatistics import RelationalStatistics
from typing import List
import matplotlib as plt
import seaborn as sns


class HRPPortfolio:
    """
    Portfolio optimization using Hierarchical Risk Parity (HRP)
    """
    def __init__(self, data):
        """
        pd.DataFrame data: data of ticker returns
        class RelationalStatistics: module for relational statistics
        """
        self.data = data
        self.stats_module = RelationalStatistics(data)

    def hierarchical_clustering(self) -> np.ndarray:
        """
        Performs hierarchical clustering on the euclidean distance matrix.
        """
        eucledian_df = self.stats_module.calc_eucledian_distance()
        # calculate the eucledian distance between the ticker
        linkage_matrix = linkage(eucledian_df, 'centroid')
        return linkage_matrix
    
    def get_cluster_order(self, linkage_matrix):
        '''Takes the linkage matrix and returns the cluster order'''

    def get_cluster_order(self) -> List[str]:
        """
        Gets the cluster order from the hierarchical clustering.
        """
        n = self.data.shape[1]  # total number of assets
        linkage_matrix = self.hierarchical_clustering()
        merged_assets = []  # assets
        merged_set = set()  # cache for assets

        for i in range(len(linkage_matrix)): # iterate through the linkage matrix
            left, right = int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1])

            # add one ticker to the merged list if not merged already
            if left < n and left not in merged_set:
                merged_assets.append(left)
                merged_set.add(left)

            if right < n and right not in merged_set:
                merged_assets.append(right)
                merged_set.add(right)

        return merged_assets


    def quasi_diagonalization(self) -> pd.DataFrame:
        """
        Use the linkage matrix and cluster order to quasi-diagonalize the covariance matrix.
        Such that the highest correlations are along the diagonal.
        """
        cluster_order = self.get_cluster_order()

        # ensure cluster_order is a 1D array, otherwise flatten
        cluster_order = np.ravel(cluster_order)
        cov_matrix = self.stats_module.calc_covariance_matrix()

        if isinstance(cov_matrix, pd.DataFrame): # ensure cov matrix is a pd.DataFrame
            reordered_matrix = cov_matrix.iloc[cluster_order, cluster_order].values
        else : # else access values using np.ix
            reordered_matrix = cov_matrix[np.ix_(cluster_order, cluster_order)]

        return reordered_matrix
    
    # ---- PROBLEMATIC ----
    def hrp_recursive_bisection(self, reordered_matrix, cluster_order, linkage_matrix, merged_clusters = None):
        """
        Perform recursive bisection on the given ordered matrix and outputs the weights
        """

        # Initialize merged_clusters dictionary if not provided
        if merged_clusters is None:
            merged_clusters = {}

        n = len(cluster_order)

        # Base case: Single asset
        if n == 1:
            return {cluster_order[0]: 1}

        # Current node from the linkage matrix
        node = linkage_matrix[len(linkage_matrix) - n]  # Get corresponding node
    
        left_child, right_child = int(node[0]), int(node[1])

        # Get clusters (base assets or merged clusters)
        if left_child < len(cluster_order):
            left_cluster = [cluster_order[left_child]]
        else:
            left_cluster = merged_clusters[left_child]

        if right_child < len(cluster_order):
            right_cluster = [cluster_order[right_child]]
        else:
            right_cluster = merged_clusters[right_child]

        # Calculate covariances
        cov_left = reordered_matrix[np.ix_(left_cluster, left_cluster)]
        cov_right = reordered_matrix[np.ix_(right_cluster, right_cluster)]

        # Inverse-variance allocation
        var_left = np.trace(np.linalg.inv(cov_left))
        var_right = np.trace(np.linalg.inv(cov_right))

        alpha_left = 1 - (var_right / (var_left + var_right))
        alpha_right = 1 - alpha_left

        # Recursive bisection
        weights_left = self.hrp_recursive_bisection(reordered_matrix, left_cluster, linkage_matrix, merged_clusters)
        weights_right = self.hrp_recursive_bisection(reordered_matrix, right_cluster, linkage_matrix, merged_clusters)

        # Scale the weights by alpha
        for asset in weights_left:
            weights_left[asset] *= alpha_left
        for asset in weights_right:
            weights_right[asset] *= alpha_right

        # Merge clusters and update merged_clusters dictionary
        merged_clusters[n + len(linkage_matrix)] = left_cluster + right_cluster
        weights = {**weights_left, **weights_right}

        return weights
        
    def plot_dendrogram(self):
        """ Plot the dendrogram based on hierarchical clustering. """
        linkage_matrix = self.hierarchical_clustering()
        return linkage_matrix

