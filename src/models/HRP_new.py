import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
from src.models.RelationalStatistics import RelationalStatistics
from scipy.spatial.distance import pdist


class HRPOptimizer:
    """
    A HRPOpt object (inheriting from BaseOptimizer) constructs a hierarchical
    risk parity portfolio based on the input assets' returns.
    
    The HRPOpt object has the following methods:
    
    - cluster_variance
    - quasi-diagonalization
    - raw_hrp_allocation
    - optimize
    """

    def __init__(self, returns: pd.DataFrame = None, cov_matrix: pd.DataFrame = None):
        """    
        Parameters
        ----------
        returns : pd.DataFrame
            A pandas DataFrame of asset returns.
        cov_matrix : pd.DataFrame
            A pandas DataFrame of asset covariance matrix.
        ----------

        returns ValueError in case returns and cov_matrix are both None.
        returns TypeError in case returns is not a pd.DataFrame.
        """

        # Initialize the returns, covariance matrix, and statistics module
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.stats_module = RelationalStatistics(returns)
        
        # Check if returns or cov_matrix is provided
        if self.returns is None and self.cov_matrix is None:
            raise ValueError("Provide returns or cov_matrix. At least one of them should be provided.")
        
        # Check if returns is a pd.DataFrame
        if returns is not None and not isinstance(returns, pd.DataFrame):
            raise TypeError("Returns should be in a pd.DataFrame format.")
    
        
        if self.returns is not None:
            tickers = self.returns.columns
        else:
            tickers = self.cov_matrix.columns

    def cluster_variance(self, covariance_matrix: pd.DataFrame, cluster_items: list) -> float:
        """
        Compute the variance per cluster.
        
        Parameters
        ----------
        cov_matrix : pd.DataFrame
            A pandas DataFrame of asset covariance matrix.

        cluster_items : list
            a list of the ticker names within the cluster to compute the variance for.
        
        -------
        float
        returns the variance.
        """
        # Compute the cluster covariance matrix
        cluster_cov = covariance_matrix.loc[cluster_items, cluster_items]

        # Compute the inverse of the weights
        weights = 1 / np.diag(cluster_cov) 

        # Normalize the weights
        weights /= weights.sum()

        # Compute the variance
        return weights @ cluster_cov @ weights
    
    def quasi_diagonalization(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the quasi-diagonal matrix (Sorts clustered items by eucledian distance).
        It rearranges the rows and columns of a covariance (or correlation) matrix such 
        that related assets (those within the same cluster) are grouped together.

        Parameters
        ----------
        cov_matrix : pd.DataFrame
            A pandas DataFrame of asset covariance matrix.
        
        -------
        float
        returns the variance.
        """

        # Compute the eucledian distance matrix, take upper triangle and convert to list
        pdistance = pdist(cov_matrix, 'euclidean')

        # Compute linkage matrix
        linkage_matrix = linkage(pdistance, 'single')

        # Sort clustered items by eucledian distance
        return sch.to_tree(linkage_matrix, rd=False).pre_order()
    
    def raw_hrp_allocation(self, cov_matrix: pd.DataFrame, tickers: list) -> pd.DataFrame:
        """
        Compute the weights of the HRP portfolio. It does so through various steps.
        The first is recursive bisection, which clusters assets into a hierarchical tree.
        The algorithm will then run the cluster_variance function to compute the variance per cluster.
        This will allow the algorithm to optimize based on variance.
        
        
        Parameters
        ----------
        cov_matrix : pd.DataFrame
            A pandas DataFrame of asset covariance matrix.

        tickers : list
            a list of the ticker names to compute the weights for.
        
        -------
        float
        returns the weights.
        """

        # Create a series of ones with the tickers as the index (equal-weights)
        w = pd.Series(1, index=tickers)

        # Create a list of the tickers called cluster_items
        cluster_items = [tickers]  

        # Throughout the process the length of the cluster must be greater than 0
        while len(cluster_items) > 0:
            
            # Create clusters by splitting the cluster_items in half
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]  
            print(cluster_items)
            # For each pair, optimize locally.
            for i in range(0, len(cluster_items), 2):
                first_cluster = cluster_items[i]
                #print(first_cluster)
                second_cluster = cluster_items[i + 1]
                # Form the inverse variance portfolio for this pair
                first_variance = HRPOptimizer.cluster_variance(cov_matrix, first_cluster)
                second_variance = HRPOptimizer.cluster_variance(cov_matrix, second_cluster)
                alpha = 1 - first_variance / (first_variance + second_variance)
                w[first_cluster] *= alpha  # weight 1
                w[second_cluster] *= 1 - alpha  # weight 2
        return w




     


