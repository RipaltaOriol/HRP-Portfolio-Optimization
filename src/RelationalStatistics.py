import pandas as pd

class RelationalStatistics():

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
    
    def fetch_shrinkage_covariance(self, shrinkage=0.1) -> pd.Data:
        """
        Fetches shrinkage covariance matrix using Ledoit-Wolf shrinkage method.
        
        :param shrinkage: Shrinkage coefficient (default: 0.1)
        :return: Shrinkage covariance matrix as a DataFrame
        """
        sample_cov = self.data.cov().values  # Sample covariance matrix
        identity = np.eye(sample_cov.shape[0])  # Identity matrix of same size
        shrinkage_cov = (1 - shrinkage) * sample_cov + shrinkage * identity
        return pd.DataFrame(shrinkage_cov, index=self.data.columns, columns=self.data.columns)