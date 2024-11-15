import numpy as np
import pandas as pd
from base import WeightAllocationModel

# TODO: understand this functions and clean them up
class HRP(WeightAllocationModel):
    def __init__(self):
        pass

    def getQuasiDiag(self, link):
        """
        A 'quasi-diagonalization' is a process usually known as matrix seriation and which can
        be performed using hierarchical clustering. This process reorganize the covariance matrix so
        similar investments will be placed together. This matrix diagonalization allow us to distribute
        weights optimally following an inverse-variance allocation.
        """
        # sort clustered items by distance
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items

        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = pd.concat([sortIx, df0]) # sortIx.append(df0)  # item 2
            sortIx = sortIx.sort_index()  # re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    def getClusterVar(self, cov,cItems):
        # Compute variance per cluster
        cov_=cov.loc[cItems,cItems] # matrix slice
        w_=getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar



    def getRecBipart(self, cov, sortIx):
        # Compute HRP alloc
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]  # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = self.getClusterVar(cov, cItems0)
                cVar1 = self.getClusterVar(cov, cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w
