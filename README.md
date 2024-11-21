# HRP Portfolio Optimization

Members:

- Santiago Diaz Tolivia
- Oriol Ripalta I Maso
- Evi Prousanidou
- Francesco Leopardi

## Motivation

**Why not simply use MPT (Modern Portfolio Theory)?**
The MPT portfolio optimization model fails to perform well in real settings due to:

1. It involves the estimation of returns for a given set of assets. In real life, estimating returns is very difficult, and small errors in estimation can lead to sub-optimal performance.
2. Mean-variance optimization methods involve the inversion of a covariance matrix for a set of assets. This matrix inversion makes the algorithm susceptible to market volatility and can heavily change the results for small changes in the correlations.

The HRP model solves some of these problems.

# Data
We will be using data from Polygon and/or Yahoo Finance through their proprietary APIs

# Choosing Assets in the Portfolio
100 assets will be chosen arbitrarily from the S&P500 based on data availability. Large cap firms will be prioritized

# HRP

The HRP portfolio optimization model can be divided into three parts:

1. `Hierarchical Clustering` which breaks down assets into hierarchical clusters
    - This is done by calculating the eucledian distance between the asset's returns. This is done by calculating the covariance matrix among the returns of all the assets and applying the eucledian distance formula
    - Then we use a linkage method to cluster the firms that are closely correlated
    - We clustered all our assets into a hierarchical tree based on similarity defined through our chosen distance measure (eucledian distance)

2. `Quasi-Diagonalization` which reorganizes the covariance matrix by placing similar assets together
    - In this step we rearrange columns and rows of the covariance matrix to be able to have highly covaried assets on the diagonal and dissimilar assets further apart.
3. `Recursive Bisection` where weights are assigned to each asset in our portfolio
    - The recursive bisection method will break every cluster in the quasi-diagonalized matrix into smaller sub-clusters. We start with the largest cluster, and use the assumption that the inverse-variance allocation of the matrix is the most optimal allocation for the portfolio


![HRP Dendogram](https://hudsonthames.org/wp-content/uploads/2020/06/dendrogram.png "HRP Dendogram")

## SETUP

*pip install -r requirements.txt*

## Backtesting Infrastructure 

1) A backtesting infrastructure for comparing and evaluating different porfolio optimization strategies against benchmarks.
1) Backcasting requires -> agents that will participate in the simulation. In order for agents to participate in the simulation they need a prediction(Weight-Allocation) model, which is responsible for predicting the necessary values/weights.
2) The current implementation doesn't save the weights after the simulation is over. It outputs the simulation results against the benchmarks we want to show.

* You can add benchmarks in the runner.py as follows: benchmarks = [b.PNL('P'),b.Sharpe('P')] denoting, the metric and its frequency.
* You can add Agents, that inherit a model, and backtest them with the Backtester Class.
* See and run runner.py

<span style="color:red">Explain different methodologies for clustering!</span>
