# HRP Portfolio Optimization

Members:

- Santiago Diaz Tolivia
- Oriol Ripalta I Maso
- Lefteris Nazos
- Evi Paraskevi Prousanidi
- Fransesco Leopaldi XVII

## SETUP

*pip install -r requirements.txt*

## Backtesting Infrastructure 

1) A backtesting infrastructure for comparing and evaluating different porfolio optimization strategies against benchmarks.
1) Backcasting requires -> agents that will participate in the simulation. In order for agents to participate in the simulation they need a prediction(Weight-Allocation) model, which is responsible for predicting the necessary values/weights.
2) The current implementation doesn't save the weights after the simulation is over. It outputs the simulation results against the benchmarks we want to show.

* You can add benchmarks in the runner.py as follows: benchmarks = [b.PNL('P'),b.Sharpe('P')] denoting, the metric and its frequency.
* You can add Agents, that inherit a model, and backtest them with the Backtester Class.
* See and run runner.py

## Motivation

**Why not simply use MPT (Modern Portfolio Theory)?**
The MPT portfolio optimization model fails to perform well in real settings due to:

1. It involves the estimation of returns for a given set of assets. In real life, estimating returns is very difficult, and small errors in estimation can lead to sub-optimal performance.
2. Mean-variance optimization methods involve the inversion of a covariance matrix for a set of assets. This matrix inversion makes the algorithm susceptible to market volatility and can heavily change the results for small changes in the correlations.

The HRP model solves some of these problems.

# HRP

The HRP portfolio optimization model can be divided into three parts:

1. `Hierarchical Clustering` which breaks down assets into hierarchical clusters
2. `Quasi-Diagonalization` which reorganizes the covariance matrix by placing similar assets together
3. `Recursive Bisection` where weights are assigned to each asset in our portfolio

![HRP Dendogram](https://hudsonthames.org/wp-content/uploads/2020/06/dendrogram.png "HRP Dendogram")

<span style="color:red">Explain different methodologies for clustering!</span>
