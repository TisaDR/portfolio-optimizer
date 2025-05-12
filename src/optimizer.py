import pandas as pd
import numpy as np

def mean_variance_optimizer(prices: pd.DataFrame):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(mean_returns)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # Normalize

    return weights
