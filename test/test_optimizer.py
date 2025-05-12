import pandas as pd
from src.optimizer import mean_variance_optimizer

def test_weights_sum_to_one():
    data = {
        'AAPL': [150, 152, 154, 156],
        'GOOG': [2700, 2720, 2735, 2750]
    }
    df = pd.DataFrame(data)
    weights = mean_variance_optimizer(df)
    assert abs(sum(weights) - 1) < 1e-6
