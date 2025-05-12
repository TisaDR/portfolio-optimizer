# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📈 Portfolio Optimization (Mean-Variance)")

uploaded_file = st.file_uploader("Upload a CSV file with stock prices", type=["csv"])

if uploaded_file:
    prices = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(prices.head())

    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    num_assets = len(mean_returns)
    risk_free_rate = 0.01

    def portfolio_performance(weights):
        returns = np.dot(weights, mean_returns) * 252
        std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = (returns - risk_free_rate) / std_dev
        return returns, std_dev, sharpe

    def neg_sharpe(weights):
        return -portfolio_performance(weights)[2]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    opt_weights = result.x
    exp_return, volatility, sharpe = portfolio_performance(opt_weights)

    st.subheader("📊 Optimal Portfolio Weights")
    weights_df = pd.DataFrame({"Ticker": mean_returns.index, "Weight": opt_weights})
    st.dataframe(weights_df)

    st.write(f"**Expected Annual Return:** {exp_return:.2%}")
    st.write(f"**Annual Volatility:** {volatility:.2%}")
    st.write(f"**Sharpe Ratio:** {sharpe:.2f}")

    st.subheader("📉 Portfolio Allocation")
    fig, ax = plt.subplots()
    ax.pie(opt_weights, labels=mean_returns.index, autopct="%1.1f%%")
    st.pyplot(fig)
