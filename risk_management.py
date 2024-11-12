import numpy as np
from scipy.stats import norm

def calculate_risk_metrics(hist):
    """Calculate risk metrics for stock data."""
    returns = hist['Close'].pct_change().dropna()
    if returns.empty:
        raise ValueError("Insufficient data to calculate returns for risk metrics.")

    mean = returns.mean()
    std_dev = returns.std()
    var = mean - norm.ppf(0.95) * std_dev
    sharpe_ratio = (returns.mean() - 0.01 / 252) / returns.std() * np.sqrt(252)

    report = f"Risk Metrics:\nValue-at-Risk (VaR): {var:.4f}\nSharpe Ratio: {sharpe_ratio:.4f}\n"
    return report
