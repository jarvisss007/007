import numpy as np

def backtest_strategy(hist):
    """Backtest moving average crossover strategy."""
    if 'MA10' not in hist.columns or 'MA50' not in hist.columns:
        raise ValueError("Missing required columns ('MA10' or 'MA50') for backtesting.")

    initial_balance = 10000
    balance = initial_balance
    position = 0
    transaction_cost = 10

    for i in range(1, len(hist)):
        if hist['MA10'].iloc[i] > hist['MA50'].iloc[i] and hist['MA10'].iloc[i - 1] <= hist['MA50'].iloc[i - 1]:
            position = balance / hist['Close'].iloc[i]
            balance = 0 - transaction_cost

        elif hist['MA10'].iloc[i] < hist['MA50'].iloc[i] and hist['MA10'].iloc[i - 1] >= hist['MA50'].iloc[i - 1]:
            balance = position * hist['Close'].iloc[i] - transaction_cost
            position = 0

    if position > 0:
        balance = position * hist['Close'].iloc[-1]

    profit = balance - initial_balance
    report = f"Backtesting Results:\nProfit: ${profit:.2f}\n"
    return report
