import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_indicators(hist):
    """Calculate technical indicators for stock data."""
    if hist.empty:
        raise ValueError("Historical data is empty. Cannot calculate indicators.")

    # Moving Averages
    hist['MA10'] = hist['Close'].rolling(window=10).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    if hist['MA10'].isna().all() or hist['MA50'].isna().all():
        raise ValueError("Insufficient data to calculate moving averages (MA10, MA50). Please try a longer period.")

    # RSI
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()

    return hist

def plot_indicators(hist, ax):
    """Plot technical indicators for stock data."""
    ax.plot(hist.index, hist['Close'], label='Close Price', color='blue')
    if 'MA10' in hist.columns:
        ax.plot(hist.index, hist['MA10'], label='MA10', color='green')
    if 'MA50' in hist.columns:
        ax.plot(hist.index, hist['MA50'], label='MA50', color='red')
    ax.set_title('Technical Indicators')
    ax.legend()
