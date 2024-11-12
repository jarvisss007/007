# technical_indicators.py
import pandas as pd
import numpy as np

def calculate_indicators(hist):
    """Calculate technical indicators for the given stock data."""
    if hist.empty:
        return hist

    # Moving Averages
    hist['50_MA'] = hist['Close'].rolling(window=50).mean()
    hist['200_MA'] = hist['Close'].rolling(window=200).mean()

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

# Example usage:
if __name__ == "__main__":
    import data_fetching_analysis as dfa
    symbol = 'LRCX'
    period = '6mo'
    data = dfa.fetch_stock_data(symbol, period)
    updated_data = calculate_indicators(data)
    print(updated_data.tail())
