import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from fredapi import Fred
from functools import lru_cache

# Initialize FRED API
fred_api_key = "fbf3d5724e2f198249e17de383a5c3d1"  # Replace with your FRED API key
fred = Fred(api_key=fred_api_key)

@lru_cache(maxsize=128)
def fetch_stock_data(symbol, period='6mo'):
    """Fetch historical stock data using Yahoo Finance."""
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    if hist.empty:
        raise ValueError(f"No data found for {symbol}. Please check the symbol.")
    return hist

def fetch_options_data(symbol):
    """Fetch options chain data for a given symbol and calculate Greeks and moneyness."""
    ticker = yf.Ticker(symbol)
    expiration_dates = ticker.options
    options_dataset = pd.DataFrame()

    for expiration_date in expiration_dates:
        try:
            option_chain = ticker.option_chain(expiration_date)
            calls = option_chain.calls
            calls['expiration_date'] = expiration_date
            calls['Moneyness'] = calls['lastPrice'] / calls['strike']
            calls['time_to_expiry'] = (pd.to_datetime(expiration_date) - pd.Timestamp.now()).days / 365
            calls[['delta', 'gamma', 'theta', 'vega']] = calls.apply(oa.calculate_option_greeks, axis=1, result_type='expand')
            options_dataset = pd.concat([options_dataset, calls], ignore_index=True)
        except Exception as e:
            print(f"Error fetching options data for {expiration_date}: {e}")

    return options_dataset

def fetch_fundamental_data(symbol):
    """Fetch fundamental data such as balance sheet and key ratios."""
    stock = yf.Ticker(symbol)
    try:
        pe_ratio = stock.info.get('trailingPE', 'N/A')
        pb_ratio = stock.info.get('priceToBook', 'N/A')
        roe = stock.info.get('returnOnEquity', 'N/A')
        return {
            "P/E Ratio": pe_ratio,
            "P/B Ratio": pb_ratio,
            "ROE": roe
        }
    except Exception as e:
        print(f"Error fetching fundamental data: {e}")
        return {}
