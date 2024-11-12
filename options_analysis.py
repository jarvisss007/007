import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm


# Function to fetch options chain data for a given symbol and calculate Greeks
def analyze_options(symbol):
    """Fetch options chain data for a given symbol and calculate Greeks."""
    ticker = yf.Ticker(symbol)
    expiration_dates = ticker.options[:8]  # Limit to the next 8 expiration dates
    options_summary = []

    for expiration_date in expiration_dates:
        try:
            option_chain = ticker.option_chain(expiration_date)
            calls = option_chain.calls
            puts = option_chain.puts

            avg_iv_calls = calls['impliedVolatility'].mean()
            avg_iv_puts = puts['impliedVolatility'].mean()
            avg_iv = (avg_iv_calls + avg_iv_puts) / 2

            # Calculate Greeks for calls and puts
            calls['delta'], calls['gamma'], calls['theta'], calls['vega'] = calculate_greeks(calls)
            puts['delta'], puts['gamma'], puts['theta'], puts['vega'] = calculate_greeks(puts)

            # Append only the necessary summary data
            options_summary.append({
                "expiration_date": expiration_date,
                "avg_iv": avg_iv,
                "calls": calls[['strike', 'impliedVolatility', 'delta', 'gamma', 'theta', 'vega']],
                "puts": puts[['strike', 'impliedVolatility', 'delta', 'gamma', 'theta', 'vega']]
            })
        except Exception as e:
            print(f"Error fetching options data for expiration {expiration_date}: {e}")

    return options_summary


# Function to calculate Greeks (simplified Black-Scholes calculation)
def calculate_greeks(options_df):
    try:
        S = options_df['lastPrice']
        K = options_df['strike']
        T = options_df['expiration'] / 365  # Assuming `expiration` is in days
        r = 0.02  # Assuming a constant risk-free rate
        sigma = options_df['impliedVolatility']

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)

        return delta, gamma, theta, vega
    except Exception as e:
        print(f"Error calculating Greeks: {e}")
        return np.nan, np.nan, np.nan, np.nan


# Function to generate trading suggestions based on IV and Greeks
def generate_trading_suggestions(options_summary, hist):
    suggestions = ""

    for option_data in options_summary:
        avg_iv = option_data["avg_iv"]
        if avg_iv > 0.3:
            suggestions += f"For expiry {option_data['expiration_date']}: IV is high ({avg_iv:.2f}), consider selling strategies like iron condors or covered calls.\n"
        elif avg_iv < 0.2:
            suggestions += f"For expiry {option_data['expiration_date']}: IV is low ({avg_iv:.2f}), consider buying strategies like long calls or long puts.\n"

    # Example: Incorporate technical indicators into the suggestion
    if hist['RSI'].iloc[-1] > 70:
        suggestions += "RSI indicates overbought conditions, consider taking profits or initiating short positions.\n"
    elif hist['RSI'].iloc[-1] < 30:
        suggestions += "RSI indicates oversold conditions, consider buying opportunities.\n"

    return suggestions
