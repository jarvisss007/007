import numpy as np
from scipy.stats import norm
import pandas as pd

# Function to calculate the Greeks of an option using the Black-Scholes model
def calculate_greeks(stock_price, strike_price, risk_free_rate, time_to_expiry, implied_volatility):
    """
    Calculate Greeks (Delta, Gamma, Theta, Vega) using the Black-Scholes model.

    Parameters:
    - stock_price: Current stock price.
    - strike_price: Strike price of the option.
    - risk_free_rate: Risk-free interest rate.
    - time_to_expiry: Time to expiry in years.
    - implied_volatility: Implied volatility of the option.

    Returns:
    - delta, gamma, theta, vega: The Greeks of the option.
    """
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate + 0.5 * implied_volatility**2) * time_to_expiry) / (
        implied_volatility * np.sqrt(time_to_expiry))
    d2 = d1 - implied_volatility * np.sqrt(time_to_expiry)

    # Delta Calculation
    delta = norm.cdf(d1)

    # Gamma Calculation
    gamma = norm.pdf(d1) / (stock_price * implied_volatility * np.sqrt(time_to_expiry))

    # Theta Calculation
    theta = -(stock_price * norm.pdf(d1) * implied_volatility) / (2 * np.sqrt(time_to_expiry)) \
            - risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)

    # Vega Calculation
    vega = stock_price * norm.pdf(d1) * np.sqrt(time_to_expiry)

    return delta, gamma, theta, vega

# Function to add calculated Greeks to the options DataFrame
def feature_engineering(option_data, stock_price, time_to_expiry, risk_free_rate=0.02):
    """
    Add calculated Greeks (Delta, Gamma, Theta, Vega) and moneyness as features to the options DataFrame.

    Parameters:
    - option_data: DataFrame containing options data.
    - stock_price: Current stock price.
    - time_to_expiry: Time to expiry in years.
    - risk_free_rate: The risk-free interest rate (default is 2%).

    Returns:
    - option_data: DataFrame with additional columns for Greeks and moneyness.
    """
    option_data['days_to_expiry'] = time_to_expiry * 365  # Convert years to days
    option_data['moneyness'] = stock_price / option_data['strike']  # Calculate moneyness

    # Calculate Greeks and add them to the DataFrame
    greeks = option_data.apply(
        lambda row: calculate_greeks(
            stock_price, row['strike'], risk_free_rate, time_to_expiry, row['impliedVolatility']
        ), axis=1, result_type='expand'
    )

    option_data[['delta', 'gamma', 'theta', 'vega']] = greeks

    return option_data

# Example usage
if __name__ == "__main__":
    # Example data for demonstration purposes
    data = {
        'strike': [100, 105, 110],
        'impliedVolatility': [0.2, 0.25, 0.3]
    }
    option_df = pd.DataFrame(data)
    stock_price = 100
    time_to_expiry = 0.5  # Half a year

    # Apply feature engineering
    option_df = feature_engineering(option_df, stock_price, time_to_expiry)
    print(option_df)
