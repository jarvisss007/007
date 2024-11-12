import logging
import yfinance as yf
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import numpy as np

# Configure logging to include all details
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to calculate option Greeks
def calculate_option_greeks(stock_price, strike_price, risk_free_rate, time_to_expiry, implied_volatility):
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate + 0.5 * implied_volatility ** 2) * time_to_expiry) / (
            implied_volatility * np.sqrt(time_to_expiry))
    d2 = d1 - implied_volatility * np.sqrt(time_to_expiry)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (stock_price * implied_volatility * np.sqrt(time_to_expiry))
    theta = -(stock_price * norm.pdf(d1) * implied_volatility) / (2 * np.sqrt(time_to_expiry)) - risk_free_rate * strike_price * np.exp(
        -risk_free_rate * time_to_expiry) * norm.cdf(d2)
    vega = stock_price * norm.pdf(d1) * np.sqrt(time_to_expiry)

    return delta, gamma, theta, vega

# Function to calculate technical indicators
def calculate_technical_indicators(hist):
    if hist.empty:
        return hist

    hist['50_MA'] = hist['Close'].rolling(window=50).mean()
    hist['200_MA'] = hist['Close'].rolling(window=200).mean()
    hist['RSI'] = calculate_rsi(hist['Close'])
    hist['BB_Upper'], hist['BB_Lower'] = calculate_bollinger_bands(hist['Close'])
    return hist

# Function to calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(series, window=20, num_std_dev=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

# Function to calculate Sharpe Ratio for backtesting
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

# Optimize trading strategy using RandomForestRegressor
def optimize_strategy(stock_data):
    def objective(trial):
        # Define hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        # Prepare training data
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'impliedVolatility']
        if 'impliedVolatility' not in stock_data.columns:
            raise ValueError("Stock data must contain 'impliedVolatility' column for optimization.")

        X = stock_data[features]
        y = stock_data['Close'].shift(-1).ffill()  # Updated to avoid deprecation warning

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        model.fit(X_train, y_train)

        # Predict and calculate MSE
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Calculate average implied volatility for logging
        avg_iv = stock_data['impliedVolatility'].mean()

        # Log all metrics including MSE, Greeks, and technical indicators
        logging.info(
            f"Trial completed: n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}, Avg IV={avg_iv:.4f}, MSE={mse:.4f}"
        )

        return mse

    # Running optimization using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Log the best parameters found
    best_params = study.best_params
    avg_iv = stock_data['impliedVolatility'].mean()  # Calculate average IV for the entire study
    logging.info(f"Best parameters found: {best_params}")
    logging.info(f"Best MSE value achieved: {study.best_value:.4f}")
    logging.info(f"Average IV during study: {avg_iv:.4f}")

    return study

# Example code to fetch data, calculate Greeks, technical indicators, and optimize
def main():
