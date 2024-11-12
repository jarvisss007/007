import tkinter as tk
from ctypes import py_object
import 1.py
import 2.py
import 3.py
import 4.py
import 5.py
import 6.py
from tkinter import scrolledtext, messagebox
from data_fetching import fetch_stock_data, get_weekly_expirations, get_fed_funds_rate
from technical_analysis import calculate_indicators
from technical_indicators import backtest_strategy
from suggestions import generate_suggestions
from risk_management import build_lstm_model, reduce_features
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# GUI to Display Reports and Charts
def show_report():
    symbol = entry.get()
    period = period_var.get()
    try:
        # Step 1: Fetch Data
        hist = fetch_stock_data(symbol, period)

        if hist.empty:
            messagebox.showerror("Error", f"No data found for {symbol}. Please check the symbol.")
            return

        # Step 2: Calculate Indicators
        hist = calculate_indicators(hist)

        # Step 3: Backtest Strategy
        backtest_profit, sharpe_ratio = backtest_strategy(hist)

        # Step 4: Fetch Weekly IV Data and Macroeconomic Data
        weekly_iv_data = get_weekly_expirations(symbol)
        fed_funds_rate = get_fed_funds_rate()

        # Step 5: Generate Actionable Suggestions
        suggestions = generate_suggestions(weekly_iv_data, hist)

        # Step 6: Generate Report
        report = f"""
        Stock Report for {symbol}:

        **Technical Analysis:**
        - Moving Average Cross: {'Golden Cross (Bullish)' if hist['Golden_Cross'].iloc[-1] else 'Death Cross (Bearish)'}
        - MACD Signal: {'Bullish' if hist['MACD'].iloc[-1] > hist['Signal_Line'].iloc[-1] else 'Bearish'}
        - RSI: {'Overbought' if hist['RSI'].iloc[-1] > 70 else 'Oversold' if hist['RSI'].iloc[-1] < 30 else 'Neutral'}
        - Backtest Profit/Loss: ${backtest_profit:.2f}
        - Sharpe Ratio: {sharpe_ratio:.2f}

        **Weekly Implied Volatility Data (Next 8 Weeks):**
        """
        for exp, iv_data in weekly_iv_data.items():
            report += f"- Expiry {exp}: Avg IV: {iv_data.get('avg_iv', 'N/A')}\n"

        report += f"""
        **Macroeconomic Data (FRED):**
        - Federal Funds Rate: {fed_funds_rate}

        **Suggested Strategies:**
        {suggestions}
        """

        # Display the report in the main window
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.INSERT, report)

        # Step 7: Plot Advanced Charts
        plot_advanced_charts(hist)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Data Visualization for Analysis
def plot_advanced_charts(hist):
    fig = make_subplots(rows=3, cols=1)

    # Candlestick chart with Bollinger Bands
    candlestick = go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'],
                                 close=hist['Close'], name="Candlesticks")
    fig.add_trace(candlestick, row=1, col=1)

    # Overlay Bollinger Bands
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name='Upper Band', line=dict(color='blue')), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name='Lower Band', line=dict(color='red')), row=1, col=1)

    # RSI Plot
    fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='green')), row=2, col=1)

    # MACD and Signal Line Plot
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal_Line'], name='Signal Line', line=dict(color='purple')), row=3,
                  col=1)

    fig.update_layout(title='Stock Price with Technical Indicators', xaxis_title='Date', yaxis_title='Price')
    fig.show()


# Create the main window for the application
root = tk.Tk()
root.title("Comprehensive Stock Report Generator")

# UI elements for inputting stock symbol and selecting period
label = tk.Label(root, text="Enter Stock Symbol:")
label.pack(pady=5)

entry = tk.Entry(root)
entry.pack(pady=5)

period_var = tk.StringVar(value="6mo")
period_label = tk.Label(root, text="Select Period:")
period_label.pack(pady=5)
period_dropdown = tk.OptionMenu(root, period_var, "1mo", "3mo", "6mo", "1y", "2y")
period_dropdown.pack(pady=5)

button = tk.Button(root, text="Generate Report", command=show_report)
button.pack(pady=10)

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30)
text_area.pack(pady=10, padx=10)

root.mainloop()
