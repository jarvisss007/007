import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import traceback
import data_fetching as df
import technical_indicators as ti
import risk_management as rm
import backtesting as bt
import machine_learning_analysis as ml
import options_analysis as oa
import feature_engineering as fe
import reinforcement_learning as rl
import pandas as pd

# Initialize global variables
symbol = None
hist = None

# Main GUI Application
root = tk.Tk()
root.title("Comprehensive Stock Analysis and Options Trading Assistant")

# Create a Notebook widget for different sections
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# Create frames for each tab
frames = {
    "Data Overview": ttk.Frame(notebook),
    "Technical Indicators": ttk.Frame(notebook),
    "Risk Metrics": ttk.Frame(notebook),
    "Backtesting": ttk.Frame(notebook),
    "Options Analysis": ttk.Frame(notebook),
    "Machine Learning": ttk.Frame(notebook),
    "Reinforcement Learning": ttk.Frame(notebook)
}

# Add frames to the notebook
for title, frame in frames.items():
    notebook.add(frame, text=title)

# Entry fields to input stock symbol and select period
label = tk.Label(root, text="Enter Stock Symbol:")
label.pack(pady=5)

entry = tk.Entry(root)
entry.pack(pady=5)

period_var = tk.StringVar(value="6mo")
period_label = tk.Label(root, text="Select Period:")
period_label.pack(pady=5)
period_dropdown = tk.OptionMenu(root, period_var, "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y")
period_dropdown.pack(pady=5)

suggestion_label = tk.Label(root, text="Trading Suggestions:", font=("Arial", 12, "bold"))
suggestion_label.pack(pady=5)

suggestion_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=5)
suggestion_text.pack(pady=5, padx=5, fill='both', expand=True)

# Show text in a given frame
def show_text_in_frame(frame, text):
    for widget in frame.winfo_children():
        widget.destroy()
    text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=100, height=10)
    text_area.pack(pady=5, padx=5, fill='both', expand=True)
    text_area.insert(tk.INSERT, text)

# Show graph in a given frame
def show_graph_in_frame(frame, fig):
    for widget in frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# Data Overview
def display_data_overview():
    global symbol, hist
    symbol = entry.get().strip()
    period = period_var.get().strip()
    try:
        hist = df.fetch_stock_data(symbol, period)
        if hist.empty:
            raise ValueError(f"No data found for symbol '{symbol}'. Please check the symbol or try a different period.")
        report = f"Fetched data for {symbol} over {period}\nNumber of data points: {len(hist)}\n"
        show_text_in_frame(frames["Data Overview"], report)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch historical data: {e}")

# Technical Indicators
def display_technical_indicators():
    global hist
    try:
        if hist is None:
            raise ValueError("Please fetch stock data first.")
        hist = ti.calculate_indicators(hist)
        report = f"Technical Indicators for {symbol}:\nMoving Averages, RSI, MACD, etc., calculated successfully.\n"
        show_text_in_frame(frames["Technical Indicators"], report)

        # Plot Technical Indicators
        fig, ax = plt.subplots(figsize=(10, 5))
        ti.plot_indicators(hist, ax)  # Assuming you have a `plot_indicators()` function in `technical_indicators.py`
        show_graph_in_frame(frames["Technical Indicators"], fig)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate technical indicators: {e}")

# Risk Metrics
def display_risk_metrics():
    global hist
    try:
        if hist is None:
            raise ValueError("Please fetch stock data first.")
        report = rm.calculate_risk_metrics(hist)  # Ensure that the function `calculate_risk_metrics()` is implemented in `risk_management.py`
        show_text_in_frame(frames["Risk Metrics"], report)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate risk metrics: {e}")

# Backtesting
def display_backtesting():
    global hist
    try:
        if hist is None:
            raise ValueError("Please fetch stock data first.")
        report = bt.backtest_strategy(hist)  # Ensure `backtest_strategy()` is implemented in `backtesting.py`
        show_text_in_frame(frames["Backtesting"], report)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to complete backtesting: {e}")

# Options Analysis
def display_options_analysis():
    global symbol, hist
    try:
        if symbol is None:
            raise ValueError("Please fetch stock data first.")
        options_summary = oa.analyze_options(symbol)
        report = f"Options Analysis for {symbol}:\n\n"

        # Only show IV and Greeks for the next 8 weeks
        for option_data in options_summary:
            report += f"Expiration Date: {option_data['expiration_date']}\n"
            report += f"Average IV: {option_data['avg_iv']:.2f}\n"
            report += f"Greeks for Calls and Puts:\n"
            report += f"Calls:\n{option_data['calls']}\n"
            report += f"Puts:\n{option_data['puts']}\n\n"

        show_text_in_frame(frames["Options Analysis"], report)

        # Generate trading suggestions
        suggestions = oa.generate_trading_suggestions(options_summary, hist)
        suggestion_text.delete(1.0, tk.END)
        suggestion_text.insert(tk.INSERT, suggestions)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyze options: {e}")

# Machine Learning
def display_machine_learning():
    global hist
    try:
        if hist is None:
            raise ValueError("Please fetch stock data first.")
        report = ml.perform_ml_analysis(hist)
        show_text_in_frame(frames["Machine Learning"], report)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to perform machine learning analysis: {e}")

# Reinforcement Learning
def display_reinforcement_learning():
    try:
        report = rl.train_rl_agent()
        show_text_in_frame(frames["Reinforcement Learning"], report)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to perform reinforcement learning: {e}")

# Buttons for fetching data and displaying results
fetch_data_button = tk.Button(root, text="Fetch Data", command=display_data_overview)
fetch_data_button.pack(pady=10)

technical_button = tk.Button(frames["Technical Indicators"], text="Show Technical Indicators", command=display_technical_indicators)
technical_button.pack(pady=10)

risk_button = tk.Button(frames["Risk Metrics"], text="Show Risk Metrics", command=display_risk_metrics)
risk_button.pack(pady=10)

backtest_button = tk.Button(frames["Backtesting"], text="Show Backtesting Results", command=display_backtesting)
backtest_button.pack(pady=10)

options_button = tk.Button(frames["Options Analysis"], text="Show Options Analysis", command=display_options_analysis)
options_button.pack(pady=10)

ml_button = tk.Button(frames["Machine Learning"], text="Show Machine Learning Analysis", command=display_machine_learning)
ml_button.pack(pady=10)

rl_button = tk.Button(frames["Reinforcement Learning"], text="Show Reinforcement Learning Analysis", command=display_reinforcement_learning)
rl_button.pack(pady=10)

# Main loop for the GUI
root.mainloop()
