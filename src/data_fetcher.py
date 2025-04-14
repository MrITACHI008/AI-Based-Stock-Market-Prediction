# Script to fetch live & historical market data
import pandas as pd
import yfinance as yf
import requests
from tradingview_ta import TA_Handler, Interval, Exchange

# Function to fetch historical data
def fetch_historical_data(ticker="^NSEI", start="1999-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end, interval="1d")
    data.to_csv("../data/historical_data.csv")
    print("✅ Historical Data Saved!")

# Function to fetch real-time NIFTY50 data
def fetch_live_data():
    handler = TA_Handler(
        symbol="NIFTY",
        exchange="NSE",
        screener="india",
        interval=Interval.INTERVAL_1_MINUTE
    )
    analysis = handler.get_analysis()
    return analysis.indicators

if __name__ == "__main__":
    fetch_historical_data()
    print(fetch_live_data())  # Prints live data
