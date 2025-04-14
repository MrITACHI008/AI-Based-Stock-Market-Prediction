# 📈 Streamlit Dashboard for NIFTY 50 AI Trading
import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from data_fetcher import fetch_live_data
from live_prediction import predict_next_movement
import numpy as np  # 🔥 Import NumPy for NaN Handling

# Streamlit Page Configuration
st.set_page_config(page_title="NIFTY 50 AI Trading Dashboard", layout="wide")

# Title
st.title("📈 NIFTY 50 AI Trading Dashboard")

# Sidebar: Live Market Data
st.sidebar.header("📊 Live Market Data")


def get_live_data():
    """Fetch and format live market data."""
    live_data = fetch_live_data()
    live_df = pd.DataFrame(live_data.items(), columns=["Indicator", "Value"])

    # 🔥 FIX: Replace "N/A" with NaN & convert to appropriate type
    live_df["Value"] = live_df["Value"].replace("N/A", np.nan)
    live_df["Value"] = pd.to_numeric(live_df["Value"], errors="coerce")  # Convert to float

    return live_data, live_df


live_data, live_df = get_live_data()
st.sidebar.dataframe(live_df)

# AI Market Prediction
st.sidebar.subheader("🧠 AI Prediction")
prediction = predict_next_movement()
if "UP" in prediction:
    st.sidebar.success(f"🚀 Market Prediction: {prediction}")
elif "DOWN" in prediction:
    st.sidebar.error(f"📉 Market Prediction: {prediction}")
else:
    st.sidebar.warning(f"⚠️ Market Prediction: {prediction}")

# Trading Decision
st.sidebar.subheader("💹 Trading Decision")
if "UP" in prediction:
    st.sidebar.success("📊 Suggested Trade: **BUY CALL OPTION (CE)**")
elif "DOWN" in prediction:
    st.sidebar.error("📉 Suggested Trade: **BUY PUT OPTION (PE)**")
else:
    st.sidebar.warning("⚠️ Suggested Trade: **NO TRADE**")

# 📈 Load and Display Historical Market Data
st.subheader("📈 Historical Market Data & Trends")

try:
    file_path = "F:/Data_Science/Projects/nifty50_trading_ai/data/processed_data.csv"
    df = pd.read_csv(file_path)

    # 🛠️ Fix Date Column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise KeyError("Missing 'Date' column in dataset!")

    # 📉 Candlestick Chart for Market Trends
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Market Trend"
        )
    ])
    fig.update_layout(title="📊 NIFTY 50 Market Trend", xaxis_title="Date", yaxis_title="Price",
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("⚠️ Error loading historical data!")
    st.error(e)

# 📊 Technical Indicators Section
st.subheader("📊 Technical Indicators")

if live_data:
    try:
        indicators = {
            "RSI": live_data.get("RSI", "N/A"),
            "MACD": live_data.get("MACD", "N/A"),
            "MACD Signal": live_data.get("MACD_SIGNAL", "N/A"),
            "VWAP": live_data.get("VWAP", "N/A"),
            "EMA9": live_data.get("EMA9", "N/A"),
            "EMA21": live_data.get("EMA21", "N/A"),
            "Fib 38.2%": live_data.get("Fib_38.2", "N/A"),
            "Fib 50%": live_data.get("Fib_50", "N/A"),
            "Fib 61.8%": live_data.get("Fib_61.8", "N/A"),
        }
        indicators_df = pd.DataFrame(indicators.items(), columns=["Indicator", "Value"])

        # 🔥 FIX: Replace "N/A" with NaN & convert column type
        indicators_df["Value"] = indicators_df["Value"].replace("N/A", np.nan)
        indicators_df["Value"] = pd.to_numeric(indicators_df["Value"], errors="coerce")

        st.table(indicators_df)
    except Exception as e:
        st.error("⚠️ Error fetching indicators!")
        st.error(e)

# 🔄 Auto-Refresh Logic (Every 10 Seconds)
st.sidebar.write("🔄 Auto-refreshing every 10 seconds...")
time.sleep(10)
st.rerun()  # Updated from `st.experimental_rerun()`
