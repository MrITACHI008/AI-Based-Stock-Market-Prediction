# Train AI model on past NIFTY 50 data
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import ta

# Load data
df = pd.read_csv("../data/processed_data.csv")

# Calculate additional indicators
df["VWAP"] = ta.volatility.BollingerBands(df["Close"]).bollinger_mavg()
df["EMA9"] = ta.trend.EMAIndicator(df["Close"], window=9).ema_indicator()
df["EMA21"] = ta.trend.EMAIndicator(df["Close"], window=21).ema_indicator()

# Fibonacci retracement levels (for support/resistance)
high, low = df["High"].rolling(20).max(), df["Low"].rolling(20).min()
df["Fib_38.2"] = low + (high - low) * 0.382
df["Fib_50"] = low + (high - low) * 0.5
df["Fib_61.8"] = low + (high - low) * 0.618

# Prepare dataset
features = ["Close", "RSI", "MACD", "MACD_Signal", "VWAP", "EMA9", "EMA21", "Fib_38.2", "Fib_50", "Fib_61.8"]
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[features])

# Creating training sequences
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i, 0])  # Predicting Close price

X, y = np.array(X), np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(50, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# Train model
model.fit(X, y, epochs=500, batch_size=32)
model.save("../models/trained_model.h5")
joblib.dump(scaler, "../models/scaler.pkl")

print("✅ Model Trained & Saved with Enhanced Features!")
