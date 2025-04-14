# Predict live market movements
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from data_fetcher import fetch_live_data
import ta
from collections import deque

# Define the MSE function explicitly
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

custom_objects = {"mse": mse}  # Register custom loss manually

# Load model & scaler with correct paths
model_path = "F:/Data_Science/Projects/nifty50_trading_ai/models/trained_model.h5"
scaler_path = "F:/Data_Science/Projects/nifty50_trading_ai/models/scaler.pkl"

try:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    scaler = joblib.load(scaler_path)
    print("✅ Model & Scaler Loaded Successfully!")
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit()

# Get expected feature names from scaler
expected_features = scaler.feature_names_in_  # Fetch expected feature names
print(f"ℹ Expected Features: {expected_features}")

# Store the last 60 data points in memory (FIFO queue)
data_queue = deque([[0] * len(expected_features)] * 60, maxlen=60)


def preprocess_live_data(live_data):
    """Ensure all required indicators are present."""
    close_series = pd.Series([live_data["close"]])  # Convert to Pandas Series

    # Compute indicators
    live_data["RSI"] = ta.momentum.RSIIndicator(close_series, window=14).rsi().iloc[-1]

    macd = ta.trend.MACD(close_series)
    live_data["MACD"] = macd.macd().iloc[-1]
    live_data["MACD_SIGNAL"] = macd.macd_signal().iloc[-1]

    # Fill missing features with 0
    for feature in expected_features:
        if feature not in live_data:
            live_data[feature] = 0

    return live_data


def predict_next_movement():
    """Fetch live data, preprocess it, and predict the next market move."""
    live_data = fetch_live_data()
    live_data = preprocess_live_data(live_data)

    # Prepare input data with correct feature order
    latest_data = [live_data[feature] for feature in expected_features]
    data_queue.append(latest_data)

    # Ensure we have 60 time steps before making predictions
    if len(data_queue) < 60:
        print("⏳ Not enough data yet. Waiting for 60 time steps.")
        return "Waiting"

    # Convert queue to numpy array and reshape for model input
    input_sequence = np.array(data_queue).reshape(60, len(expected_features))  # Shape (60, features)

    # Scale the input sequence
    scaled_input = scaler.transform(input_sequence)  # Transform each feature
    scaled_input = scaled_input.reshape(1, 60, len(expected_features))  # Reshape for LSTM

    # Predict market movement
    prediction = model.predict(scaled_input)[0][0]

    return "UP" if prediction > live_data["close"] else "DOWN"


if __name__ == "__main__":
    print(f"🔮 Market Prediction: {predict_next_movement()}")
