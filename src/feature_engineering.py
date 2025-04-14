# Script to add indicators like RSI, MACD
import pandas as pd
import ta


def add_technical_indicators(filepath="../data/historical_data.csv"):
    try:
        df = pd.read_csv(filepath)

        # ✅ Ensure "Close" column is numeric
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)  # Remove NaN rows

        # RSI (Relative Strength Index)
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()

        # Save processed data
        df.to_csv("../data/processed_data.csv", index=False)
        print("✅ Features Added!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    add_technical_indicators()
