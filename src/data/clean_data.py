from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def clean_ticker(symbol: str):
    file_path = RAW / f"{symbol}_raw.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"]).set_index("Date")

    # --- Basic cleaning ---
    df = df.sort_index()               # ensure ascending by date
    df = df[~df.index.duplicated()]    # drop any duplicate rows

    # handle missing (simple forward fill then drop any left)
    df = df.ffill().dropna()

    # --- Feature engineering ---
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df['Volatility_10'] = df['Daily_Return'].rolling(10).std()
    df['DayOfWeek'] = df.index.dayofweek

    # drop initial NaNs caused by rolling
    df = df.dropna()

    # Save
    out_path = PROC / f"{symbol}_clean.csv"
    df.to_csv(out_path)
    print(f"Saved cleaned {symbol} → {out_path}")

if __name__ == "__main__":
    clean_ticker("AAPL")
