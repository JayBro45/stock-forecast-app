import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

today = datetime.today().date()

def download_stock(symbol: str, start=(today-relativedelta(years=1)).strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d')):
    print(f"Fetching data for {symbol} from {start} to {end}")
    df = yf.download(symbol, start=start, end=end,progress=False,auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

def save_to_csv(df: pd.DataFrame, symbol: str):
    file_path = RAW_DATA_DIR / f"{symbol}_raw.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved raw data -> {file_path}")

if __name__ == "__main__":
    symbol = "AAPL"
    df = download_stock(symbol)
    save_to_csv(df, symbol)
