# Stock Forecast & Analysis App

> ⚠ Educational project – **not** financial advice.

## Goal
An end-to-end application for:
- Collecting and cleaning OHLCV stock data
- Forecasting prices using classical & ML time-series models
- Serving predictions through a FastAPI backend
- Visualizing results via an interactive dashboard

## Planned Features
- Data ingestion from Yahoo Finance
- Feature engineering (lags, technical indicators)
- Baseline models (Naive, SMA)
- ARIMA & Gradient Boosting forecasts
- Walk-forward validation & backtesting
- FastAPI REST endpoints
- Streamlit / Plotly visualizations

### Data Pipeline
- Uses [yfinance](https://pypi.org/project/yfinance/) to pull OHLCV data
- Stores unmodified CSVs in `data/raw/`
- Notebooks access `data/raw` for exploratory plots


## Environment Setup
```bash
git clone https://github.com/JayBro45/stock-forecast-app.git
cd stock-forecast-app
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
