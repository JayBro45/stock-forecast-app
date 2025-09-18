# Stock Forecast & Analysis App

> ⚠ Educational project – **not** financial advice.

## Goal
An end-to-end application for:
- Collecting and cleaning OHLCV stock data
- Forecasting prices using classical & ML time-series models
- Serving predictions through a FastAPI backend
- Visualizing results via an interactive dashboard

---

## ✅ Current Features
- **Data Handling**
  - CSV ingestion & preprocessing
  - Added technical indicators (returns, log returns, SMA20, SMA50)

- **Models Implemented**
  - **Naive Baseline** (last observed value forecast)  
  - **Prophet** (Facebook Prophet for time-series forecasting)  
  - **NeuralProphet** (neural extension of Prophet with autoregressive terms)  

- **Automation**
  - Each script performs: **train/test split → training → forecasting → evaluation → saving results**
  - Metrics (MAE, RMSE) automatically computed
  - Forecast plots saved for comparison

- **Outputs**
  - Model artifacts saved as `.pkl`
  - Forecast plots saved as `.png`
  - Metrics returned as a Python dict (JSON-like)

---

## 🚀 Planned Features
- More models:
  - SMA / EMA baselines
  - ARIMA / SARIMA
  - Gradient Boosting & ML regressors
- Walk-forward validation & backtesting
- REST API with **FastAPI**
- Interactive visualization with **Streamlit / Plotly**

---

## 📂 Data Pipeline
- Uses [yfinance](https://pypi.org/project/yfinance/) to pull OHLCV data  
- Stores raw CSVs in `data/raw/`  
- Cleaned datasets in `data/processed/`  
- Models train & test on processed datasets  

---

## 🛠 Environment Setup
```bash
git clone https://github.com/JayBro45/stock-forecast-app.git
cd stock-forecast-app
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
