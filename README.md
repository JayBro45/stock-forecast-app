# Stock Forecast App

## 📝 Overview

This project is a Python-based web application designed to forecast stock prices using a variety of time-series models. It fetches historical stock data, cleans and preprocesses it, trains several different forecasting models, and serves the predictions through a RESTful API built with FastAPI.

## ✨ Features

* **Data Pipeline**: Scripts to fetch historical data from Yahoo Finance and process it for modeling.
* **Multiple Models**: Implements and evaluates several forecasting models, from simple baselines to complex neural networks.
* **REST API**: Exposes a simple API endpoint to get a 30-day forecast for a given model.
* **Model Evaluation**: Notebooks for analyzing and comparing the performance of different models.

## 🛠️ Tech Stack

* **Backend Framework**: FastAPI
* **Data Manipulation**: Pandas, NumPy
* **Forecasting Models**:
    * XGBoost
    * ARIMA (`statsmodels`)
    * LSTM (`tensorflow`)
    * NeuralProphet
* **Data Fetching**: `yfinance`
* **Plotting**: `matplotlib`, `plotly`

## 📂 Project Structure

```

stock-forecast-app/
├── data/
│   ├── raw/          \# Raw, unprocessed data
│   └── processed/    \# Cleaned data for modeling
├── notebooks/        \# Jupyter notebooks for EDA, modeling, and evaluation
├── src/
│   ├── data/         \# Scripts for fetching and cleaning data
│   ├── models/       \# Scripts for training models
│   └── serve/        \# FastAPI application for serving forecasts
└── requirements.txt  \# Project dependencies

````

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd stock-forecast-app
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the data pipeline:**
    First, fetch the raw data, then clean it.
    ```bash
    python src/data/fetch_data.py
    python src/data/clean_data.py
    ```

2.  **Train the models:**
    Run the training scripts for each model you want to use.
    ```bash
    python src/models/arima.py
    python src/models/xgboost.py
    python src/models/lstm.py
    # ... and so on for other models
    ```

3.  **Start the API server:**
    ```bash
    uvicorn src.serve.main:app --reload
    ```
    The server will be available at `http://127.0.0.1:8000`.

## 🤖 API Endpoints

### Get Forecast

* **Endpoint**: `/forecast/{model_name}`
* **Method**: `GET`
* **Description**: Generates a 30-business-day forecast using the specified model.
* **URL Params**:
    * `model_name`: The name of the model to use. Must be one of:
        * `arima`
        * `lstm`
        * `xgboost`
        * `neuralprophet`
        * `naive_baseline`

* **Example Request**:
    ```bash
    curl [http://127.0.0.1:8000/forecast/xgboost](http://127.0.0.1:8000/forecast/xgboost)
    ```

* **Example Response**:
    ```json
    {
      "model": "xgboost",
      "forecast_days": 30,
      "forecast": [
        {
          "Date": "2025-09-23T00:00:00",
          "Forecast": 515.123456
        },
        {
          "Date": "2025-09-24T00:00:00",
          "Forecast": 516.789012
        }
        // ... 28 more days
      ]
    }
    ```
````