from fastapi import APIRouter
import joblib
import pandas as pd
import numpy as np
import os
from datetime import timedelta

router = APIRouter(prefix="/forecast", tags=["Forecast"])

@router.get("/{model_name}")
def forecast_next_30_days(model_name: str):
    """
    Generate next 30-day forecast for the given model.

    Example: /forecast/xgboost
    """

    model_path = f"models/{model_name}/{model_name}_model.pkl"
    data_path = "data/processed/MSFT_clean.csv"  # <-- path to your last used dataset

    if not os.path.exists(model_path):
        return {"error": f"Model '{model_name}' not found at {model_path}"}

    if not os.path.exists(data_path):
        return {"error": "Dataset not found. Please ensure 'data/processed/MSFT_clean.csv' exists."}

    # Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # Ensure correct column names
    if "Date" not in df.columns or "Close" not in df.columns:
        return {"error": "Dataset must have 'Date' and 'Close' columns."}

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Prepare data for forecasting
    last_date = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=30)  # 30 business days

    # Forecast based on model type
    if model_name.lower() == "naive_baseline":
        last_value = df["Close"].iloc[-1]
        forecast = np.full(30, last_value)

    elif model_name.lower() == "arima":
        forecast = model.forecast(steps=30)

    elif model_name.lower() == "xgboost":
        last_value = df["Close"].iloc[-1]
        X_future = np.arange(last_value, last_value + 30).reshape(-1, 1)
        forecast = model.predict(X_future)

    elif model_name.lower() == "neuralprophet":
        from neuralprophet import NeuralProphet
        prophet_model = NeuralProphet.load(model_path)
        train_df = df.rename(columns={"Date": "ds", "Close": "y"})
        future = prophet_model.make_future_dataframe(train_df, periods=30)
        forecast_df = prophet_model.predict(future)
        forecast = forecast_df["yhat1"].iloc[-30:].values

    elif model_name.lower() == "lstm":
        import tensorflow as tf
        lstm_model = tf.keras.models.load_model(model_path)
        last_value = df["Close"].values[-30:]
        scaled_input = np.array(last_value).reshape(1, 30, 1)
        forecast = lstm_model.predict(scaled_input)[0, :30]

    else:
        return {"error": f"Unsupported model: {model_name}"}

    # Create forecast DataFrame
    result_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast
    })

    # Convert to list of dicts for JSON response
    return {
        "model": model_name,
        "forecast_days": 30,
        "forecast": result_df.to_dict(orient="records")
    }
