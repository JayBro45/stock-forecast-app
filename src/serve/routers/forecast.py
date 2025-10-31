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
    Generate next 30-business-day forecast for the given model.
    Example: /forecast/xgboost
    """

    
    # Use absolute paths relative to this file (src/serve)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..","models","metrics", model_name.lower()))
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..", "data", "processed", "MSFT_clean.csv"))


    # Ensure model directory exists
    if not os.path.exists(base_path):
        return {"error": f"Model directory not found at {base_path}"}

    # Try to find model file dynamically
    possible_exts = [".pkl", ".h5", ".json", ".joblib"]
    model_path = None
    for ext in possible_exts:
        candidate = os.path.join(base_path, f"{model_name}_model{ext}")
        if os.path.exists(candidate):
            model_path = candidate
            break

    # Fallback: check if any file with 'model' in its name exists
    if not model_path:
        available = os.listdir(base_path)
        for f in available:
            if "model" in f.lower():
                model_path = os.path.join(base_path, f)
                break

        if not model_path:
            return {"error": f"No model file found in {base_path}. Available: {available}"}

    # Validate dataset
    if not os.path.exists(data_path):
        return {"error": f"Dataset not found at {data_path}."}

    df = pd.read_csv(data_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return {"error": "Dataset must contain 'Date' and 'Close' columns."}

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    last_date = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=30)

    model_type = model_name.lower()

    try:
        if model_type == "baseline":
            last_value = df["Close"].iloc[-1]
            forecast = np.full(30, last_value)

        elif model_type == "arima":
            model = joblib.load(model_path)
            forecast = model.forecast(steps=30)

        elif model_type == "neuralprophet":
            from neuralprophet import NeuralProphet
            model = NeuralProphet.load(model_path)
            prophet_df = df.rename(columns={"Date": "ds", "Close": "y"})
            future = model.make_future_dataframe(prophet_df, periods=30)
            forecast_df = model.predict(future)
            forecast = forecast_df["yhat1"].iloc[-30:].values

        elif model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            last_value = df["Close"].iloc[-1]
            forecast = []
            for _ in range(30):
                next_val = model.predict(np.array([[last_value]]))[0]
                forecast.append(next_val)
                last_value = next_val

        elif model_type == "lstm":
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler

            model = tf.keras.models.load_model(model_path)
            scaler_path = os.path.join(base_path, "scaler.pkl")
            if not os.path.exists(scaler_path):
                return {"error": "Scaler file missing for LSTM model."}

            scaler = joblib.load(scaler_path)
            last_lookback = df["Close"].values[-30:]
            scaled_input = scaler.transform(last_lookback.reshape(-1, 1)).reshape(1, 30, 1)
            forecast_scaled = model.predict(scaled_input)[0]
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

        else:
            return {"error": f"Unsupported model: {model_name}"}

    except Exception as e:
        return {"error": f"Forecast failed: {str(e)}"}

    result_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast
    })

    return {
        "model": model_name,
        "forecast_days": 30,
        "forecast": result_df.to_dict(orient="records")
    }
