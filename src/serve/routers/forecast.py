from fastapi import APIRouter
import joblib
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import traceback

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
    possible_exts = [".pkl", ".h5", ".joblib"]
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
        if model_type == "naive_baseline":
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
            HORIZON = 30  # forecast range
            # --- Load model ---
            model = xgb.XGBRegressor()
            model.load_model(model_path)

            # --- Recreate feature pipeline (must match model training) ---
            feat = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].copy()

            feat['EMA_9'] = feat['Close'].ewm(span=9).mean().shift()
            feat['SMA_5'] = feat['Close'].rolling(5).mean().shift()
            feat['SMA_10'] = feat['Close'].rolling(10).mean().shift()
            feat['SMA_15'] = feat['Close'].rolling(15).mean().shift()
            feat['SMA_30'] = feat['Close'].rolling(30).mean().shift()
            feat['DayOfWeek'] = feat['Date'].dt.dayofweek

            def rsi_fn(df, n=14):
                delta = df['Close'].diff()
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                roll_up = pd.Series(gain).rolling(n).mean()
                roll_down = pd.Series(loss).rolling(n).mean()
                rs = roll_up / roll_down
                return 100 - (100 / (1 + rs))

            feat['RSI'] = rsi_fn(feat).fillna(0)

            EMA_12 = feat['Close'].ewm(span=12).mean()
            EMA_26 = feat['Close'].ewm(span=26).mean()
            feat['MACD'] = EMA_12 - EMA_26
            feat['MACD_signal'] = feat['MACD'].ewm(span=9).mean()
            feat['High-Low'] = feat['High'] - feat['Low']
            feat['High-Prev_Close'] = np.abs(feat['High'] - feat['Close'].shift(1))
            feat['Low-Prev_Close'] = np.abs(feat['Low'] - feat['Close'].shift(1))
            feat['TR'] = feat[['High-Low', 'High-Prev_Close', 'Low-Prev_Close']].max(axis=1)
            feat['ATR'] = feat['TR'].rolling(14).mean()
            feat['Month'] = feat['Date'].dt.month
            feat['Quarter'] = feat['Date'].dt.quarter
            feat['DayOfYear'] = feat['Date'].dt.dayofyear

            # lag features
            for i in range(1, 6):
                feat[f'Close_lag_{i}'] = feat['Close'].shift(i)
                feat[f'Volume_lag_{i}'] = feat['Volume'].shift(i)
                feat[f'ATR_lag_{i}'] = feat['ATR'].shift(i)

            feat = feat.dropna().reset_index(drop=True)

            # --- Use last row as input ---
            X_last = feat.drop(columns=["Date", "Close"]).iloc[-1].values.reshape(1, -1)

            # --- Model predicts 30 outputs at once ---
            forecast = model.predict(X_last)[0]  # shape: (30,)


        elif model_type == "lstm":
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler

            model = tf.keras.models.load_model(model_path)
            scaler_path = os.path.join(base_path, "scaler.pkl")
            if not os.path.exists(scaler_path):
                return {"error": "Scaler file missing for LSTM model."}

            scaler = joblib.load(scaler_path)
            last_lookback = df["Close"].values[-30:].tolist() # Start with the last 30 known values
            forecast = []

            for _ in range(30):
                # Scale and reshape the input for the model
                scaled_input = scaler.transform(np.array(last_lookback[-30:]).reshape(-1, 1))
                scaled_input = scaled_input.reshape(1, 30, 1)

                # Predict the next value
                forecast_scaled = model.predict(scaled_input)[0]
                
                # Inverse transform the prediction to get the actual value
                next_val = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()[0]
                
                # Append the prediction to our forecast list
                forecast.append(next_val)
                
                # Append the prediction to the lookback list to use in the next iteration
                last_lookback.append(next_val)

        else:
            return {"error": f"Unsupported model: {model_name}"}

    except Exception as e:
        traceback.print_exc() 
        
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
