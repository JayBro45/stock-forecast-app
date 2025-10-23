import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


def ArimaModel(train_df, test_df, order=(6, 3, 1), save_dir="models/arima"):
    """
    Train an ARIMA model, forecast on test data, and save results.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Train model
    model = ARIMA(train_df["Close"], order=order)
    model_fit = model.fit()

    # Forecast
    steps = len(test_df)
    forecast = model_fit.forecast(steps=steps)

    # Evaluate
    y_true = test_df["Close"].values
    y_pred = forecast.values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {"MAE": mae, "RMSE": rmse}

    # Save model
    model_path = os.path.join(save_dir, "arima_model.pkl")
    joblib.dump(model_fit, model_path)
    print(f"💾 ARIMA model saved at: {model_path}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_df["Close"], label="Train")
    plt.plot(test_df["Date"], test_df["Close"], label="Actual")
    plt.plot(test_df["Date"], y_pred, label="ARIMA Forecast", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"ARIMA{order} Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "arima_forecast.png"))
    plt.close()

    return model_fit, forecast, metrics
