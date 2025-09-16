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

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with columns ['ds', 'y'].
    test_df : pd.DataFrame
        Testing data with columns ['ds', 'y'].
    order : tuple
        ARIMA order (p, d, q).
    save_dir : str
        Directory to save model and plots.

    Returns
    -------
    model_fit : ARIMA
        Trained ARIMA model.
    forecast : pd.Series
        Forecasted values.
    metrics : dict
        MAE and RMSE scores.
    """

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Train ARIMA model
    model = ARIMA(train_df["Close"], order=order)
    model_fit = model.fit()

    # Forecast
    steps = len(test_df)
    forecast = model_fit.forecast(steps=steps)

    # Align predictions with test data
    y_true = test_df["Close"].values
    y_pred = forecast.values

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {"MAE": mae, "RMSE": rmse}

    # Save model
    model_path = os.path.join(save_dir, "arima_model.pkl")
    joblib.dump(model_fit, model_path)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_df["Close"], label="Train")
    plt.plot(test_df["Date"], test_df["Close"], label="Actual")
    plt.plot(test_df["Date"], y_pred, label="Forecast", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(f"ARIMA{order} Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, "arima_forecast.png")
    plt.savefig(plot_path)
    plt.close()

    return model_fit, forecast, metrics
