import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def NaiveBaselineModel(train_df, test_df, save_dir="models/naive_baseline"):
    """
    Naive baseline model: forecast by repeating the last observed value.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with columns ['ds', 'y'].
    test_df : pd.DataFrame
        Testing data with columns ['ds', 'y'].
    save_dir : str
        Directory to save model and plots.

    Returns
    -------
    last_observation : float
        The last observed value from training data.
    forecast : np.ndarray
        Forecasted values (constant).
    metrics : dict
        MAE and RMSE scores.
    """

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Fit: store last observation
    last_observation = train_df["Close"].iloc[-1]

    # Forecast: repeat last value for length of test
    forecast = np.full(shape=len(test_df), fill_value=last_observation)

    # Metrics
    y_true = test_df["Close"].values
    y_pred = forecast
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {"MAE": mae, "RMSE": rmse}

    # Save model (just the last observation)
    model_path = os.path.join(save_dir, "naive_baseline.pkl")
    joblib.dump(last_observation, model_path)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_df["Close"], label="Train")
    plt.plot(test_df["Date"], test_df["Close"], label="Actual")
    plt.plot(test_df["Date"], y_pred, label="Naive Forecast", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Naive Baseline Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, "naive_forecast.png")
    plt.savefig(plot_path)
    plt.close()

    return last_observation, forecast, metrics
