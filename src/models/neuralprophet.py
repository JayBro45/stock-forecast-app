import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def NeuralProphetModel(train_df, test_df, save_dir="models/neuralprophet", epochs=100):
    """
    Train a NeuralProphet model, forecast on test data, and save results.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with columns including ['Date', 'Close'].
    test_df : pd.DataFrame
        Testing data with columns including ['Date', 'Close'].
    save_dir : str
        Directory to save model and plots.
    epochs : int
        Number of training epochs.

    Returns
    -------
    model : NeuralProphet
        Trained NeuralProphet model.
    forecast : pd.DataFrame
        Forecasted values.
    metrics : dict
        MAE and RMSE scores.
    """

    # --- Preprocess: Keep only Date & Close, rename to ds & y ---
    train_df = train_df.rename(columns={"Date": "ds", "Close": "y"})[["ds", "y"]]
    test_df = test_df.rename(columns={"Date": "ds", "Close": "y"})[["ds", "y"]]

    # Ensure datetime type
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    test_df["ds"] = pd.to_datetime(test_df["ds"])

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize NeuralProphet
    model = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    # Train
    model.fit(train_df, freq="B", epochs=epochs)

    # Forecast
    future = model.make_future_dataframe(train_df, periods=len(test_df))
    forecast = model.predict(future)

    # Align predictions with test data
    y_true = test_df["y"].values
    y_pred = forecast["yhat1"].iloc[-len(test_df):].values

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {"MAE": mae, "RMSE": rmse}

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["ds"], train_df["y"], label="Train")
    plt.plot(test_df["ds"], test_df["y"], label="Actual")
    plt.plot(test_df["ds"], y_pred, label="Forecast", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("NeuralProphet Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, "neuralprophet_forecast.png")
    plt.savefig(plot_path)
    plt.close()

    return model, forecast, metrics
