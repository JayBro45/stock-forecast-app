import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def LSTMModel(train_df, test_df, save_dir="models/lstm", look_back=60, epochs=50, batch_size=30):
    """
    Train an LSTM model for stock forecasting, forecast on test data, and save results.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with columns ['Date', 'Close'].
    test_df : pd.DataFrame
        Testing data with columns ['Date', 'Close'].
    save_dir : str
        Directory to save model and plots.
    look_back : int
        Number of previous time steps to use as features.
    epochs : int
        Training epochs.
    batch_size : int
        Batch size for training.

    Returns
    -------
    model : Sequential
        Trained LSTM model.
    y_pred : np.ndarray
        Forecasted values for test data.
    metrics : dict
        MAE, MSE, and RMSE scores.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Combine train and test for scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_all = scaler.fit_transform(pd.concat([train_df, test_df])[["Close"]])

    scaled_train = scaled_all[:len(train_df)]
    scaled_test = scaled_all[len(train_df) - look_back:]

    # Create sequences
    def create_sequences(data, look_back):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i - look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_train, look_back)
    X_test, y_test = create_sequences(scaled_test, look_back)

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predictions
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Align with test_df
    y_pred = y_pred.flatten()
    y_test_rescaled = y_test_rescaled.flatten()
    test_dates = test_df["Date"].iloc[-len(y_test_rescaled):]

    # Metrics
    mse = mean_squared_error(y_test_rescaled, y_pred)
    mae = mean_absolute_error(y_test_rescaled, y_pred)
    rmse = np.sqrt(mse)
    metrics = {"MAE": mae, "RMSE": rmse}

    # Save model
    model.save(os.path.join(save_dir, "lstm_model.h5"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))


    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_df["Close"], label="Train")
    plt.plot(test_df["Date"], test_df["Close"], label="Actual")
    plt.plot(test_dates, y_pred, label="LSTM Forecast", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("LSTM Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "lstm_forecast.png"))
    plt.close()

    return model, y_pred, metrics

