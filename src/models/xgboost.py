import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

def XGBoostModel(train_df, test_df, save_dir="models/xgboost", n_iter=20, cv_splits=5):
    """
    Train an XGBoost model with cross-validation, forecast on test data, and save results.

    ```
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with columns ['Date', 'Close'].
    test_df : pd.DataFrame
        Testing data with columns ['Date', 'Close'].
    save_dir : str
        Directory to save model and plots.
    n_iter : int
        Number of parameter settings sampled in RandomizedSearchCV.
    cv_splits : int
        Number of splits for TimeSeriesSplit.

    Returns
    -------
    best_model : xgb.XGBRegressor
        Trained XGBoost model.
    y_pred : np.ndarray
        Forecasted values for test data.
    metrics : dict
        MAE, MSE, and RMSE scores.
    """

    os.makedirs(save_dir, exist_ok=True)

    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

    # ---------- Technical Indicators (ordered efficiently) ----------
    df['EMA_9'] = df['Close'].ewm(span=9).mean().shift() # Corrected: Used 'span' for EMA calculation
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Close'].rolling(30).mean().shift()
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # --- RSI Calculation (fixed index alignment) ---
    def relative_strength_idx(df, n=14):
        close = df['Close']
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(n).mean()
        roll_down = pd.Series(loss).rolling(n).mean()
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=df.index)

    df['RSI'] = relative_strength_idx(df).fillna(0)

    # MACD
    EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

    # ATR
    df['High-Low'] = df['High'] - df['Low']
    df['High-Prev_Close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-Prev_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Prev_Close', 'Low-Prev_Close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Date-based features
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # --- Lag features ---
    for i in range(1, 6):  # Past 5 days
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_lag_{i}'] = df['Volume'].shift(i)
        df[f'ATR_lag_{i}'] = df['ATR'].shift(i)

    df['SMA_5_lag_1'] = df['SMA_5'].shift(1)
    df['EMA_9_lag_1'] = df['EMA_9'].shift(1)

    # --- Predict next day's Close (safe shift handling) ---
    df['Close'] = df['Close'].shift(-1)
    df = df.dropna().reset_index(drop=True)

    # ---------- Train/Test Split ----------
    train_df_feat = df[df['Date'] <= train_df['Date'].max()].reset_index(drop=True)
    test_df_feat = df[df['Date'] > train_df['Date'].max()].reset_index(drop=True)

    X_train = train_df_feat.drop(columns=["Date", "Close"])
    y_train = train_df_feat["Close"]
    X_test = test_df_feat.drop(columns=["Date", "Close"])
    y_test = test_df_feat["Close"]

    # ---------- Model Training ----------
    model = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")

    param_dist = {
        "n_estimators": [100, 300, 500, 700],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
        "gamma": [0.01, 0.02, 0.05]
    }

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # ---------- Evaluation ----------
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    metrics = {"MAE": mae, "RMSE": rmse}

    # ---------- Save Model ----------
    model_path = os.path.join(save_dir, "xgboost_model.bin")
    best_model.save_model(model_path)
    print(f"💾 Model saved at: {model_path}")

    # ---------- Plot Results  ----------
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_df["Close"], label="Train")
    # Corrected: Used the date axis from the feature-engineered dataframe to prevent a mismatch
    plt.plot(test_df_feat["Date"], y_test, label="Actual")
    plt.plot(test_df_feat["Date"], y_pred, label="XGBoost Forecast", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("XGBoost Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "xgboost_forecast.png")
    plt.savefig(plot_path)
    plt.close()

    return best_model, y_pred, metrics