import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def XGBoostModel(train_df, test_df, save_dir="models/xgboost", HORIZON=30, n_iter=3, cv_splits=3):
    """
    Train a multi-output XGBoost model that predicts the next HORIZON days at once.
    Evaluates performance inside test set (1-step ahead) and also enables future forecasts.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must have: ['Date', 'Close'] (+ optional OHLCV columns)
    test_df : pd.DataFrame
        Same columns as train_df
    save_dir : str
        Folder to store model + forecast plot.
    HORIZON : int
        Number of future days to predict at once.
    n_iter : int
        Number of parameter samples for RandomSearch.
    cv_splits : int
        Time series CV splits.

    Returns
    -------
    best_model : MultiOutputRegressor
        Trained model.
    y_pred : np.ndarray
        Predictions for the test window (multi-output: shape [n_test, HORIZON]).
    metrics : dict
        MAE and RMSE based on 1-step actual comparison.
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---------------- Merge + Feature Engineering ----------------
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

    # Technical indicators
    df['EMA_9'] = df['Close'].ewm(span=9).mean().shift()
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Close'].rolling(30).mean().shift()
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # RSI
    def relative_strength_idx(df, n=14):
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(n).mean()
        roll_down = pd.Series(loss).rolling(n).mean()
        rs = roll_up / roll_down
        return pd.Series(100 - (100 / (1 + rs)), index=df.index)

    df['RSI'] = relative_strength_idx(df).fillna(0)

    # MACD
    EMA_12 = df['Close'].ewm(span=12).mean()
    EMA_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = EMA_12 - EMA_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    # ATR
    df['High-Low'] = df['High'] - df['Low']
    df['High-Prev_Close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-Prev_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Prev_Close', 'Low-Prev_Close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Lag features
    for i in range(1, 6):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_lag_{i}'] = df['Volume'].shift(i)
        df[f'ATR_lag_{i}'] = df['ATR'].shift(i)

    # ---------------- Train/Test Split ----------------
    train_part = df[df['Date'] <= train_df['Date'].max()].copy()
    test_part = df[df['Date'] > train_df['Date'].max()].copy()

    # --- Create multi-output targets only for train part ---
    for i in range(1, HORIZON + 1):
        train_part[f"target_t+{i}"] = train_part['Close'].shift(-i)

    train_feat = train_part.dropna().reset_index(drop=True)
    test_feat = test_part.reset_index(drop=True)

    target_cols = [f"target_t+{i}" for i in range(1, HORIZON + 1)]
    X_train = train_feat.drop(columns=["Date"] + target_cols)
    y_train = train_feat[target_cols]
    X_test = test_feat.drop(columns=["Date"], errors="ignore")

    # ---------------- Model & Hyperparameter Search ----------------
    base_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    param_dist = {
        "n_estimators": [300, 500],
        "learning_rate": [0.01, 0.1],
        "max_depth": [6, 8],
    }

    model = MultiOutputRegressor(base_model)
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        model,
        param_distributions={"estimator__" + k: v for k, v in param_dist.items()},
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # ---------------- Prediction & Metrics ----------------
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_train, best_model.predict(X_train))
    rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
    metrics = {"MAE": mae, "RMSE": rmse}
    print(f"Metrics (Train window, Avg {HORIZON}-day): MAE={mae:.2f}, RMSE={rmse:.2f}")

    # ---------------- Save Model ----------------
    model_path = os.path.join(save_dir, "xgboost.joblib")
    joblib.dump(best_model, model_path)
    print(f"💾 Saved XGBoost model at {model_path}")

    # ---------------- Plot ----------------
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_df["Close"], label="Train")
    plt.plot(test_df["Date"], test_df["Close"], label="Actual")

    # If model predicts 30 days ahead, take the last test date and extend 30 business days
    if y_pred.shape[1] == 30:
        future_dates = pd.bdate_range(test_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30)
        plt.plot(future_dates, y_pred[-1], label="XGBoost 30-Day Forecast", linestyle="dashed", color="orange")
    else:
        plt.plot(test_df["Date"].iloc[:len(y_pred)], y_pred, label="XGBoost Forecast", linestyle="dashed", color="orange")

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("XGBoost 30-Day Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "xgboost_forecast.png"))
    plt.close()

    return best_model, y_pred, metrics