import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def XGBoostModel(train_df, test_df, save_dir="models/xgboost",
                 HORIZON=30, n_iter=3, cv_splits=3):

    os.makedirs(save_dir, exist_ok=True)

    # ---------------- Merge & Feature Engineering ----------------
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

    # === FIX: Shift indicators by 1 (no leakage) ===
    df['EMA_9'] = df['Close'].ewm(span=9).mean().shift(1)
    df['SMA_5'] = df['Close'].rolling(5).mean().shift(1)
    df['SMA_10'] = df['Close'].rolling(10).mean().shift(1)
    df['SMA_15'] = df['Close'].rolling(15).mean().shift(1)
    df['SMA_30'] = df['Close'].rolling(30).mean().shift(1)
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # RSI
    def rsi_fn(df, n=14):
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(n).mean()
        roll_down = pd.Series(loss).rolling(n).mean()
        rs = roll_up / roll_down
        return 100 - (100 / (1 + rs))

    df['RSI'] = rsi_fn(df).shift(1).fillna(0)

    # MACD
    EMA_12 = df['Close'].ewm(span=12).mean()
    EMA_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = (EMA_12 - EMA_26).shift(1)
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean().shift(1)

    # ATR
    df['High-Low'] = df['High'] - df['Low']
    df['High-Prev_Close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-Prev_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Prev_Close', 'Low-Prev_Close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean().shift(1)

    # Calendar
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

    # Multi-output targets for TRAIN only
    for i in range(1, HORIZON + 1):
        train_part[f"target_t+{i}"] = train_part['Close'].shift(-i)

    train_feat = train_part.dropna().reset_index(drop=True)

    # === TEST EVALUATION: only t+1 (Option B) ===
    test_eval = test_part.copy()
    test_eval["target_t+1"] = test_eval["Close"].shift(-1)
    test_eval = test_eval.dropna().reset_index(drop=True)

    if len(test_eval) == 0:
        raise ValueError("Test set is too small to evaluate even t+1 predictions.")

    # ---------------- Prepare ML Inputs ----------------
    target_cols = [f"target_t+{i}" for i in range(1, HORIZON + 1)]

    X_train = train_feat.drop(columns=["Date"] + target_cols)
    y_train = train_feat[target_cols]

    X_test_eval = test_eval.drop(columns=["Date", "target_t+1"])
    y_test_eval = test_eval["target_t+1"]  # <= ONLY t+1

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

    # ---------------- t+1 Evaluation ----------------
    y_pred_test_full = best_model.predict(X_test_eval)
    y_pred_test = y_pred_test_full[:, 0]  # first horizon only

    mae = mean_absolute_error(y_test_eval, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_test))

    metrics = {"MAE": mae, "RMSE": rmse}
    print(f"Metrics (TEST t+1): MAE={mae:.4f}, RMSE={rmse:.4f}")

    # ---------------- Save Model ----------------
    model_path = os.path.join(save_dir, "xgboost_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"💾 Saved XGBoost model at {model_path}")

    # ---------------- Plot (ARIMA-style) ----------------
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_df["Close"], label="Train")
    plt.plot(test_df["Date"], test_df["Close"], label="Actual")

    plt.plot(
        test_eval["Date"].iloc[:len(y_pred_test)],
        y_pred_test,
        label="XGBoost Forecast (t+1 only)",
        linestyle="dashed",
        color="orange"
    )

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("XGBoost Rolling Forecast (t+1) vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "xgboost_forecast.png"))
    plt.close()

    return best_model, y_pred_test, metrics