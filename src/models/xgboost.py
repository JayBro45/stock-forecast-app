import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


def XGBoostModel(train_df, test_df, save_dir="models/xgboost", n_iter=20, cv_splits=5):
    """
    Train an XGBoost model with cross-validation, forecast on test data, and save results.

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

    # Prepare features (lags)
    def create_features(df, lags=5):
        df = df.copy()
        for lag in range(1, lags + 1):
            df[f"lag_{lag}"] = df["Close"].shift(lag)
        df = df.dropna()
        return df

    train_df_feat = create_features(train_df)
    test_df_feat = create_features(pd.concat([train_df.tail(5), test_df]))

    X_train = train_df_feat.drop(columns=["Date", "Close"])
    y_train = train_df_feat["Close"]

    X_test = test_df_feat.drop(columns=["Date", "Close"])
    y_test = test_df_feat["Close"]

    # Define model
    model = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")

    # Parameter grid
    param_dist = {
        "n_estimators": [100, 300, 500, 700],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    # TimeSeries CV
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

    # Predictions
    y_pred = best_model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse}

    # Save model
    model_path = os.path.join(save_dir, "xgboost_model.json")
    best_model.save_model(model_path)
    print(f"💾 Model saved at: {model_path}")



    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["Date"], train_df["Close"], label="Train")
    plt.plot(test_df["Date"], y_test, label="Actual")
    plt.plot(test_df["Date"], y_pred, label="XGBoost Forecast", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("XGBoost Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, "xgboost_forecast.png")
    plt.savefig(plot_path)
    plt.close()

    return best_model, y_pred, metrics

