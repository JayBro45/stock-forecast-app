import os
import pandas as pd
import matplotlib.pyplot as plt

# Import model functions
from src.models.naive_baseline import NaiveBaselineModel
from src.models.arima import ArimaModel
from src.models.neuralprophet import NeuralProphetModel
from src.models.xgboost import XGBoostModel
from src.models.lstm import LSTMModel


def evaluate_all_models(train_df, test_df, save_dir="../data/models/metrics"):
    """
    Train & evaluate all models on the same dataset.
    Saves metrics and plots for comparison.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Testing data
    save_dir : str
        Directory to save results
    """

    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # ---- Run Models ----
    # 1. Naive
    _, _, metrics_naive = NaiveBaselineModel(train_df, test_df, save_dir="models/naive_baseline")
    results["NaiveBaseline"] = metrics_naive

    # 2. ARIMA
    _, metrics_arima = ArimaModel(train_df, test_df, save_dir="models/arima")
    results["ARIMA"] = metrics_arima

    # 3. NeuralProphet
    _, metrics_prophet = NeuralProphetModel(train_df, test_df, save_dir="models/neuralprophet")
    results["NeuralProphet"] = metrics_prophet

    # 4. XGBoost
    _, metrics_xgb = XGBoostModel(train_df, test_df, save_dir="models/xgboost")
    results["XGBoost"] = metrics_xgb

    # 5. LSTM
    _, metrics_lstm = LSTMModel(train_df, test_df, save_dir="models/lstm")
    results["LSTM"] = metrics_lstm

    # ---- Collect Results ----
    df_results = pd.DataFrame(results).T
    df_results.to_csv(os.path.join(save_dir, "model_comparison.csv"))

    # ---- Bar Plot ----
    df_results.plot(kind="bar", figsize=(10,6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Error")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_plot.png"))
    plt.close()

    return df_results
