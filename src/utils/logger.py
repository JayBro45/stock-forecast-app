import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Import model functions
from src.models.baseline import NaiveBaselineModel
from src.models.arima import ArimaModel
from src.models.neuralprophet import NeuralProphetModel
from src.models.xgboost import XGBoostModel
from src.models.lstm import LSTMModel


def evaluate_all_models(train_df, test_df, save_dir="../data/models"):
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

    # Helper for model-specific folder
    def model_dir(name):
        return os.path.join(save_dir, name)

    # ---- Run Models ----
    print("\n⚡ Running Naive Baseline...")
    _, _, metrics_naive = NaiveBaselineModel(train_df, test_df, save_dir=model_dir("naive_baseline"))
    results["NaiveBaseline"] = metrics_naive
    print(f"✅ Done Naive Baseline | MAE: {metrics_naive['MAE']:.4f}, RMSE: {metrics_naive['RMSE']:.4f}")

    print("\n⚡ Running ARIMA...")
    _, _, metrics_arima = ArimaModel(train_df, test_df, save_dir=model_dir("arima"))
    results["ARIMA"] = metrics_arima
    print(f"✅ Done ARIMA | MAE: {metrics_arima['MAE']:.4f}, RMSE: {metrics_arima['RMSE']:.4f}")

    print("\n⚡ Running NeuralProphet...")
    _, _, metrics_prophet = NeuralProphetModel(train_df, test_df, save_dir=model_dir("neuralprophet"))
    results["NeuralProphet"] = metrics_prophet
    print(f"✅ Done NeuralProphet | MAE: {metrics_prophet['MAE']:.4f}, RMSE: {metrics_prophet['RMSE']:.4f}")

    print("\n⚡ Running XGBoost...")
    _, _, metrics_xgb = XGBoostModel(train_df, test_df, save_dir=model_dir("xgboost"))
    results["XGBoost"] = metrics_xgb
    print(f"✅ Done XGBoost | MAE: {metrics_xgb['MAE']:.4f}, RMSE: {metrics_xgb['RMSE']:.4f}")

    print("\n⚡ Running LSTM...")
    _, _, metrics_lstm = LSTMModel(train_df, test_df, save_dir=model_dir("lstm"))
    results["LSTM"] = metrics_lstm
    print(f"✅ Done LSTM | MAE: {metrics_lstm['MAE']:.4f}, RMSE: {metrics_lstm['RMSE']:.4f}")

    # ---- Collect Results ----
    print("\n📊 Collecting metrics...")
    df_results = pd.DataFrame(results).T
    metrics_path = os.path.join(save_dir, "metrics")
    os.makedirs(metrics_path, exist_ok=True)

    df_results.to_csv(os.path.join(metrics_path, "model_comparison.csv"))
    print(f"📁 Metrics saved to {metrics_path}/model_comparison.csv")

    # ---- Bar Plot ----
    df_results.plot(kind="bar", figsize=(10, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Error")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_path, "comparison_plot.png"))
    plt.close()
    print(f"📈 Comparison plot saved to {metrics_path}/comparison_plot.png")

    # ---- Leaderboard ----
    leaderboard = df_results.sort_values("RMSE")
    print("\n🏆 Leaderboard (sorted by RMSE):")
    print(leaderboard)

    return leaderboard