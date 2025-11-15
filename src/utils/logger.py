import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Import model functions
from src.models.naive_baseline import NaiveBaselineModel
from src.models.arima import ArimaModel
from src.models.neuralprophet import NeuralProphetModel
from src.models.xgboost import XGBoostModel
from src.models.lstm import LSTMModel


def evaluate_all_models(
    train_df,
    test_df,
    save_dir="../data/models",
    models_to_run=("naive", "arima", "neuralprophet", "xgboost", "lstm"),
):
    """
    Train & evaluate only selected models.
    
    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    save_dir : str
        Directory to save results
    models_to_run : tuple or list
        Example:
            ("arima", "xgboost")
            ("naive", "lstm")
            ("xgboost",)
    """

    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # Helper for model-specific folder
    def model_dir(name):
        return os.path.join(save_dir, name)

    # Mapping model keywords → functions
    MODEL_FUNCTIONS = {
        "naive":      ("Naive Baseline", NaiveBaselineModel),
        "arima":      ("ARIMA", ArimaModel),
        "neuralprophet": ("NeuralProphet", NeuralProphetModel),
        "xgboost":    ("XGBoost", XGBoostModel),
        "lstm":       ("LSTM", LSTMModel),
    }

    # Validate the model list
    for m in models_to_run:
        if m.lower() not in MODEL_FUNCTIONS:
            raise ValueError(f"Unknown model '{m}'. Valid options = {list(MODEL_FUNCTIONS.keys())}")

    # ---- Run Requested Models Only ----
    for key in models_to_run:
        name, func = MODEL_FUNCTIONS[key]
        print(f"\n⚡ Running {name}...")

        _, _, metrics = func(train_df, test_df, save_dir=model_dir(key.lower()))
        results[name] = metrics

        print(f"✅ Done {name} | "
              f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

    # ---- Collect Results ----
    print("\n📊 Collecting metrics...")
    df_results = pd.DataFrame(results).T
    metrics_path = os.path.join(save_dir, "metrics")
    os.makedirs(metrics_path, exist_ok=True)

    df_results.to_csv(os.path.join(metrics_path, "model_comparison.csv"))
    print(f"📁 Metrics saved to {metrics_path}/model_comparison.csv")

    # ---- Plot ----
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
