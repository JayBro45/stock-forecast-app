import csv
import os

def log_metrics(model_name, mae, rmse, filepath="../data/models/mertics.csv"):
    metrics = {
        "model": model_name,
        "MAE": round(mae,4),
        "RMSE": round(rmse,4)
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"Metrics logged to {filepath}")
