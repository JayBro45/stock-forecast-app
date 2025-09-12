import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class NaiveBaseline:
    def __init__(self):
        self.last_observation = None

    def fit(self, train: pd.Series):
        self.last_observation = train.iloc[-1]
        
    def predict(self, test_index) -> np.ndarray:
        if self.last_observation is None:
            raise ValueError("Model has not been fitted yet.")
        return np.full(shape=len(test_index), fill_value=self.last_observation)
    
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae}
