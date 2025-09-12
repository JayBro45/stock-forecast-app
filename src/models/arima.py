import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ArimaModel:
    def __init__(self, order=(6,3,1)):
        self.order = order
        self.model_fit = None

    def fit(self, series: pd.Series):
        model = ARIMA(series, order=self.order)
        self.model_fit = model.fit()

    def predict(self, steps):
        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def score(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae}