import requests

API_URL = "http://localhost:8000"

def get_forecast(model_name):
    url = f"{API_URL}/forecast/{model_name}"
    return requests.get(url).json()

def get_model_metrics():
    url = f"{API_URL}/metrics"
    return requests.get(url).json()
