from fastapi import FastAPI
from src.serve.routers import forecast, health

app = FastAPI(
    title="Stock Forecast API",
    description="Serve multiple time-series models for stock price prediction",
    version="1.0.0"
)

# Routers
app.include_router(health.router)
app.include_router(forecast.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Stock Forecast API"}
