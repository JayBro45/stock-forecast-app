import streamlit as st
from components.navbar import navbar
from components.utils import get_forecast
import plotly.graph_objects as go

navbar("Forecast")

st.title("🔮 Model Forecast")

model = st.selectbox("Choose your model:", ["xgboost", "arima", "naive_baseline", "neuralprophet", "lstm"])

if st.button("Get Forecast"):
    data = get_forecast(model)
    
    if "error" in data:
        st.error(data["error"])
    else:
        st.success(f"Forecast loaded for {model}")

        df = data["forecast"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[row["Date"] for row in df],
            y=[row["Forecast"] for row in df],
            mode="lines+markers",
            name="Forecast"
        ))

        st.plotly_chart(fig, use_container_width=True)
