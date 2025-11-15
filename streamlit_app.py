import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

# ------------------- CONFIG -------------------
API_URL = "http://localhost:8000/forecast"   # Change if deployed


# ------------------- STREAMLIT UI -------------------
st.set_page_config(
    page_title="Stock Forecasting App",
    layout="wide"
)

st.title("📈 Stock Price Forecast – 30 Day Predictive Models")

st.sidebar.header("Model Configuration")
model_name = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["xgboost", "arima", "neuralprophet", "lstm", "naive_baseline"]
)

run_button = st.sidebar.button("Generate Forecast")


# ------------------- FETCH FORECAST -------------------
def get_forecast(model_name):
    url = f"{API_URL}/{model_name}"
    response = requests.get(url)

    if response.status_code != 200:
        return None, f"❌ API Error: {response.text}"

    data = response.json()

    if "error" in data:
        return None, f"❌ Model Error: {data['error']}"

    return data, None


# ------------------- RUN FORECAST -------------------
if run_button:
    with st.spinner("Fetching forecast..."):
        data, error = get_forecast(model_name)

    if error:
        st.error(error)
    else:
        st.success(f"Forecast generated using **{model_name.upper()}**")

        forecast_df = pd.DataFrame(data["forecast"])
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

        st.subheader("📄 30-Day Forecast Output")
        st.dataframe(forecast_df, use_container_width=True)

        # ------------------- PLOT FORECAST -------------------
        st.subheader("📊 Forecast Visualization")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast"],
            mode="lines+markers",
            name=f"{model_name.upper()} Forecast",
            line=dict(color="orange")
        ))

        fig.update_layout(
            title=f"30-Day Forecast – {model_name.upper()}",
            xaxis_title="Date",
            yaxis_title="Predicted Price",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download Button
        csv = forecast_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Forecast CSV",
            csv,
            f"{model_name}_forecast.csv",
            "text/csv"
        )


# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown(
    "<center>💡 Built with FastAPI + Streamlit · Stock Forecasting Framework</center>",
    unsafe_allow_html=True
)
