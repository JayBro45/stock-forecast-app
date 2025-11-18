import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import os

# -------------------------------------------------------------
# 🔧 Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Forecast Dashboard",
    layout="wide",
    page_icon="📈"
)

# Path base (frontend/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------
# 🎨 Load Custom CSS
# -------------------------------------------------------------
css_path = os.path.join(BASE_DIR, "assets", "styles.css")

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found: {css_path}")

# Fallback card styling (in case CSS missing)
st.markdown("""
<style>
.price-card {
    background: #1e1e1e;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #333;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 🟩 Helper - Fetch Live Price
# -------------------------------------------------------------
def get_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")

        if data.empty:
            return None, None, None

        price = data["Close"].iloc[-1]

        info = stock.info
        prev_close = info.get("previousClose", price)

        change = price - prev_close
        pct = (change / prev_close) * 100 if prev_close else 0

        return price, change, pct

    except Exception:
        return None, None, None

# -------------------------------------------------------------
# 📊 Sidebar Stock Selector
# -------------------------------------------------------------
st.sidebar.header("📊 Stock Selector")

default_stocks = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NVDA"]

selected_stock = st.sidebar.selectbox("Choose a Stock", default_stocks + ["Custom..."])

if selected_stock == "Custom...":
    selected_stock = st.sidebar.text_input("Enter Custom Ticker", "").upper()

    # prevent empty input crash
    if selected_stock.strip() == "":
        st.sidebar.error("Please enter a valid ticker to proceed.")
        st.stop()

st.sidebar.success(f"Selected: {selected_stock}")

# -------------------------------------------------------------
# 🟢 Live Stock Price
# -------------------------------------------------------------
st.markdown("## 🟢 Live Stock Price")

price, change, pct = get_live_price(selected_stock)

if price is not None:
    color = "green" if change >= 0 else "red"
    arrow = "▲" if change >= 0 else "▼"

    st.markdown(f"""
    <div class="price-card">
        <h2>{selected_stock}</h2>
        <h1>${price:.2f}</h1>
        <p style="color:{color}; font-size:18px;">
            {arrow} {change:.2f} ({pct:.2f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Failed to fetch live price. Invalid ticker?")

# -------------------------------------------------------------
# 📈 Recent Price History (6 months)
# -------------------------------------------------------------
try:
    history = yf.Ticker(selected_stock).history(period="6mo")
    if not history.empty:
        st.line_chart(history["Close"])
except:
    st.warning("Unable to load historical chart.")

# -------------------------------------------------------------
# 🤖 AI Forecast Section
# -------------------------------------------------------------
st.markdown("---")
st.markdown("## 🤖 AI Forecast (Next 30 Days)")

model_choice = st.selectbox(
    "Select Model",
    ["xgboost", "arima", "lstm", "neuralprophet", "naive_baseline"]
)

if st.button("Generate Forecast"):
    with st.spinner("Fetching forecast..."):
        try:
            # 🔥 NOW includes ticker parameter
            url = f"http://localhost:8000/forecast/{model_choice}?ticker={selected_stock}"

            resp = requests.get(url)
            data = resp.json()

            if "error" in data:
                st.error(data["error"])
            else:
                df_forecast = pd.DataFrame(data["forecast"])

                # Fix datetime + sorting
                df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
                df_forecast = df_forecast.sort_values("Date")

                st.success("Forecast Generated Successfully!")
                st.line_chart(df_forecast.set_index("Date")["Forecast"])

                st.dataframe(df_forecast)

        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.markdown("Made with ❤️ using FastAPI + Streamlit + XGBoost + LSTM + ARIMA")
