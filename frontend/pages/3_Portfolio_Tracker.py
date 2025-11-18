import streamlit as st
import yfinance as yf
import pandas as pd
import os

st.set_page_config(page_title="Portfolio Tracker", layout="wide")

DATA_FILE = "frontend/portfolio_data.csv"

# --------------------------
# Load / Save Portfolio Data
# --------------------------
def load_portfolio():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["Ticker", "Quantity", "BuyPrice"])

def save_portfolio(df):
    df.to_csv(DATA_FILE, index=False)


# --------------------------
# Live Price Fetcher
# --------------------------
def get_live_price(ticker):
    try:
        price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        return float(price)
    except:
        return None


st.title("📊 Portfolio Tracker")
st.markdown("Manage your investments and track live performance.")

portfolio = load_portfolio()

# --------------------------
# ADD STOCK FORM
# --------------------------
with st.expander("➕ Add a Stock to Portfolio"):
    ticker = st.text_input("Ticker Symbol", "")
    qty = st.number_input("Quantity", min_value=1, step=1)
    buy_price = st.number_input("Buy Price (per share)", min_value=0.0)

    if st.button("Add to Portfolio"):
        if ticker == "":
            st.error("Ticker cannot be empty.")
        else:
            new_row = pd.DataFrame([[ticker.upper(), qty, buy_price]],
                                   columns=["Ticker", "Quantity", "BuyPrice"])
            portfolio = pd.concat([portfolio, new_row], ignore_index=True)
            save_portfolio(portfolio)
            st.success(f"Added {ticker.upper()} to portfolio!")


# --------------------------
# DISPLAY PORTFOLIO TABLE
# --------------------------
st.subheader("📄 Current Portfolio")

if len(portfolio) == 0:
    st.info("Your portfolio is empty. Add some stocks!")
    st.stop()

# Fetch live prices
live_prices = []
for ticker in portfolio["Ticker"]:
    price = get_live_price(ticker)
    live_prices.append(price)

portfolio["LivePrice"] = live_prices
portfolio["CurrentValue"] = portfolio["Quantity"] * portfolio["LivePrice"]
portfolio["Invested"] = portfolio["Quantity"] * portfolio["BuyPrice"]
portfolio["PnL"] = portfolio["CurrentValue"] - portfolio["Invested"]
portfolio["ReturnPct"] = (portfolio["PnL"] / portfolio["Invested"]) * 100

st.dataframe(portfolio.style.format({
    "BuyPrice": "{:.2f}",
    "LivePrice": "{:.2f}",
    "CurrentValue": "{:.2f}",
    "Invested": "{:.2f}",
    "PnL": "{:.2f}",
    "ReturnPct": "{:.2f}%"
}))


# --------------------------
# PORTFOLIO SUMMARY
# --------------------------
total_value = portfolio["CurrentValue"].sum()
total_invested = portfolio["Invested"].sum()
total_pnl = portfolio["PnL"].sum()
total_pct = (total_pnl / total_invested) * 100 if total_invested else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Invested", f"${total_invested:.2f}")
col2.metric("Current Value", f"${total_value:.2f}")
col3.metric("Profit / Loss", f"${total_pnl:.2f}",
            f"{total_pct:.2f}%")
col4.metric("Number of Stocks", len(portfolio))


# --------------------------
# REMOVE STOCK
# --------------------------
with st.expander("🗑 Remove Stock"):
    remove_ticker = st.selectbox("Select Ticker", portfolio["Ticker"].unique())

    if st.button("Remove"):
        portfolio = portfolio[portfolio["Ticker"] != remove_ticker]
        save_portfolio(portfolio)
        st.warning(f"Removed {remove_ticker} from portfolio!")
        st.experimental_rerun()
