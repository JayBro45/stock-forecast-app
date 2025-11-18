import streamlit as st
from components.navbar import navbar
from components.cards import metric_card
import plotly.graph_objects as go
import requests

st.set_page_config(layout="wide")
navbar("Dashboard")

st.title("📊 Market Overview")

col1, col2, col3 = st.columns(3)
metric_card("Latest Close Price", "$330.12")
metric_card("Trend", "Bullish", "#2196F3")
metric_card("Volatility", "Low", "#FF9800")

# Sample plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[1,2,3,4],
    y=[10,12,15,13],
    mode='lines',
    name='Sample'
))

st.plotly_chart(fig, use_container_width=True)
